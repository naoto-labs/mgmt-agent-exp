"""
Shared Tools Framework - BaseToolとRegistryの実装
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Tool実行結果"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float
    timestamp: datetime


class BaseTool(ABC):
    """Shared Toolの基底抽象クラス"""

    def __init__(self, tool_id: str, category: str, agent_access: list = None):
        self.tool_id = tool_id
        self.category = category
        self.agent_access = agent_access or [
            "management"
        ]  # デフォルトでmanagementのみアクセス
        self.usage_count = 0
        self.last_used = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """ツール実行の抽象メソッド"""
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """入力パラメータ検証"""
        pass

    def can_access(self, agent_type: str) -> bool:
        """Agentアクセス権限チェック"""
        if not self.agent_access or "*" in self.agent_access:
            return True  # 全アクセス許可
        return agent_type in self.agent_access

    def log_execution(self, success: bool):
        """実行ログ記録"""
        self.usage_count += 1
        self.last_used = datetime.now()
        logger.info(
            f"Tool {self.tool_id} executed by {self.agent_access}: success={success}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """ツール統計情報"""
        return {
            "tool_id": self.tool_id,
            "category": self.category,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class ToolRegistry:
    """Shared Toolsの集中管理Registry"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)

    def register_tool(self, tool: BaseTool):
        """ツールをRegistryに登録"""
        if tool.tool_id in self._tools:
            self.logger.warning(f"Tool {tool.tool_id} already registered, replacing")
            # 既存のものを置き換え
            self._tools[tool.tool_id] = tool
        else:
            self._tools[tool.tool_id] = tool

        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.tool_id not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.tool_id)

        self.logger.info(
            f"Registered tool: {tool.tool_id} in category: {tool.category}"
        )

    def unregister_tool(self, tool_id: str) -> bool:
        """ツールをRegistryから削除"""
        if tool_id not in self._tools:
            self.logger.warning(f"Tool {tool_id} not found in registry")
            return False

        tool = self._tools[tool_id]
        category = tool.category

        # Registryから削除
        del self._tools[tool_id]

        # カテゴリリストからも削除
        if tool_id in self._categories[category]:
            self._categories[category].remove(tool_id)

        self.logger.info(f"Unregistered tool: {tool_id}")
        return True

    def get_tool(self, tool_id: str, agent_type: str = None) -> Optional[BaseTool]:
        """ツール取得（アクセス権限チェック付き）"""
        tool = self._tools.get(tool_id)
        if not tool:
            self.logger.warning(f"Tool {tool_id} not found in registry")
            return None

        if agent_type and not tool.can_access(agent_type):
            self.logger.warning(f"Access denied: {tool_id} for agent: {agent_type}")
            return None

        return tool

    def get_tools_by_category(
        self, category: str, agent_type: str = None
    ) -> Dict[str, BaseTool]:
        """カテゴリ別ツール一括取得"""
        tool_ids = self._categories.get(category, [])
        tools = {}
        for tool_id in tool_ids:
            tool = self.get_tool(tool_id, agent_type)
            if tool:
                tools[tool_id] = tool
        return tools

    def get_all_tools(self, agent_type: str = None) -> Dict[str, BaseTool]:
        """全ツール取得"""
        return {
            tool_id: self.get_tool(tool_id, agent_type)
            for tool_id in self._tools.keys()
        }
        # Noneの場合は含めない

    def execute_tool(
        self, tool_id: str, agent_type: str = "management", **kwargs
    ) -> ToolResult:
        """ツール実行（Registry経由）"""
        tool = self.get_tool(tool_id, agent_type)
        if not tool:
            return ToolResult(
                success=False,
                error_message="Tool not accessible",
                execution_time=0,
                timestamp=datetime.now(),
            )

        if not tool.validate_input(**kwargs):
            return ToolResult(
                success=False,
                error_message="Invalid input parameters",
                execution_time=0,
                timestamp=datetime.now(),
            )

        start_time = datetime.now()
        try:
            result = tool.execute(**kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.timestamp = start_time

            tool.log_execution(result.success)
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            tool.log_execution(False)
            return ToolResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                timestamp=start_time,
            )

    def get_registry_stats(self) -> Dict[str, Any]:
        """Registry全体の統計情報"""
        stats = {
            "total_tools": len(self._tools),
            "categories": {},
            "total_usage": sum(tool.usage_count for tool in self._tools.values()),
            "tool_details": {},
        }

        for category, tool_ids in self._categories.items():
            stats["categories"][category] = len(tool_ids)

        for tool_id, tool in self._tools.items():
            stats["tool_details"][tool_id] = tool.get_stats()

        return stats

    def get_tool_categories(self) -> List[str]:
        """使用可能なカテゴリ一覧"""
        return list(self._categories.keys())


# グローバルRegistryインスタンス
shared_tools_registry = ToolRegistry()

# Toolカテゴリの定数定義
TOOL_CATEGORIES = {
    "data_retrieval": "データ取得ツール",
    "customer_tools": "顧客対応ツール",
    "procurement_tools": "調達管理ツール",
    "market_tools": "市場分析ツール",
    "management_tools": "経営判断ツール",
    "analytics_tools": "分析ツール",
    "recorder_tools": "記録ツール",
}

# Agentアクセス権限の定数定義
AGENT_ACCESS_LEVELS = {
    "management": [
        "data_retrieval",
        "customer_tools",
        "procurement_tools",
        "market_tools",
        "management_tools",
    ],
    "analytics": ["data_retrieval", "analytics_tools", "market_tools"],
    "recorder": ["data_retrieval", "recorder_tools", "customer_tools"],
}
