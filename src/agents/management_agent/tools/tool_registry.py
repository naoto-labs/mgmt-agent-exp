"""
tool_registry.py - 管理Agent全ツールの統合レジストリ

全カテゴリ(management/customer/procurement)ツールの
一元化管理と設定を提供
"""

from typing import List

from langchain.tools import StructuredTool

from src.agents.management_agent.customer_tools import create_customer_tools

# カテゴリ別ツール集約関数をインポート
from src.agents.management_agent.management_tools import create_management_tools
from src.agents.management_agent.procurement_tools import create_procurement_tools


def create_tool_registry() -> List[StructuredTool]:
    """
    全カテゴリツールを統合したStructuredToolリストを作成

    Returns:
        List[StructuredTool]: 全カテゴリの全ツールリスト
    """
    all_tools = []

    # 経営管理ツール
    all_tools.extend(create_management_tools())

    # 顧客対応ツール
    all_tools.extend(create_customer_tools())

    # 調達・在庫管理ツール
    all_tools.extend(create_procurement_tools())

    return all_tools


# 使用例:
# from src.agents.management_agent.tools.tool_registry import create_tool_registry
# tools = create_tool_registry()
# agent = ManagementAgent(tools=tools)
