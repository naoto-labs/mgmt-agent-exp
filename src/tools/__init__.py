"""
Shared Tools Framework - 初期化と自動登録
"""

import logging
from typing import Dict, List

from . import shared_tools  # BaseToolとRegistryのインポート
from .shared_tools import shared_tools_registry

# 各ツールカテゴリのインポート
try:
    from .market_tools.search_products import (
        MarketAnalysisTool,
        SearchProductsTool,
        SupplierResearchTool,
    )

    MARKET_TOOLS_AVAILABLE = True
except ImportError as e:
    MARKET_TOOLS_AVAILABLE = False
    logging.warning(f"Market tools import failed: {e}")

# その他のカテゴリツールのインポート
# from .customer_tools.customer_service import ...
# from .procurement_tools.procurement import ...
# from .data_retrieval.inventory import ...

logger = logging.getLogger(__name__)


def initialize_tools_registry() -> bool:
    """Shared Tools Registryの初期化とすべてツールの自動登録"""
    try:
        logger.info("Initializing Shared Tools Registry...")

        tools_registered = 0

        # Market Toolsの登録
        if MARKET_TOOLS_AVAILABLE:
            try:
                shared_tools_registry.register_tool(SearchProductsTool())
                shared_tools_registry.register_tool(SupplierResearchTool())
                shared_tools_registry.register_tool(MarketAnalysisTool())
                tools_registered += 3
                logger.info("Registered 3 market tools")
            except Exception as e:
                logger.error(f"Market tools registration failed: {e}")

        # Management Toolsの登録 (現在はSessionBasedManagementAgentのメソッドを使用)
        # 後でmanagement_agentから抽出して個別ツール化予定

        # Customer Toolsの登録
        # TODO: customer_agentからToolクラス作成

        # Procurement Toolsの登録
        # TODO: procurement_agentからToolクラス作成

        # Data Retrieval Toolsの登録
        # TODO: inventory_serviceからToolクラス作成

        # Analytics Toolsの登録
        # TODO: analytics_agentから追加分析Tool作成

        # Recorder Toolsの登録
        # TODO: recorder_agentからToolクラス作成

        # Registry統計取得
        registry_stats = shared_tools_registry.get_registry_stats()
        logger.info(
            f"Shared Tools Registry initialized with {registry_stats['total_tools']} tools"
        )

        return True

    except Exception as e:
        logger.error(f"Shared Tools Registry initialization failed: {e}")
        return False


def get_available_tools(agent_type: str = None) -> Dict[str, List[str]]:
    """利用可能なツール一覧取得（カテゴリ別）"""
    if agent_type:
        # Agentタイプ別の利用可能カテゴリを取得
        from .shared_tools import AGENT_ACCESS_LEVELS

        allowed_categories = AGENT_ACCESS_LEVELS.get(agent_type, [])

        tools_by_category = {}
        for category in allowed_categories:
            category_tools = shared_tools_registry.get_tools_by_category(
                category, agent_type
            )
            if category_tools:
                tools_by_category[category] = list(category_tools.keys())

        return tools_by_category
    else:
        # 全ツール取得
        registry_stats = shared_tools_registry.get_registry_stats()
        return {
            cat: registry_stats.get("categories", {}).get(cat, 0)
            for cat in shared_tools.TOOL_CATEGORIES.keys()
        }


def get_tools_stats(agent_type: str = None) -> Dict:
    """ツール使用統計取得"""
    return shared_tools_registry.get_registry_stats()


# モジュール初期化時にRegistryを初期化
logger.info("Initializing Shared Tools Framework...")
success = initialize_tools_registry()

if success:
    logger.info("✅ Shared Tools Framework initialized successfully")
    # 利用可能なツール一覧をログ出力
    available_tools = get_available_tools()
    if available_tools:
        logger.info(f"Available tool categories: {list(available_tools.keys())}")
    else:
        logger.warning("⚠️ No tools available - Agent functionality may be limited")
else:
    logger.error("❌ Shared Tools Framework initialization failed")
    logger.error("This may impact Agent functionality - check tool configurations")
