"""
management_tools - 経営管理ツール群の集約モジュール
"""

from typing import List

from langchain.tools import StructuredTool

from .analyze_financial_performance import analyze_financial_performance

# 個別のツール関数をインポート
from .get_business_metrics import get_business_metrics
from .plan_agent_operations import plan_agent_operations
from .plan_sales_strategy import plan_sales_strategy
from .update_pricing import update_pricing


def create_management_tools() -> List[StructuredTool]:
    """
    経営管理ツール群をStructuredToolリストとして作成

    Returns:
        List[StructuredTool]: 経営管理領域の全ツールリスト
    """
    tools = []

    # ビジネスメトリクス取得ツール
    tools.append(
        StructuredTool.from_function(
            func=get_business_metrics,
            name="get_business_data",
            description="売上、在庫、顧客データをシステムから取得",
        )
    )

    # 財務パフォーマンス分析ツール
    tools.append(
        StructuredTool.from_function(
            func=analyze_financial_performance,
            name="analyze_financials",
            description="財務実績を分析し、収益性を評価",
        )
    )

    # 最新更新ツール (関数として直接使う)
    tools.append(
        StructuredTool.from_function(
            func=update_pricing,
            name="update_pricing",
            description="価格戦略を決定し、システムに反映",
        )
    )

    # agent運用計画ツール
    tools.append(
        StructuredTool.from_function(
            func=plan_agent_operations,
            name="plan_agent_operations",
            description="Agentの運用戦略を計画・提案",
        )
    )

    # 売上戦略計画ツール
    tools.append(
        StructuredTool.from_function(
            func=plan_sales_strategy,
            name="plan_sales_strategy",
            description="売上向上のための戦略を立案",
        )
    )

    return tools
