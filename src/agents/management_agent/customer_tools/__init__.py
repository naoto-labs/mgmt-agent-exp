"""
customer_tools - 顧客対応ツール群の集約モジュール
"""

from typing import List

from langchain.tools import StructuredTool

from .create_customer_engagement_campaign import create_customer_engagement_campaign
from .handle_customer_complaint import handle_customer_complaint

# 個別のツール関数をインポート
from .respond_to_customer_inquiry import respond_to_customer_inquiry


def create_customer_tools() -> List[StructuredTool]:
    """
    顧客対応ツール群をStructuredToolリストとして作成

    Returns:
        List[StructuredTool]: 顧客対応領域の全ツールリスト
    """
    tools = []

    # 顧客問い合わせ対応ツール
    tools.append(
        StructuredTool.from_function(
            func=respond_to_customer_inquiry,
            name="customer_response",
            description="顧客からの問い合わせに回答",
        )
    )

    # 顧客苦情処理ツール
    tools.append(
        StructuredTool.from_function(
            func=handle_customer_complaint,
            name="handle_complaint",
            description="顧客苦情の処理と解決策提案",
        )
    )

    # 顧客エンゲージメントキャンペーン作成ツール
    tools.append(
        StructuredTool.from_function(
            func=create_customer_engagement_campaign,
            name="create_campaign",
            description="顧客エンゲージメント施策の企画",
        )
    )

    return tools
