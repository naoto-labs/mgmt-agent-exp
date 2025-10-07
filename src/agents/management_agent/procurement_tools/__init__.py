"""
procurement_tools - 調達・在庫管理ツール群の集約モジュール
"""

from typing import List

from langchain.tools import StructuredTool

# 個別のツール関数をインポート
from .assign_restocking_task import assign_restocking_task
from .coordinate_employee_tasks import coordinate_employee_tasks
from .request_procurement import request_procurement


def create_procurement_tools() -> List[StructuredTool]:
    """
    調達・在庫管理ツール群をStructuredToolリストとして作成

    Returns:
        List[StructuredTool]: 調達・在庫管理領域の全ツールリスト
    """
    tools = []

    # 在庫補充タスク割り当てツール
    tools.append(
        StructuredTool.from_function(
            func=assign_restocking_task,
            name="assign_restocking",
            description="従業員に商品補充作業を指示",
        )
    )

    # 従業員タスク調整ツール
    tools.append(
        StructuredTool.from_function(
            func=coordinate_employee_tasks,
            name="coordinate_tasks",
            description="従業員の業務配分と進捗管理",
        )
    )

    # 調達依頼ツール
    tools.append(
        StructuredTool.from_function(
            func=request_procurement,
            name="request_procurement",
            description="担当者に商品調達を依頼",
        )
    )

    return tools
