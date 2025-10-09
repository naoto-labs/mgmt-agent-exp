"""
assign_restocking_task.py - 補充タスク割り当てツール

従業員に商品補充作業を指示Tool
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger(__name__)


def assign_restocking_task(
    products: List[str], urgency: str = "normal", source: str = "storage"
) -> Dict[str, Any]:
    """補充タスクを割り当て（STORAGEからVENDING_MACHINEへの転送）"""
    logger.info("Tool assign_restocking called")
    logger.info(
        f"Assigning restocking task for {products} with urgency {urgency}, source: {source}"
    )
    task_id = str(uuid4())

    task_info = {
        "task_id": task_id,
        "task_type": "restocking",
        "products": products,
        "urgency": urgency,
        "source": source,  # "storage" または "direct"（調達直送）
        "assigned": True,
        "deadline": (
            datetime.now() + timedelta(hours=2 if urgency == "urgent" else 24)
        ).isoformat(),
    }

    return task_info


def execute_restocking_task(task_id: str) -> Dict[str, Any]:
    """補充タスクを実行（STORAGEからVENDING_MACHINEへ移動）"""
    logger.info(f"Executing restocking task: {task_id}")

    from src.application.services.inventory_service import inventory_service

    # タスク情報を格納（本来はDBなどから取得）
    # ここでは簡単のためグローバル変数で管理（本番ではデータベース化）
    if not hasattr(execute_restocking_task, "active_tasks"):
        execute_restocking_task.active_tasks = {}

    # 仮のタスク情報（本番ではDBから取得）
    task_info = {
        "products": ["cola_regular", "cola_diet", "water_mineral"],  # デフォルト商品
        "urgency": "normal",
        "source": "storage",
    }

    completed_transfers = []
    failed_transfers = []

    try:
        for product_id in task_info["products"]:
            # STORAGEの在庫を確認
            storage_inventory = inventory_service.get_total_inventory(product_id)
            storage_stock = storage_inventory["storage_stock"]

            if storage_stock == 0:
                failed_transfers.append(
                    {
                        "product_id": product_id,
                        "error": "STORAGEに在庫がありません",
                        "storage_stock": storage_stock,
                    }
                )
                continue

            # 転送数量を決定（VENDING_MACHINEの空き容量に応じて）
            transfer_quantity = min(storage_stock, 20)  # 最大20個ずつ転送

            # STORAGEからVENDING_MACHINEへ転送
            success, message = inventory_service.transfer_to_vending_machine(
                product_id, transfer_quantity
            )

            if success:
                completed_transfers.append(
                    {
                        "product_id": product_id,
                        "transferred_quantity": transfer_quantity,
                        "message": message,
                    }
                )
                logger.info(f"補充成功: {product_id} x{transfer_quantity}")
            else:
                failed_transfers.append(
                    {
                        "product_id": product_id,
                        "error": message,
                        "storage_stock": storage_stock,
                    }
                )
                logger.warning(f"補充失敗: {product_id} - {message}")

        # 会計処理：移動費用などの記録（必要に応じて）
        # transfer_to_vending_machine は数量移動のみで原価転送は不要

        return {
            "success": True,
            "task_id": task_id,
            "completed_transfers": completed_transfers,
            "failed_transfers": failed_transfers,
            "completion_time": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"補充タスク実行エラー: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "error": str(e),
            "failed_transfers": failed_transfers,
        }


def get_restocking_status() -> Dict[str, Any]:
    """補充タスクの実行状況を取得"""
    from src.application.services.inventory_service import inventory_service

    inventory_report = inventory_service.get_inventory_report()

    return {
        "overall_status": "operational",
        "inventory_summary": inventory_report,
        "last_updated": datetime.now().isoformat(),
    }
