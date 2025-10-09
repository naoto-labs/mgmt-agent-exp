"""
request_procurement.py - 調達依頼ツール

補充必要品の発注依頼生成、サプライヤ連絡Tool
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger(__name__)


def request_procurement(
    products: List[str],
    quantity: Dict[str, int],
    supplier_costs: Dict[str, float] = None,
) -> Dict[str, Any]:
    """調達を依頼"""
    logger.info(f"Requesting procurement for {products}")
    order_id = str(uuid4())

    # 仕入コストが指定されていない場合はデフォルト値を使用
    if supplier_costs is None:
        from src.application.services.inventory_service import get_product_by_id

        supplier_costs = {}
        for product in products:
            product_obj = get_product_by_id(product)
            supplier_costs[product] = product_obj.cost if product_obj else 100.0

    return {
        "order_id": order_id,
        "products": products,
        "quantity": quantity,
        "supplier_costs": supplier_costs,  # 仕入原価情報追加
        "total_cost": sum(
            supplier_costs[p] * quantity[p] for p in products if p in supplier_costs
        ),
        "status": "pending",
        "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
    }


def complete_procurement(order: Dict[str, Any]) -> Dict[str, Any]:
    """調達完了処理：STORAGEに投入し原価を登録"""
    logger.info(f"Completing procurement for order {order['order_id']}")

    from src.application.services.inventory_service import (
        get_product_by_id,
        inventory_service,
    )
    from src.domain.accounting.journal_entry import journal_processor

    try:
        # 各商品をSTORAGEに投入し、原価を登録
        completed_items = {}

        for product_id in order["products"]:
            quantity = order["quantity"][product_id]
            supplier_cost = order["supplier_costs"][product_id]

            # 1. STORAGEスロットに商品を保管
            success, message = inventory_service.restock_to_storage(
                product_id, quantity, supplier_cost
            )
            if not success:
                logger.error(f"Failed to restock to storage: {message}")
                continue

            # 2. 会計仕訳：実際の仕入原価で記録
            product_obj = get_product_by_id(product_id)
            if product_obj:
                # 商品情報を更新（実際の仕入原価を反映）
                product_obj.cost = supplier_cost
                product_obj.update_cost_data()

                # 仕入仕訳記録
                journal_entries = journal_processor.record_purchase(
                    product=product_obj,
                    quantity=quantity,
                    supplier_name="Default Supplier",
                    order_id=order["order_id"],
                )

                completed_items[product_id] = {
                    "quantity": quantity,
                    "supplier_cost": supplier_cost,
                    "actual_cost": supplier_cost * quantity,
                    "journal_entries": len(journal_entries) if journal_entries else 0,
                }

                logger.info(
                    f"Procurement completed: {product_id} x{quantity} at ¥{supplier_cost:.0f}/unit"
                )

        order["status"] = "completed"
        order["completion_time"] = datetime.now().isoformat()
        order["completed_items"] = completed_items

        logger.info(f"Procurement completion successful for order {order['order_id']}")
        return {"success": True, "order": order}

    except Exception as e:
        logger.error(f"Procurement completion failed: {e}")
        return {"success": False, "error": str(e)}


def simulate_procurement_delay(
    order: Dict[str, Any], delay_days: int = 3
) -> Dict[str, Any]:
    """調達遅延のシミュレーション"""
    logger.info(f"Simulating procurement delay for {delay_days} days")
    order["status"] = "delayed"
    order["delay_days"] = delay_days
    order["new_delivery"] = (datetime.now() + timedelta(days=delay_days)).isoformat()
    return order
