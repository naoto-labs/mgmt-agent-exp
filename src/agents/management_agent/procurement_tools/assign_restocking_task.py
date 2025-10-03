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
    products: List[str], urgency: str = "normal"
) -> Dict[str, Any]:
    """補充タスクを割り当て"""
    logger.info("Tool assign_restocking called")
    logger.info(f"Assigning restocking task for {products} with urgency {urgency}")
    task_id = str(uuid4())
    return {
        "task_id": task_id,
        "task_type": "restocking",
        "products": products,
        "urgency": urgency,
        "assigned": True,
        "deadline": (
            datetime.now() + timedelta(hours=2 if urgency == "urgent" else 24)
        ).isoformat(),
    }
