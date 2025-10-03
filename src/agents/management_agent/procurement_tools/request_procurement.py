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
    products: List[str], quantity: Dict[str, int]
) -> Dict[str, Any]:
    """調達を依頼"""
    logger.info(f"Requesting procurement for {products}")
    order_id = str(uuid4())
    return {
        "order_id": order_id,
        "products": products,
        "quantity": quantity,
        "status": "pending",
        "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
    }
