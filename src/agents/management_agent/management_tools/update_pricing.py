"""
update_pricing.py - 価格更新ツール

価格戦略を決定し、システムに反映するTool
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.application.services.inventory_service import inventory_service

logger = logging.getLogger(__name__)


def update_pricing(product_id: str, price: float) -> Dict[str, Any]:
    """価格戦略を更新"""
    logger.info("Tool update_pricing called")
    logger.info(f"Updating pricing for product_id: {product_id} to ¥{price}")

    try:
        # 価格更新の検証
        if price <= 0:
            raise ValueError("価格は正の値である必要があります")

        if price > 10000:  # 過度に高い価格のチェック
            logger.warning(f"非常に高い価格設定を検知: {product_id} -> ¥{price}")

        # 現在の価格を取得
        current_price = inventory_service.get_product_price(product_id)
        if current_price is None:
            raise ValueError(f"商品が見つかりません: {product_id}")

        # InventoryService経由で価格更新
        update_success = inventory_service.update_product_price(product_id, price)

        if update_success:
            # 価格更新処理結果
            update_result = {
                "success": True,
                "product_id": product_id,
                "previous_price": current_price,
                "new_price": price,
                "price_change": price - current_price,
                "effective_date": datetime.now().isoformat(),
                "updated_by": "management_agent",
                "update_reason": "LLMベース価格戦略による自動更新",
            }

            logger.info(f"価格更新成功: {product_id} ¥{current_price} -> ¥{price}")
            return update_result
        else:
            raise ValueError(f"価格更新に失敗しました: {product_id}")

    except Exception as e:
        logger.error(f"価格更新失敗 {product_id}: {e}")
        return {
            "success": False,
            "product_id": product_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
