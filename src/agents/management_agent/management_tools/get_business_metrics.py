"""
get_business_metrics.py - ビジネスメトリクス取得ツール

売上、在庫、顧客データをシステムから取得するTool
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict

from src.application.services.inventory_service import inventory_service
from src.domain.accounting.management_accounting import management_analyzer

logger = logging.getLogger(__name__)


def get_business_metrics() -> Dict[str, Any]:
    """ビジネスメトリクスを取得（実際のシステムと連携）"""
    logger.info("Getting business metrics from actual systems")

    try:
        # 在庫情報を取得
        inventory_summary = inventory_service.get_inventory_summary()
        inventory_level = {}

        # 商品別在庫を集計
        for slot in inventory_service.vending_machine_slots.values():
            product_name = slot.product_name.lower()
            if product_name not in inventory_level:
                inventory_level[product_name] = 0
            inventory_level[product_name] += slot.current_quantity

        # 財務情報を取得（管理会計から）
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        # 売上情報を取得（会計システムから）
        sales = abs(
            management_analyzer.journal_processor.get_account_balance(
                "4001", start_date, end_date
            )
        )

        period_profitability = management_analyzer.analyze_period_profitability(
            start_date, end_date
        )
        profit_margin = period_profitability.get("gross_margin", 0.35)

        # 顧客満足度の計算
        # 在庫充足率と売上実績から推定
        total_inventory = sum(inventory_level.values())
        max_inventory = (
            len(inventory_service.vending_machine_slots) * 50
        )  # 想定最大在庫
        inventory_score = (
            min(total_inventory / max_inventory, 1.0) if max_inventory > 0 else 0.5
        )

        # 売上目標との比較（月間目標: 100万円）
        monthly_target = 1000000
        sales_score = min(sales / monthly_target, 1.0)

        # 総合満足度（3.0-5.0のスケール）
        customer_satisfaction = 3.0 + (inventory_score * 1.0 + sales_score * 1.0)

        metrics_result = {
            "sales": round(sales, 2),
            "profit_margin": round(profit_margin, 3),
            "inventory_level": inventory_level,
            "customer_satisfaction": round(customer_satisfaction, 2),
            "timestamp": datetime.now().isoformat(),
            "inventory_status": {
                "total_slots": len(inventory_service.vending_machine_slots),
                "low_stock_count": len(inventory_service.get_low_stock_slots()),
                "out_of_stock_count": len(inventory_service.get_out_of_stock_slots()),
            },
            "sales_stats": {
                "total_revenue": sales,  # 会計システムから取得
            },
        }

        # デバッグログ: 取得したビジネスデータをログ出力
        logger.debug("=== BUSINESS METRICS RETRIEVED ===")
        logger.debug(f"Sales (accounting_system): ¥{sales:.2f}")
        logger.debug(f"Profit Margin: {profit_margin:.3f}")
        logger.debug(f"Inventory Level: {inventory_level}")
        logger.debug(f"Inventory Status: {metrics_result['inventory_status']}")
        logger.debug(f"Customer Satisfaction: {customer_satisfaction:.2f}")
        logger.debug("=== END BUSINESS METRICS ===")

        return metrics_result

    except Exception as e:
        logger.error(f"ビジネスメトリクス取得エラー: {e}", exc_info=True)
        # エラー時はフォールバック値を返す
        return {
            "sales": 0.0,
            "profit_margin": 0.0,
            "inventory_level": {},
            "customer_satisfaction": 3.0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
