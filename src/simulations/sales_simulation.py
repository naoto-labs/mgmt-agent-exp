"""
販売イベントシミュレーションモジュール

価格考慮型購入シミュレーションを実装
"""

import asyncio
import logging
import math
import random
from datetime import date

from src.accounting.journal_entry import journal_processor
from src.models.product import SAMPLE_PRODUCTS
from src.models.transaction import PaymentMethod
from src.services.inventory_service import inventory_service
from src.services.payment_service import payment_service

logger = logging.getLogger(__name__)


def sample_poisson(lambda_val: float) -> int:
    """ポアソン分布からサンプリング"""
    L = math.exp(-lambda_val)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def generate_customer_budget() -> float:
    """顧客の予算を生成（安い商品を好む傾向）"""
    # 予算分布: 多くが低価格帯
    budget_models = {
        "low": (80, 120),  # 60% 低予算
        "mid": (120, 180),  # 30% 中予算
        "high": (180, 250),  # 10% 高予算
    }

    choice = random.choices(list(budget_models.keys()), weights=[60, 30, 10])[0]
    min_b, max_b = budget_models[choice]
    return random.uniform(min_b, max_b)


async def simulate_purchase_events(
    sales_lambda: float, verbose: bool = True, period_name: str = "販売"
) -> dict:
    """
    購入イベントをシミュレート（価格考慮）

    Args:
        sales_lambda: 平均購入イベント数（ポアソン分布）
        verbose: 詳細出力するか
        period_name: 期間名（ログ用）

    Returns:
        シミュレーション結果辞書
    """
    num_events = sample_poisson(sales_lambda)
    total_revenue = 0.0
    successful_sales = 0

    if verbose:
        logger.info(f"{period_name}イベント ({num_events}件予測)")

    for i in range(num_events):
        # 顧客予算生成
        customer_budget = generate_customer_budget()

        # 予算内商品を選定
        affordable_products = [
            p
            for p in SAMPLE_PRODUCTS
            if p.is_available() and p.price <= customer_budget
        ]

        if not affordable_products:
            if verbose:
                logger.debug(
                    f"  イベント{i + 1}: 予算内で購入可能商品なし (予算: ¥{customer_budget:.0f})"
                )
            continue

        # 価格重み付き選択（安い方が高い確率で選ばれる）
        # weight = (budget - price) / budget + 0.1  で安い方が有利
        weights = [
            max(0.1, (customer_budget - p.price) / customer_budget + 0.1)
            for p in affordable_products
        ]

        product = random.choices(affordable_products, weights=weights)[0]

        # 支払い方法選択
        valid_payment_methods = [PaymentMethod.CARD, PaymentMethod.CASH]
        payment_method = random.choice(valid_payment_methods)

        # 決済処理
        result = await payment_service.process_payment(product.price, payment_method)

        if result.success:
            # 会計記録: 売上収入
            journal_processor.add_entry(
                account_number="1001",  # 現金
                date=date.today(),
                amount=product.price,
                entry_type="debit",
                description=f"商品販売: {product.name} x1 - {payment_method.value}",
            )
            journal_processor.add_entry(
                account_number="4001",  # 売上高
                date=date.today(),
                amount=product.price,
                entry_type="credit",
                description=f"商品販売: {product.name} x1 - {payment_method.value}",
            )

            # 在庫減算
            inventory_service.dispense_product(product.product_id, 1)

            # 販売データ更新
            product.sales_count += 1
            product.update_sales_data()

            successful_sales += 1
            total_revenue += product.price

            if verbose:
                logger.info(
                    f"  イベント{i + 1}: {product.name} x1 - {payment_method.value} - ¥{product.price} (予算¥{customer_budget:.0f})"
                )
        else:
            if verbose:
                logger.warning(f"  イベント{i + 1}: {product.name} - 支払い失敗")

    # 統計計算
    average_budget = (
        sum(generate_customer_budget() for _ in range(10)) / 10
        if num_events > 0
        else 0.0
    )

    result_stats = {
        "total_events": num_events,
        "successful_sales": successful_sales,
        "total_revenue": total_revenue,
        "conversion_rate": successful_sales / num_events if num_events > 0 else 0.0,
        "average_budget": average_budget,
    }

    if verbose:
        logger.info(
            f"{period_name}完了: {successful_sales}/{num_events} 成功 ({result_stats['conversion_rate']:.1%}) 売上¥{total_revenue:.0f}"
        )

    return result_stats
