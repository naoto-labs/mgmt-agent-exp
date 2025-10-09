"""
販売イベントシミュレーションモジュール

価格考慮型購入シミュレーションを実装
"""

import asyncio
import logging
import math
import random
import time
import uuid
from datetime import date

from src.application.services.inventory_service import inventory_service
from src.application.services.payment_service import payment_service
from src.domain.accounting.journal_entry import journal_processor
from src.domain.models.product import (
    SAMPLE_PRODUCTS,
    Product,
    ProductCategory,
    ProductSize,
)


def get_available_products():
    """在庫サービスから利用可能な商品リストを取得"""
    # 在庫スロットに存在する商品に基づいて商品リストを作成
    available_products = []

    for slot in list(inventory_service.vending_machine_slots.values()) + list(
        inventory_service.storage_slots.values()
    ):
        # 既存の商品リストに含まれていなければ、新しくProductオブジェクトを作成
        if not any(p.product_id == slot.product_id for p in available_products):
            # スロットの情報からProductオブジェクトを作成
            product = Product(
                product_id=slot.product_id,
                name=slot.product_name,
                description=f"{slot.product_name}商品",
                category=ProductCategory.DRINK
                if slot.product_id.startswith(("cola", "water", "energy"))
                else ProductCategory.SNACK,
                price=slot.price,
                cost=slot.price * 0.7,  # 仮定のコスト：価格の70%
                stock_quantity=slot.current_quantity,
                max_stock_quantity=slot.max_quantity,
                min_stock_quantity=slot.min_quantity,
                size=ProductSize.MEDIUM,
            )
            available_products.append(product)

    return available_products


from src.domain.models.transaction import PaymentMethod

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
    # 予算分布: 現実的な購買行動に基づく配分（余裕を持たせた修正版）
    budget_models = {
        "low": (150, 220),  # 30% 低予算 - 安い商品（ソフトドリンク）対応 余裕を持たせる
        "mid": (200, 280),  # 50% 中予算 - 標準的な購買層
        "high": (280, 400),  # 20% 高予算 - 高級商品・複数個購入層
    }

    choice = random.choices(list(budget_models.keys()), weights=[30, 50, 20])[0]
    min_b, max_b = budget_models[choice]
    return random.uniform(min_b, max_b)


async def simulate_purchase_events(
    sales_lambda: float,
    verbose: bool = True,
    period_name: str = "販売",
    dry_run: bool = False,
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
    # 最低1イベントを保証してコンバージョン0を防ぐ
    num_events = max(sample_poisson(sales_lambda), 1)
    total_revenue = 0.0
    successful_sales = 0

    if verbose:
        logger.info(f"{period_name}イベント ({num_events}件予測)")

    for i in range(num_events):
        # 顧客予算生成
        customer_budget = generate_customer_budget()

        # 利用可能な商品リストを取得
        available_products = get_available_products()

        # 予算内商品を選定（在庫サービスから利用可能かをチェック）
        logger.debug(
            f"購入イベント{i + 1}: 顧客予算¥{customer_budget:.0f}, 利用可能な全商品数={len(available_products)}"
        )
        affordable_products = [
            p
            for p in available_products
            if inventory_service.is_product_available(p.product_id)
        ]
        logger.debug(
            f"購入イベント{i + 1}: 在庫利用可能商品数={len(affordable_products)}"
        )
        if not affordable_products:
            logger.warning(
                f"購入イベント{i + 1}: 在庫利用可能商品が1つもない！ 全商品チェック:"
            )
            for p in available_products:
                available = inventory_service.is_product_available(p.product_id)
                logger.warning(f"  商品{p.product_id} ({p.name}): 利用可能={available}")

        # 在庫サービスから最新価格を取得して予算チェック（フォールバック付き）
        affordable_products_with_prices = []
        for p in affordable_products:
            # まず在庫サービスから価格取得
            price = inventory_service.get_product_price(p.product_id)

            # 在庫サービスで価格が取得できない場合、スロットから価格を取得
            if price is None:
                # スロット情報を取得して価格を確認
                slots = inventory_service.get_product_slots(p.product_id)
                if slots:
                    # 最初の利用可能スロットの価格を使用
                    available_slots = [slot for slot in slots if slot.is_available()]
                    if available_slots:
                        price = available_slots[0].price
                        logger.debug(f"スロットから価格取得 {p.product_id}: ¥{price}")
                    else:
                        price = p.price
                        logger.debug(
                            f"利用可能スロットなし {p.product_id}、SAMPLE_PRODUCTSから価格使用: ¥{price}"
                        )
                else:
                    price = p.price
                    logger.debug(
                        f"スロットなし {p.product_id}、SAMPLE_PRODUCTSから直接取得: ¥{price}"
                    )

            # 価格チェック
            if price <= customer_budget:
                affordable_products_with_prices.append((p, price))

        # affordable_productsを更新
        affordable_products = [p for p, price in affordable_products_with_prices]

        if not affordable_products:
            if verbose:
                logger.debug(
                    f"  イベント{i + 1}: 予算内で購入可能商品なし (予算: ¥{customer_budget:.0f})"
                )
            # 予算に合う商品がない場合、最も安い商品以上に予算を調整
            available_prices = [
                p.price
                for p in available_products
                if inventory_service.is_product_available(p.product_id)
            ]

            if available_prices:
                # 最も安い利用可能商品の価格に予算を調整
                min_price = min(available_prices)
                if customer_budget < min_price:
                    customer_budget = min_price
                    logger.debug(
                        f"  予算を調整: ¥{customer_budget:.0f} (最も安い商品に合わせ: ¥{min_price})"
                    )

                # 予算調整後に再度選定（在庫サービスから最新価格取得）
                affordable_products = []
                for p in available_products:
                    if inventory_service.is_product_available(p.product_id):
                        price = (
                            inventory_service.get_product_price(p.product_id) or p.price
                        )
                        if price <= customer_budget:
                            affordable_products.append(p)
                            affordable_products_with_prices.append((p, price))
            else:
                # 利用可能商品が存在しない場合、このイベントをスキップ
                if verbose:
                    logger.debug(f"  イベント{i + 1}: 利用可能商品なし、スリップ")
                continue

        # 価格重み付き選択（安い方が高い確率で選ばれる）
        # affordable_products_with_pricesから商品と価格のペアで重み計算
        product_price_pairs = [
            (p, price)
            for p, price in affordable_products_with_prices
            if p in affordable_products
        ]

        if not product_price_pairs:
            if verbose:
                logger.debug(
                    f"  イベント{i + 1}: 予算内で購入可能商品が見つかりません (予算: ¥{customer_budget:.0f})"
                )
            continue

        weights = [
            max(0.1, (customer_budget - price) / customer_budget + 0.1)
            for _, price in product_price_pairs
        ]

        # 重み付き選択
        selected_pair = random.choices(product_price_pairs, weights=weights)[0]
        product, current_price = selected_pair

        # 支払い方法選択
        valid_payment_methods = [PaymentMethod.CARD, PaymentMethod.CASH]
        payment_method = random.choice(valid_payment_methods)

        # 実際の決済処理
        result = await payment_service.process_payment(current_price, payment_method)

        if result.success:
            # 会計記録: 売上収入（重複チェック付き）
            transaction_id = f"sale_{i}_{uuid.uuid4().hex}"

            journal_processor.add_entry(
                account_number="1001",  # 現金
                date=date.today(),
                amount=current_price,
                entry_type="debit",
                description=f"商品販売: {product.name} x1 - {payment_method.value}",
                transaction_id=transaction_id,
            )
            journal_processor.add_entry(
                account_number="4001",  # 売上高
                date=date.today(),
                amount=current_price,
                entry_type="credit",
                description=f"商品販売: {product.name} x1 - {payment_method.value}",
                transaction_id=transaction_id,
            )

            # 在庫減算
            inventory_service.dispense_product(product.product_id, 1)

            # 販売データ更新
            product.sales_count += 1
            product.update_sales_data()

            # 管理エージェントに販売完了通知
            try:
                from src.agents.management_agent.agent import management_agent

                await management_agent.notify_sale_completed(
                    {
                        "product_name": product.name,
                        "price": current_price,
                        "payment_method": payment_method.value,
                        "transaction_id": transaction_id,
                    }
                )
                logger.debug(f"販売完了通知送信: {product.name}")
            except Exception as e:
                logger.warning(f"販売完了通知失敗: {e}")

            successful_sales += 1
            total_revenue += current_price

            if verbose:
                logger.info(
                    f"  イベント{i + 1}: {product.name} x1 - {payment_method.value} - ¥{current_price} (予算¥{customer_budget:.0f})"
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
