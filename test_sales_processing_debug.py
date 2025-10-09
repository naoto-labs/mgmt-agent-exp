#!/usr/bin/env python3
"""
test_sales_processing_debug.py - sales_processing 問題デバッグ用テスト

sales_processing で販売が発生しない理由を特定するための単体テスト
"""

import asyncio
import logging
import sys
from datetime import datetime

# プロジェクトパスを追加
sys.path.append("src")

from src.application.services import inventory_service
from src.simulations.sales_simulation import (
    SAMPLE_PRODUCTS,
    generate_customer_budget,
    sample_poisson,
    simulate_purchase_events,
)

# ログレベルをDEBUGに設定
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_inventory_service_basic():
    """在庫サービスの基本テスト"""
    print("=== 在庫サービス基本テスト ===")

    # 在庫サービス初期化ステータス確認
    print("在庫サービス状態:")
    print(f"  自販機スロット数: {len(inventory_service.vending_machine_slots)}")
    print(f"  保管庫スロット数: {len(inventory_service.storage_slots)}")

    # get_available_products()関数の動作を確認
    from src.simulations.sales_simulation import get_available_products

    available_products = get_available_products()
    print(f"\nget_available_products() 返却商品数: {len(available_products)}")

    for product in available_products:
        available = inventory_service.is_product_available(product.product_id)
        print(
            f"  {product.product_id} ({product.name}): 利用可能={available}, 価格=¥{product.price}"
        )

    # SAMPLE_PRODUCTSは別ものとして確認
    print(f"\nSAMPLE_PRODUCTS 比較 (総数: {len(SAMPLE_PRODUCTS)}):")
    sample_product_ids = {p.product_id for p in SAMPLE_PRODUCTS}
    available_product_ids = {p.product_id for p in available_products}

    overlap = sample_product_ids & available_product_ids
    print(f"  重複商品ID数: {len(overlap)}")
    if len(overlap) == 0:
        print("  → SAMPLE_PRODUCTSとスロット商品は完全に別物（意図通り）")
    else:
        print(f"  → 重複あり: {overlap}")


def test_customer_budget_generation():
    """顧客予算生成テスト"""
    print("\n=== 顧客予算生成テスト ===")

    budgets = [generate_customer_budget() for _ in range(10)]
    print(f"生成された予算 (10回): {['¥{:.0f}'.format(b) for b in budgets]}")
    print(f"平均予算: ¥{sum(budgets) / len(budgets):.0f}")
    print(f"最小予算: ¥{min(budgets):.0f}")
    print(f"最大予算: ¥{max(budgets):.0f}")


def test_poisson_sampling():
    """ポアソン分布サンプリングテスト"""
    print("\n=== ポアソン分布サンプリングテスト ===")

    lambda_val = 5.0
    samples = [sample_poisson(lambda_val) for _ in range(10)]
    print(f"lambda={lambda_val} のサンプル (10回): {samples}")
    print(f"平均: {sum(samples) / len(samples):.1f}")


async def test_simulate_purchase_events_detailed():
    """販売シミュレーション詳細テスト"""
    print("\n=== 販売シミュレーション詳細テスト ===")

    try:
        # 1イベントのみでテスト（詳細ログ確認用）
        result = await simulate_purchase_events(
            sales_lambda=1.0, verbose=True, period_name="デバッグテスト"
        )

        print("\nシミュレーション結果:")
        print(f"  総イベント数: {result['total_events']}")
        print(f"  成功販売数: {result['successful_sales']}")
        print(f"  総売上: ¥{result['total_revenue']:.0f}")
        print(f"  コンバージョン率: {result['conversion_rate']:.1%}")
        print(f"  平均予算: ¥{result['average_budget']:.0f}")

    except Exception as e:
        print(f"シミュレーションエラー: {e}")
        import traceback

        traceback.print_exc()


def test_inventory_existence():
    """在庫存在確認テスト"""
    print("\n=== 在庫存在確認テスト ===")

    print("在庫サービス詳細情報:")
    print(
        f"  vending_machine_slots: {list(inventory_service.vending_machine_slots.keys())}"
    )
    print(f"  storage_slots: {list(inventory_service.storage_slots.keys())}")

    # 全スロットの詳細表示
    print("\n全スロット詳細:")
    all_slots = list(inventory_service.vending_machine_slots.values()) + list(
        inventory_service.storage_slots.values()
    )
    for slot in all_slots:
        print(
            f"  {slot.slot_id}: {slot.product_name} - quantity={slot.current_quantity}, location={slot.location}"
        )


async def main():
    """メイン実行関数"""
    print("=== Sales Processing デバッグテスト開始 ===")
    print(f"実行時刻: {datetime.now()}")

    # 環境初期化（在庫セットアップ）
    print("\n=== 環境初期化 ===")

    try:
        # continuous_multi_day_simulationから環境初期化を使用
        from continuous_multi_day_simulation import setup_simulation_environment

        await setup_simulation_environment()
        print("環境初期化完了")
    except Exception as e:
        print(f"環境初期化失敗: {e}")

    # 各テスト実行
    test_inventory_service_basic()
    test_customer_budget_generation()
    test_poisson_sampling()
    test_inventory_existence()
    await test_simulate_purchase_events_detailed()

    print("\n=== デバッグテスト完了 ===")


if __name__ == "__main__":
    asyncio.run(main())
