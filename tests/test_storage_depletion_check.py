#!/usr/bin/env python3
"""
STORAGE枯渇問題の詳細診断テスト

長期間のシミュレーションでSTORAGEが枯渇する原因を特定
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.agents.management_agent.agent import management_agent
from src.agents.management_agent.models import ManagementState
from src.application.services.inventory_service import inventory_service
from src.domain.models.inventory import InventoryLocation, InventorySlot
from src.domain.models.product import Product, ProductCategory, ProductSize


async def setup_test_inventory():
    """テスト用の在庫環境を初期化（STORAGEを制限された量で開始）"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 STORAGE枯渇テスト用在庫環境を初期化...")

    # 商品データを設定
    test_products = [
        Product(
            product_id="cola_regular",
            name="コカ・コーラ レギュラー",
            description="美味しい炭酸飲料",
            category=ProductCategory.DRINK,
            price=150.0,
            cost=100.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
        Product(
            product_id="cola_diet",
            name="コカ・コーラ ダイエット",
            description="カロリーオフの炭酸飲料",
            category=ProductCategory.DRINK,
            price=150.0,
            cost=100.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
    ]

    # グローバル製品データを更新
    import src.domain.models.product as product_module

    product_module.SAMPLE_PRODUCTS = test_products

    # VENDING_MACHINE在庫（満杯状態から開始）
    test_inventory_slots = [
        InventorySlot(
            slot_id=f"VM001_{product.product_id}",
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id=product.product_id,
            product_name=product.name,
            price=product.price,
            current_quantity=50,  # 満杯
            max_quantity=50,
            min_quantity=5,
            slot_number=i + 1,
        )
        for i, product in enumerate(test_products)
    ]

    # STORAGE在庫（制限された量 - 問題を再現するため）
    storage_stock_quantity = 30  # 制限されたSTORAGE（枯渇しやすくする）
    test_storage_slots = [
        InventorySlot(
            slot_id=f"STORAGE_{product.product_id}",
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id=product.product_id,
            product_name=product.name,
            price=product.price,
            current_quantity=storage_stock_quantity,
            max_quantity=100,  # STORAGEはより多く保持可能
            min_quantity=10,
            slot_number=i + 1,
        )
        for i, product in enumerate(test_products)
    ]

    # 在庫サービスをクリアして再初期化
    inventory_service.vending_machine_slots = {}
    inventory_service.storage_slots = {}

    for slot in test_inventory_slots:
        inventory_service.add_slot(slot)

    for slot in test_storage_slots:
        inventory_service.add_slot(slot)

    logger.info(
        f"✅ STORAGE枯渇テスト用環境初期化完了: VENDING_MACHINE={len(test_inventory_slots)}, STORAGE={len(test_storage_slots)}"
    )
    logger.info(f"STORAGE初期量: {storage_stock_quantity}個ずつ")

    return test_products


async def simulate_extended_sales(days: int = 3):
    """長期間の販売をシミュレートしてSTORAGE枯渇を再現"""
    logger = logging.getLogger(__name__)
    logger.info(f"🏪 {days}日分の延長販売シミュレーション開始")

    total_sales = 0
    daily_reports = []

    for day in range(days):
        logger.info(f"\n📅 Day {day + 1}/{days}")
        daily_sales = 0

        # 1日分の販売をシミュレート（多く販売してSTORAGE消費を加速）
        sales_per_day = 5  # 1日5件の販売（現実的な自動販売機の1日販売数）

        # 販売可能な商品を取得
        available_products = []
        for slot in inventory_service.vending_machine_slots.values():
            if slot.current_quantity > 0:
                available_products.append(slot.product_id)

        if not available_products:
            logger.warning(f"Day {day + 1}: 販売可能な商品がありません")
            break

        # 指定数の販売を実行
        for i in range(sales_per_day):
            if not available_products:
                logger.warning(
                    f"Day {day + 1}: 在庫切れにより販売停止 ({i}/{sales_per_day}件)"
                )
                break

            product_id = available_products[0]  # 最初の商品を優先的に販売
            success, message = inventory_service.dispense_product(product_id, 1)

            if success:
                daily_sales += 1
                total_sales += 1
            else:
                # 販売失敗時はその商品をリストから除去
                if product_id in available_products:
                    available_products.remove(product_id)
                logger.debug(f"販売失敗: {message}")

        # 販売後の在庫補充をシミュレート（STORAGEからVENDING_MACHINEへ移動）
        restock_performed = await perform_inventory_restock()

        # 日次レポート作成
        daily_report = {
            "day": day + 1,
            "sales": daily_sales,
            "restock_performed": restock_performed,
            "inventory_status": await get_inventory_snapshot(),
        }
        daily_reports.append(daily_report)

        logger.info(
            f"Day {day + 1} 完了: 販売{daily_sales}件, 補充{len(restock_performed)}件"
        )

    logger.info(f"✅ 延長販売シミュレーション完了: 総販売数 {total_sales}件")
    return daily_reports


async def perform_inventory_restock():
    """在庫補充を実行（STORAGE → VENDING_MACHINE）"""
    restock_actions = []

    # 低在庫のVENDING_MACHINE商品を特定
    low_stock_products = []
    for slot in inventory_service.vending_machine_slots.values():
        if slot.current_quantity < 20:  # 20個未満で補充対象
            low_stock_products.append(slot.product_id)

    # 各低在庫商品に対してSTORAGEから補充
    for product_id in low_stock_products:
        # STORAGEに十分な在庫があるか確認
        storage_slots = [
            slot
            for slot in inventory_service.storage_slots.values()
            if slot.product_id == product_id and slot.current_quantity >= 10
        ]

        if storage_slots:
            # 補充を実行
            success, message = inventory_service.transfer_to_vending_machine(
                product_id,
                10,  # 10個ずつ補充
            )

            if success:
                restock_actions.append(f"{product_id}: 成功")
                logging.info(f"補充成功: {message}")
            else:
                restock_actions.append(f"{product_id}: 失敗 - {message}")
                logging.warning(f"補充失敗: {message}")
        else:
            restock_actions.append(f"{product_id}: STORAGE不足")
            logging.warning(f"STORAGE在庫不足: {product_id}")

    return restock_actions


async def get_inventory_snapshot():
    """現在の在庫状況のスナップショットを取得"""
    snapshot = {}

    # 全スロットの集計
    all_slots = list(inventory_service.vending_machine_slots.values()) + list(
        inventory_service.storage_slots.values()
    )

    for slot in all_slots:
        product_id = slot.product_id
        location = (
            "VENDING"
            if slot.location == InventoryLocation.VENDING_MACHINE
            else "STORAGE"
        )

        if product_id not in snapshot:
            snapshot[product_id] = {
                "product_name": slot.product_name,
                "vending_stock": 0,
                "storage_stock": 0,
                "total_stock": 0,
            }

        if location == "VENDING":
            snapshot[product_id]["vending_stock"] += slot.current_quantity
        else:
            snapshot[product_id]["storage_stock"] += slot.current_quantity

        snapshot[product_id]["total_stock"] = (
            snapshot[product_id]["vending_stock"]
            + snapshot[product_id]["storage_stock"]
        )

    return snapshot


async def analyze_depletion_pattern(daily_reports):
    """STORAGE枯渇パターンを分析"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 STORAGE枯渇パターン分析開始")

    print("\n" + "=" * 80)
    print("📊 STORAGE枯渇パターン分析レポート")
    print("=" * 80)

    depletion_events = []
    critical_days = []

    for report in daily_reports:
        day = report["day"]
        sales = report["sales"]
        restock_actions = report["restock_performed"]
        inventory = report["inventory_status"]

        print(f"\n📅 Day {day}:")
        print(f"  販売数: {sales}件")
        print(f"  補充アクション: {len(restock_actions)}件")

        # 商品別在庫状況を表示
        for product_id, data in inventory.items():
            vending = data["vending_stock"]
            storage = data["storage_stock"]
            total = data["total_stock"]

            status = (
                "✅ 正常"
                if vending > 10 and storage > 20
                else "🟡 注意"
                if vending > 5 or storage > 10
                else "❌ 危機"
            )

            print(
                f"  {data['product_name']:<12}: 自販機{vending:>2}個, STORAGE{storage:>2}個, 合計{total:>2}個 {status}"
            )

            # 枯渇イベントの検出
            if storage == 0 and vending < 20:
                depletion_events.append(
                    {
                        "day": day,
                        "product": data["product_name"],
                        "vending_remaining": vending,
                        "critical": True,
                    }
                )

        # 全体的な危機状況判定
        total_vending = sum(data["vending_stock"] for data in inventory.values())
        total_storage = sum(data["storage_stock"] for data in inventory.values())

        if total_storage == 0:
            critical_days.append(day)
            print("  💥 STORAGE完全枯渇!")
        elif total_storage < 20:
            print("  ⚠️ STORAGE枯渇間近")
        elif total_vending < 50:
            print("  🟡 自販機在庫全体が少ない")

    # 分析結果サマリー
    print(f"\n🎯 分析結果サマリー:")
    print(f"  シミュレーション日数: {len(daily_reports)}日")
    print(f"  枯渇イベント数: {len(depletion_events)}件")
    print(f"  STORAGE完全枯渇日: {len(critical_days)}日")

    if depletion_events:
        print(f"  枯渇商品リスト:")
        for event in depletion_events:
            print(
                f"    - Day {event['day']}: {event['product']} (自販機残り{event['vending_remaining']}個)"
            )

    # 根本原因推定
    print("\n🔍 根本原因分析:")
    if len(critical_days) > 0:
        print("  ❌ 問題: STORAGEが完全に枯渇し、補充が機能していない")
        print("  💡 推定原因: 調達プロセスがSTORAGEへ商品を届ける処理が不足")
    elif len(depletion_events) > len(daily_reports) * 0.5:
        print("  ⚠️ 問題: STORAGE枯渇が頻発")
        print("  💡 推定原因: 調達タイミングが遅いか、補充量が不足")
    else:
        print("  ✅ STORAGE枯渇は発生していない")
        print("  💡 補充システムは正常動作中")

    logger.info("✅ STORAGE枯渇パターン分析完了")


async def test_procurement_trigger():
    """調達トリガーのテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 調達トリガーテスト開始")

    # STORAGE在庫を意図的に低く設定
    for slot in inventory_service.storage_slots.values():
        slot.current_quantity = 5  # 最低限に設定

    print("\nSTORAGEを故意に低在庫状態に設定")

    # 現在の在庫状況を表示
    await get_inventory_snapshot()

    # 調達トリガーをテスト（実際のManagementAgent調達ノードを呼び出し）
    try:
        from src.agents.management_agent.agent import NodeBasedManagementAgent

        agent = NodeBasedManagementAgent()

        # 調達が必要な状態を作成
        state = ManagementState(
            session_id="procurement_test_001",
            inventory_analysis={
                "low_stock_items": [
                    "cola_regular",
                    "cola_diet",
                ],
                "reorder_needed": [
                    "cola_regular",
                    "cola_diet",
                ],
                "estimated_stockout": {
                    "cola_regular": "枯渇間近",
                    "cola_diet": "枯渇間近",
                },
            },
        )

        logger.info("調達ノード実行を開始...")
        # 調達ノードを呼び出し（LLM発注判断ノードを使用）
        result_state = await agent.procurement_request_generation_node(state)

        logger.info("調達ノード実行完了")

        # 調達結果を確認
        executed_actions = (
            result_state.executed_actions
            if hasattr(result_state, "executed_actions")
            else []
        )
        procurement_actions = [
            a for a in executed_actions if "procurement" in a.get("type", "")
        ]

        print(f"調達アクション実行数: {len(procurement_actions)}")

        if procurement_actions:
            for action in procurement_actions:
                print(f"  - {action}")
        else:
            print("  調達アクションが実行されませんでした")

    except Exception as e:
        logger.error(f"調達トリガーテスト失敗: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 STORAGE枯渇問題診断テスト")
    print("=" * 50)
    print("ステップ1: 制限されたSTORAGE環境初期化")
    print("ステップ2: 延長販売シミュレーション")
    print("ステップ3: 枯渇パターン分析")
    print("ステップ4: 調達トリガーテスト")
    print("=" * 50)

    try:
        # ステップ1: 制限されたSTORAGE環境初期化
        print("\n🔧 ステップ1: STORAGE制限環境初期化...")
        products = await setup_test_inventory()

        # 初期状態確認
        initial_snapshot = await get_inventory_snapshot()
        print("\n初期在庫状態:")
        for product_id, data in initial_snapshot.items():
            print(
                f"  {data['product_name']}: 自販機{data['vending_stock']}個, STORAGE{data['storage_stock']}個"
            )

        # ステップ2: 延長販売シミュレーション
        print("\n🏪 ステップ2: 延長販売シミュレーション...")
        daily_reports = await simulate_extended_sales(days=3)

        # ステップ3: 分析
        print("\n🔍 ステップ3: 枯渇パターン分析...")
        await analyze_depletion_pattern(daily_reports)

        # ステップ4: 調達トリガーテスト
        print("\n🔄 ステップ4: 調達トリガーテスト...")
        await test_procurement_trigger()

        print("\n" + "=" * 50)
        print("🎯 STORAGE枯渇問題診断完了")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
