#!/usr/bin/env python3
"""
在庫切れ問題の包括的診断テスト

STORAGEも含めた在庫を初期化し、その後の可視化と診断を行う
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


async def setup_inventory():
    """テスト用の在庫環境を初期化"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 テスト用在庫環境を初期化...")

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
        Product(
            product_id="water_mineral",
            name="ミネラルウォーター",
            description="爽やかなミネラルウォーター",
            category=ProductCategory.DRINK,
            price=120.0,
            cost=80.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
        Product(
            product_id="energy_drink",
            name="エナジードリンク",
            description="元気が出るドリンク",
            category=ProductCategory.DRINK,
            price=180.0,
            cost=120.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
        Product(
            product_id="snack_chips",
            name="ポテトチップス",
            description="サクサクのスナック",
            category=ProductCategory.SNACK,
            price=180.0,
            cost=120.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
        Product(
            product_id="snack_chocolate",
            name="チョコレートバー",
            description="甘いチョコレート",
            category=ProductCategory.SNACK,
            price=160.0,
            cost=110.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
    ]

    # グローバル製品データを更新
    import src.domain.models.product as product_module

    product_module.SAMPLE_PRODUCTS = test_products

    # 在庫初期化（低在庫状態から開始して、補充テストを可能にする）
    initial_stock_quantity = 8  # 低めで開始
    test_inventory_slots = [
        InventorySlot(
            slot_id=f"VM001_{product.product_id}",
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id=product.product_id,
            product_name=product.name,
            price=product.price,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=i + 1,
        )
        for i, product in enumerate(test_products)
    ]

    # STORAGE在庫（十分なストック）
    storage_stock_quantity = 100
    test_storage_slots = [
        InventorySlot(
            slot_id=f"STORAGE_{product.product_id}",
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id=product.product_id,
            product_name=product.name,
            price=product.price,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=20,
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
        f"✅ 低在庫状態で初期化完了: VENDING_MACHINE={len(test_inventory_slots)}, STORAGE={len(test_storage_slots)}"
    )

    return test_products


async def simulate_sales_and_check_inventory(sales_count: int = 3):
    """販売をシミュレートして在庫変化を観察"""
    logger = logging.getLogger(__name__)
    logger.info(f"🛒 {sales_count}件の販売をシミュレート...")

    # 販売可能な商品を取得
    available_products = []
    for slot in inventory_service.vending_machine_slots.values():
        if slot.current_quantity > 0:
            available_products.append(slot.product_id)

    if not available_products:
        logger.warning("販売可能な商品がありません")
        return 0

    # ランダムな商品を複数回販売
    import random

    successful_sales = 0

    for i in range(sales_count):
        if not available_products:
            logger.warning("販売可能な商品がなくなりました")
            break

        product_id = random.choice(available_products)
        success, message = inventory_service.dispense_product(product_id, 1)

        if success:
            successful_sales += 1
            logger.info(f"販売{i + 1}: {message}")
        else:
            logger.warning(f"販売{i + 1}失敗: {message}")
            # 失敗した商品をリストから削除
            if product_id in available_products:
                available_products.remove(product_id)

    logger.info(f"✅ 販売完了: {successful_sales}/{sales_count}件成功")
    return successful_sales


async def analyze_inventory_issues():
    """在庫問題の包括的分析"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 STORAGE + VENDING_MACHINE 総合在庫分析")

    # 全在庫データを取得
    all_inventory = inventory_service.get_inventory_by_location()
    vending_inventory = all_inventory.get("vending_machine", [])
    storage_inventory = all_inventory.get("storage", [])

    print("\n" + "=" * 80)
    print("📊 在庫分析レポート")
    print("=" * 80)

    print(f"📦 在庫状況全体:")
    print(f"  VENDING MACHINE: {len(vending_inventory)}スロット")
    print(f"  STORAGE: {len(storage_inventory)}スロット")
    print(f"  総スロット数: {len(vending_inventory) + len(storage_inventory)}")

    # 商品別集計
    product_summary = {}

    for slot in vending_inventory + storage_inventory:
        product_name = slot.product_name
        location = "VENDING" if slot in vending_inventory else "STORAGE"

        if product_name not in product_summary:
            product_summary[product_name] = {
                "product_id": slot.product_id,
                "vending_qty": 0,
                "storage_qty": 0,
                "total_qty": 0,
                "slots_vending": 0,
                "slots_storage": 0,
            }

        if location == "VENDING":
            product_summary[product_name]["vending_qty"] += slot.current_quantity
            product_summary[product_name]["slots_vending"] += 1
        else:
            product_summary[product_name]["storage_qty"] += slot.current_quantity
            product_summary[product_name]["slots_storage"] += 1

        product_summary[product_name]["total_qty"] = (
            product_summary[product_name]["vending_qty"]
            + product_summary[product_name]["storage_qty"]
        )

    print("\n📋 商品別詳細在庫:")
    print("-" * 80)
    print(
        f"{'商品名':<15} {'VENDING':>8} {'STORAGE':>8} {'合計':>8} {'スロット':<10} {'ステータス'}"
    )
    print("-" * 80)

    critical_items = []
    low_stock_items = []
    storage_only_items = []
    needs_restock = []

    for product_name, data in product_summary.items():
        vending_qty = data["vending_qty"]
        storage_qty = data["storage_qty"]
        total_qty = data["total_qty"]

        # ステータス判定
        if vending_qty == 0 and storage_qty == 0:
            status = "❌ 完全欠品"
            critical_items.append(product_name)
        elif vending_qty == 0 and storage_qty > 0:
            status = "⚠️  STORAGEのみ"
            storage_only_items.append(product_name)
        elif vending_qty < 15:  # 在庫が少ない閾値
            status = "🟡 在庫不足"
            low_stock_items.append(product_name)
            if storage_qty >= 10:  # 補充可能
                needs_restock.append(product_name)
        else:
            status = "✅ 正常"

        vending_slots = data["slots_vending"]
        storage_slots = data["slots_storage"]
        slot_info = f"V:{vending_slots}/S:{storage_slots}"

        print(
            f"{product_name:<15} {vending_qty:>8} {storage_qty:>8} {total_qty:>8} {slot_info:<10} {status}"
        )

    print("\n🚨 問題状況サマリー:")
    print(f"  - 完全欠品商品: {len(critical_items)}個 {critical_items}")
    print(f"  - 在庫不足商品: {len(low_stock_items)}個 {low_stock_items}")
    print(f"  - STORAGEのみ商品: {len(storage_only_items)}個 {storage_only_items}")
    print(f"  - 補充推奨商品: {len(needs_restock)}個 {needs_restock}")
    print(f"  - 全商品数: {len(product_summary)}個")

    # 補充処理分析
    print("\n🔄 補充可能性分析:")

    total_vending_stock = sum(data["vending_qty"] for data in product_summary.values())
    total_storage_stock = sum(data["storage_qty"] for data in product_summary.values())

    print(f"  自販機総在庫量: {total_vending_stock}")
    print(f"  STORAGE総在庫量: {total_storage_stock}")
    if total_vending_stock > 0:
        ratio = (total_storage_stock / total_vending_stock) * 100
        print(f"  STORAGE/VENDING比率: {ratio:.1f}% (理想: 100-200%)")

    # 補充推奨詳細
    if needs_restock:
        print(f"\n📦 補充推奨項目:")
        for product_name in needs_restock:
            data = product_summary[product_name]
            needed = 20 - data["vending_qty"]  # 目標在庫20個
            available = min(needed, data["storage_qty"])
            print(f"  - {product_name}: 需要{needed}個, 移動可能{available}個")

    # 根本原因分析
    print("\n🔍 根本原因分析:")
    issues_found = []

    if storage_only_items:
        issues_found.append("STORAGEに在庫があるのにVENDING_MACHINEにない商品あり")
        print(f"  ⚠️ 補充漏れ: {', '.join(storage_only_items)}")

    if critical_items:
        issues_found.append("完全に在庫切れの商品あり")
        print(f"  ❌ 欠品危機: {', '.join(critical_items)}")

    if not storage_inventory:
        issues_found.append("STORAGE自体に何も入っていない")
        print("  💥 STORAGEが空 - 調達プロセスが機能していない可能性")
    elif total_storage_stock < 50:
        issues_found.append("STORAGEの総在庫が非常に少ない")
        print("  ⚠️ STORAGE在庫が少ない - 調達が追いついていない")

    if low_stock_items and not needs_restock:
        issues_found.append("在庫が少ない商品があるがSTORAGEに補充元がない")
        print("  🔄 補充ソース不足")

    if not issues_found:
        print("  ✅ 重大な問題は検出されませんでした")

    return {
        "critical_items": critical_items,
        "storage_only_items": storage_only_items,
        "low_stock_items": low_stock_items,
        "needs_restock": needs_restock,
        "product_summary": product_summary,
        "issues": issues_found,
    }


async def test_restock_process():
    """補充プロセスをテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 補充プロセステスト開始")

    # 現在の状態を分析
    before_analysis = await analyze_inventory_issues()
    restock_candidates = before_analysis["needs_restock"]

    if not restock_candidates:
        logger.info("⏭️ 補充が必要な商品なし - テストスキップ")
        return {"success": True, "message": "補充不要"}

    logger.info(f"📦 補充対象商品: {', '.join(restock_candidates)}")

    # 補充プロセス実行（特定の商品を1つ選んでテスト）
    test_product = restock_candidates[0]
    logger.info(f"🧪 テスト補充実行: {test_product}")

    # 補充を実行
    success, message = inventory_service.transfer_to_vending_machine(
        before_analysis["product_summary"][test_product]["product_id"], 10
    )

    if success:
        logger.info(f"✅ 補充成功: {message}")

        # 補充後の状態確認
        print("\n--- 補充後状態 ---")
        after_analysis = await analyze_inventory_issues()

        # 変化を検証
        before_qty = before_analysis["product_summary"][test_product]["vending_qty"]
        after_qty = after_analysis["product_summary"][test_product]["vending_qty"]
        change = after_qty - before_qty

        logger.info(
            f"🔍 補充検証: {test_product} - {before_qty} → {after_qty} (変化: {change})"
        )

        return {
            "success": True,
            "message": f"補充成功 - {change}個増加",
            "change": change,
        }
    else:
        logger.error(f"❌ 補充失敗: {message}")
        return {"success": False, "message": message}


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 在庫切れ問題総合診断テスト")
    print("=======================================")
    print("ステップ1: 在庫初期化 (低状態)")
    print("ステップ2: 販売シミュレーション")
    print("ステップ3: 問題分析")
    print("ステップ4: 補充テスト")
    print("=======================================")

    try:
        # ステップ1: 在庫初期化
        print("\n🔧 ステップ1: 在庫初期化...")
        products = await setup_inventory()

        # 初期状態確認
        await analyze_inventory_issues()

        # ステップ2: 販売シミュレート（在庫不足状態を作る）
        print("\n🛒 ステップ2: 販売シミュレーション...")
        sales_count = await simulate_sales_and_check_inventory(15)  # 15件の販売

        # 販売後の状態確認
        print("\n--- 販売後状態 ---")
        await analyze_inventory_issues()

        # ステップ3: 補充プロセステスト
        print("\n🔄 ステップ3: 補充プロセステスト...")
        restock_result = await test_restock_process()

        # 最終診断
        print("\n" + "=" * 60)
        print("🎯 最終診断結果")
        print("=" * 60)

        if restock_result["success"]:
            print("✅ 補充プロセス: 正常動作")
            if "change" in restock_result and restock_result["change"] > 0:
                print(f"✅ 在庫移動: {restock_result['change']}個増加を確認")
            else:
                print("ℹ️ 補充不要だったため変更なし")
        else:
            print(f"❌ 補充プロセス: 失敗 - {restock_result['message']}")
            print("💥 STORAGEからVENDING_MACHINEへの補充が機能していない可能性")

        print("\n📋 テスト完了サマリー:")
        print("- 在庫初期化: 成功")
        print(f"- 販売シミュレーション: {sales_count}件成功")
        print(f"- 補充テスト: {'成功' if restock_result['success'] else '失敗'}")

        print("\n🎉 在庫切れ問題診断完了")
        print("=======================================")

        # 最終状態を返す
        return {
            "inventory_setup": "success",
            "sales_simulation": f"{sales_count}_sales",
            "restock_test": "success" if restock_result["success"] else "failed",
        }

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
