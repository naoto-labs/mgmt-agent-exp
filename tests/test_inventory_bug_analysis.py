#!/usr/bin/env python3
"""
在庫減少バグの詳細分析テスト

ユーザーの指摘通り、在庫が100%から8個売れただけで0になるバグを再現・分析
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
    """テスト用の在庫環境を初期化（バグ再現用）"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 バグ再現テスト用在庫環境を初期化...")

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
    ]

    # グローバル製品データを更新
    import src.domain.models.product as product_module

    product_module.SAMPLE_PRODUCTS = test_products

    # VENDING_MACHINE在庫（満杯状態から開始）
    test_inventory_slots = [
        InventorySlot(
            slot_id="VM001_cola_regular",
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=50,  # 満杯状態から開始
            max_quantity=50,
            min_quantity=5,
            slot_number=1,
        )
    ]

    # STORAGE在庫（十分な量）
    storage_stock_quantity = 100
    test_storage_slots = [
        InventorySlot(
            slot_id="STORAGE_cola_regular",
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=20,
            slot_number=1,
        )
    ]

    # 在庫サービスをクリアして再初期化
    inventory_service.vending_machine_slots = {}
    inventory_service.storage_slots = {}

    for slot in test_inventory_slots:
        inventory_service.add_slot(slot)

    for slot in test_storage_slots:
        inventory_service.add_slot(slot)

    logger.info(
        f"✅ バグ再現テスト環境初期化完了: VENDING_MACHINE=50個, STORAGE={storage_stock_quantity}個"
    )

    return test_products


async def test_dispense_behavior():
    """dispenseの挙動を詳細にテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 dispense挙動詳細テスト開始")

    print("\n" + "=" * 80)
    print("🔬 在庫排出挙動詳細分析")
    print("=" * 80)

    # 初期状態確認
    initial_inventory = inventory_service.get_total_inventory("cola_regular")
    print("初期状態:")
    print(f"  自販機在庫: {initial_inventory['vending_machine_stock']}個")
    print(f"  STORAGE在庫: {initial_inventory['storage_stock']}個")
    print(f"  合計: {initial_inventory['total_stock']}個")

    # スロット詳細確認
    vending_slots = [
        slot
        for slot in inventory_service.vending_machine_slots.values()
        if slot.product_id == "cola_regular"
    ]

    print(f"\n自販機スロット詳細 ({len(vending_slots)}スロット):")
    for slot in vending_slots:
        print(f"  スロット {slot.slot_id}:")
        print(f"    current_quantity: {slot.current_quantity}")
        print(f"    max_quantity: {slot.max_quantity}")
        print(f"    min_quantity: {slot.min_quantity}")
        print(f"    status: {slot.status}")
        print(f"    is_available(): {slot.is_available()}")

    # 複数回のdispenseをテスト
    print("\n📦 複数回のdispenseテスト:")
    dispense_results = []

    for i in range(10):  # 10回連続でdispense
        success, message = inventory_service.dispense_product("cola_regular", 1)

        # 各回の状態を記録
        current_inventory = inventory_service.get_total_inventory("cola_regular")
        vending_quantity = current_inventory["vending_machine_stock"]

        dispense_results.append(
            {
                "attempt": i + 1,
                "success": success,
                "message": message,
                "vending_quantity": vending_quantity,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        print(
            f"  試行 {i + 1}: {'✅ 成功' if success else '❌ 失敗'} - 自販機残量: {vending_quantity}個"
        )

        if not success:
            print(f"    エラーメッセージ: {message}")
            break

    # 結果分析
    print("\n📊 結果分析:")
    print(f"  総試行回数: {len(dispense_results)}")

    successful_dispenses = [r for r in dispense_results if r["success"]]
    print(f"  成功回数: {len(successful_dispenses)}")

    if successful_dispenses:
        final_quantity = successful_dispenses[-1]["vending_quantity"]
        expected_quantity = 50 - len(successful_dispenses)  # 初期50個から減少を想定

        print(f"  最終自販機在庫: {final_quantity}個")
        print(f"  期待される在庫: {expected_quantity}個")

        if final_quantity != expected_quantity:
            print(f"  🚨 バグ検出! 在庫減少量が異常")
            print(f"     期待: 50 - {len(successful_dispenses)} = {expected_quantity}")
            print(f"     実際: {final_quantity}")
            print(f"     差異: {abs(final_quantity - expected_quantity)}個")
        else:
            print(f"  ✅ 在庫減少は正常")

    # 詳細なスロット状態確認
    print("\n🔍 最終スロット状態:")
    for slot in vending_slots:
        print(f"  スロット {slot.slot_id}:")
        print(f"    current_quantity: {slot.current_quantity}")
        print(f"    total_dispensed: {slot.total_dispensed}")
        print(f"    total_restocked: {slot.total_restocked}")
        print(f"    last_dispensed: {slot.last_dispensed}")
        print(f"    status: {slot.status}")

    return dispense_results


async def test_multiple_dispense_quantities():
    """複数数量のdispenseをテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 複数数量dispenseテスト開始")

    print("\n" + "=" * 80)
    print("🔬 複数数量排出テスト")
    print("=" * 80)

    # 在庫をリセット（満杯状態に）
    for slot in inventory_service.vending_machine_slots.values():
        slot.current_quantity = 50
        slot.total_dispensed = 0

    print("在庫を50個にリセット")

    # 異なる数量でdispenseをテスト
    test_cases = [1, 3, 5, 8, 10]

    for quantity in test_cases:
        print(f"\n📦 {quantity}個同時排出テスト:")

        # テスト前の状態
        before_inventory = inventory_service.get_total_inventory("cola_regular")
        before_quantity = before_inventory["vending_machine_stock"]

        print(f"  排出前: {before_quantity}個")

        # dispense実行
        success, message = inventory_service.dispense_product("cola_regular", quantity)

        # テスト後の状態
        after_inventory = inventory_service.get_total_inventory("cola_regular")
        after_quantity = after_inventory["vending_machine_stock"]

        print(f"  排出後: {after_quantity}個")
        print(f"  結果: {'✅ 成功' if success else '❌ 失敗'}")

        if success:
            expected_quantity = before_quantity - quantity
            if after_quantity == expected_quantity:
                print(
                    f"  ✅ 在庫減少正常: {before_quantity} - {quantity} = {after_quantity}"
                )
            else:
                print(
                    f"  🚨 バグ検出! 期待: {expected_quantity}, 実際: {after_quantity}"
                )
                print(f"     差異: {abs(after_quantity - expected_quantity)}個")
        else:
            print(f"  ❌ 排出失敗: {message}")

    return True


async def test_slot_selection_logic():
    """スロット選択ロジックのテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 スロット選択ロジックテスト開始")

    print("\n" + "=" * 80)
    print("🔍 スロット選択ロジック詳細テスト")
    print("=" * 80)

    # 複数スロットを作成してテスト
    test_slots = []

    # 異なる在庫量のスロットを作成
    for i in range(3):
        slot = InventorySlot(
            slot_id=f"VM001_cola_test_{i}",
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=20 + i * 10,  # 20, 30, 40個
            max_quantity=50,
            min_quantity=5,
            slot_number=i + 1,
        )
        test_slots.append(slot)
        inventory_service.add_slot(slot)

    print("テスト用スロット作成:")
    for slot in test_slots:
        print(f"  スロット {slot.slot_id}: {slot.current_quantity}個")

    # dispense_productの内部ロジックをテスト
    print("\n🎯 dispense_product内部ロジックテスト:")
    # 自販機内の利用可能スロットを取得（dispense_productと同じロジック）
    vending_slots = [
        slot
        for slot in inventory_service.get_product_slots("cola_regular")
        if slot.location == InventoryLocation.VENDING_MACHINE and slot.is_available()
    ]

    print(f"利用可能スロット数: {len(vending_slots)}")

    if vending_slots:
        # 在庫の多いスロットから優先的に使用（dispense_productと同じロジック）
        target_slot = max(vending_slots, key=lambda s: s.current_quantity)

        print(
            f"選択されたスロット: {target_slot.slot_id} ({target_slot.current_quantity}個)"
        )

        # dispense実行
        success, message = inventory_service.dispense_product("cola_regular", 1)

        print(f"dispense結果: {'✅ 成功' if success else '❌ 失敗'}")

        if success:
            print(f"選択スロットの残量: {target_slot.current_quantity}個")
            print(f"選択スロットのtotal_dispensed: {target_slot.total_dispensed}")

    return True


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 在庫減少バグ詳細分析テスト")
    print("=" * 60)
    print("ステップ1: テスト環境初期化 (満杯状態から開始)")
    print("ステップ2: 複数回dispenseテスト")
    print("ステップ3: 複数数量dispenseテスト")
    print("ステップ4: スロット選択ロジックテスト")
    print("=" * 60)

    try:
        # ステップ1: テスト環境初期化
        print("\n🔧 ステップ1: テスト環境初期化...")
        products = await setup_test_inventory()

        # ステップ2: 複数回dispenseテスト
        print("\n📦 ステップ2: 複数回dispenseテスト...")
        dispense_results = await test_dispense_behavior()

        # ステップ3: 複数数量テスト
        print("\n🔢 ステップ3: 複数数量テスト...")
        await test_multiple_dispense_quantities()

        # ステップ4: スロット選択ロジックテスト
        print("\n🎯 ステップ4: スロット選択ロジックテスト...")
        await test_slot_selection_logic()

        print("\n" + "=" * 60)
        print("🎯 在庫減少バグ分析完了")
        print("=" * 60)

        # 最終的なバグ判定
        print("\n🔍 バグ判定結果:")

        # dispense_resultsから異常を検出
        successful_dispenses = [r for r in dispense_results if r["success"]]
        if successful_dispenses:
            final_quantity = successful_dispenses[-1]["vending_quantity"]
            expected_quantity = 50 - len(successful_dispenses)

            if final_quantity != expected_quantity:
                print(f"🚨 バグ確認: 在庫減少ロジックに異常あり")
                print(f"   期待される最終在庫: {expected_quantity}個")
                print(f"   実際の最終在庫: {final_quantity}個")
                print(f"   差異: {abs(final_quantity - expected_quantity)}個")
            else:
                print(f"✅ バグなし: 在庫減少は正常")
        else:
            print(f"⚠️  dispenseが1度も成功しなかったため判定不可")

        print("\n💡 推奨アクション:")
        print("  1. InventorySlot.dispense()メソッドの詳細確認")
        print("  2. InventoryService.dispense_product()の複数スロット処理確認")
        print("  3. 在庫状態更新タイミングの確認")

        return {
            "test_completed": True,
            "bug_detected": final_quantity != expected_quantity
            if successful_dispenses
            else False,
            "dispense_results": dispense_results,
        }

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

        return {"error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nテスト結果: {result}")
