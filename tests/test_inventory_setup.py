#!/usr/bin/env python3
"""
STORAGEも含めた在庫初期化テスト

在庫可視化テストのための基本セットアップ
- VENDING_MACHINE在庫スロット初期化
- STORAGE在庫スロット初期化
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.application.services.inventory_service import inventory_service
from src.domain.models.inventory import InventoryLocation, InventorySlot
from src.domain.models.product import Product, ProductCategory, ProductSize

logger = logging.getLogger(__name__)


async def setup_test_inventory():
    """テスト用の在庫環境を初期化"""

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

    # VENDING_MACHINE在庫スロット初期化（ある程度在庫がある状態から開始）
    initial_stock_quantity = 20  # 十分な在庫量で開始
    test_inventory_slots = [
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=1,
        ),
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="cola_diet",
            product_name="コカ・コーラ ダイエット",
            price=150.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=2,
        ),
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="water_mineral",
            product_name="ミネラルウォーター",
            price=120.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=3,
        ),
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="energy_drink",
            product_name="エナジードリンク",
            price=180.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=4,
        ),
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="snack_chips",
            product_name="ポテトチップス",
            price=180.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=5,
        ),
        InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="snack_chocolate",
            product_name="チョコレートバー",
            price=160.0,
            current_quantity=initial_stock_quantity,
            max_quantity=50,
            min_quantity=5,
            slot_number=6,
        ),
    ]

    # STORAGE在庫スロットも作成（補充プロセスで使用）
    storage_stock_quantity = 150  # STORAGEには150個ずつストック
    test_storage_slots = [
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,  # STORAGEはより多く保持可能
            min_quantity=50,
            slot_number=1,
        ),
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="cola_diet",
            product_name="コカ・コーラ ダイエット",
            price=150.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=50,
            slot_number=2,
        ),
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="water_mineral",
            product_name="ミネラルウォーター",
            price=120.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=50,
            slot_number=3,
        ),
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="energy_drink",
            product_name="エナジードリンク",
            price=180.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=50,
            slot_number=4,
        ),
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="snack_chips",
            product_name="ポテトチップス",
            price=180.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=50,
            slot_number=5,
        ),
        InventorySlot(
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="snack_chocolate",
            product_name="チョコレートバー",
            price=160.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=50,
            slot_number=6,
        ),
    ]

    # 在庫サービスをクリアして再初期化
    inventory_service._slots = {}
    inventory_service._storage_slots = {}

    # VENDING_MACHINEスロットを追加
    for slot in test_inventory_slots:
        inventory_service.add_slot(slot)
        logger.info(f"在庫スロットを追加: {slot.product_name} (vending_machine)")

    # STORAGEスロットを追加
    for slot in test_storage_slots:
        inventory_service.add_slot(slot)
        logger.info(f"在庫スロットを追加: {slot.product_name} (storage)")

    logger.info("✅ テスト用在庫環境初期化完了")

    # 初期化後の確認
    all_inventory = inventory_service.get_inventory_by_location()
    vending_inventory = all_inventory.get("vending_machine", [])
    storage_inventory = all_inventory.get("storage", [])

    print(
        f"初期化完了: VENDING_MACHINE={len(vending_inventory)}スロット, STORAGE={len(storage_inventory)}スロット"
    )

    return test_products, test_inventory_slots + test_storage_slots


async def main():
    """メイン実行関数"""
    logging.basicConfig(level=logging.INFO)

    print("🚀 在庫初期化テスト開始")
    print("=" * 40)

    try:
        products, slots = await setup_test_inventory()

        print(f"✅ 商品数: {len(products)}")
        print(f"✅ 総スロット数: {len(slots)}")
        print("🎯 在庫可視化テストを実行してください")

    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
