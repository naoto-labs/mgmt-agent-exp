"""
continuous_procurement_simulation.py - 連続調達シミュレーション

調達→保管→補充の一連のフローを持つ連続シミュレーションを実装
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

from src.agents.management_agent.procurement_tools.assign_restocking_task import (
    execute_restocking_task,
    get_restocking_status,
)
from src.agents.management_agent.procurement_tools.request_procurement import (
    complete_procurement,
    request_procurement,
    simulate_procurement_delay,
)
from src.application.services.inventory_service import inventory_service
from src.domain.accounting.journal_entry import journal_processor
from src.simulations.sales_simulation import simulate_purchase_events

logger = logging.getLogger(__name__)


async def initialize_simulation_data():
    """シミュレーション用の商品データと在庫を初期化"""
    logger.info("シミュレーション用初期化データをセットアップ...")

    try:
        # テスト用の商品データを作成（BusinessMetricsと一致させる）
        test_products = [
            {
                "product_id": "cola_regular",
                "name": "コカ・コーラ レギュラー",
                "description": "美味しい炭酸飲料",
                "category": "DRINK",
                "price": 150.0,
                "cost": 100.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "cola_diet",
                "name": "コカ・コーラ ダイエット",
                "description": "カロリーオフの炭酸飲料",
                "category": "DRINK",
                "price": 150.0,
                "cost": 100.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "water_mineral",
                "name": "ミネラルウォーター",
                "description": "爽やかなミネラルウォーター",
                "category": "DRINK",
                "price": 120.0,
                "cost": 80.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "energy_drink",
                "name": "エナジードリンク",
                "description": "元気が出るドリンク",
                "category": "DRINK",
                "price": 180.0,
                "cost": 120.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "snack_chips",
                "name": "ポテトチップス",
                "description": "サクサクのスナック",
                "category": "SNACK",
                "price": 180.0,
                "cost": 120.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "snack_chocolate",
                "name": "チョコレートバー",
                "description": "甘いチョコレート",
                "category": "SNACK",
                "price": 160.0,
                "cost": 110.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "energy_drink",
                "name": "エナジードリンク",
                "description": "元気が出るドリンク",
                "category": "DRINK",
                "price": 180.0,
                "cost": 120.0,
                "size": "MEDIUM",
            },
            {
                "product_id": "snack_chips",
                "name": "ポテトチップス",
                "description": "サクサクのスナック",
                "category": "SNACK",
                "price": 180.0,
                "cost": 120.0,
                "size": "MEDIUM",
            },
        ]

        # 商品データをグローバル変数に登録
        from src.application.services.inventory_service import get_product_by_id
        from src.domain.models.product import SAMPLE_PRODUCTS

        # 元のデータを退避
        original_products = SAMPLE_PRODUCTS.copy()

        # test_productsをProductオブジェクトに変換して設定
        from src.domain.models.product import Product, ProductCategory, ProductSize

        test_product_objects = []
        seen_ids = set()  # 重複チェック用

        for p in test_products:
            if p["product_id"] in seen_ids:
                logger.warning(f"Duplicate product_id skipped: {p['product_id']}")
                continue

            seen_ids.add(p["product_id"])

            product_obj = Product(
                product_id=p["product_id"],
                name=p["name"],
                description=p["description"],
                category=getattr(ProductCategory, p["category"]),
                price=p["price"],
                cost=p["cost"],
                stock_quantity=0,
                max_stock_quantity=50,
                min_stock_quantity=5,
                size=getattr(ProductSize, p["size"]),
            )
            test_product_objects.append(product_obj)
            logger.debug(f"Product object created: {p['product_id']} - {p['name']}")

        SAMPLE_PRODUCTS[:] = test_product_objects  # リストの内容を置き換え

        # 在庫管理サービスの初期化
        from src.domain.models.inventory import InventoryLocation, InventorySlot

        test_inventory_slots = [
            InventorySlot(
                machine_id="VM001",
                location=InventoryLocation.VENDING_MACHINE,
                product_id="cola_regular",
                product_name="コカ・コーラ レギュラー",
                price=150.0,
                current_quantity=5,
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
                current_quantity=3,
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
                current_quantity=10,
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
                current_quantity=2,
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
                current_quantity=1,
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
                current_quantity=3,
                max_quantity=50,
                min_quantity=5,
                slot_number=6,
            ),
        ]

        # 在庫サービスにスロットを追加
        for slot in test_inventory_slots:
            inventory_service.add_slot(slot)

        logger.info(
            f"✓ Set up test inventory slots in inventory_service: {len(test_inventory_slots)} slots"
        )

        # 初期化完了を保存（クリーンアップ用）
        initialize_simulation_data.original_products = original_products

    except Exception as e:
        logger.error(f"シミュレーション初期化エラー: {e}")
        raise


async def cleanup_simulation_data():
    """シミュレーション終了後のクリーンアップ"""
    if hasattr(initialize_simulation_data, "original_products"):
        from src.domain.models.product import SAMPLE_PRODUCTS

        SAMPLE_PRODUCTS[:] = initialize_simulation_data.original_products
        logger.info("シミュレーションデータをクリーンアップしました")


async def simulate_continuous_procurement_cycle(
    duration_days: int = 2,
    delay_probability: float = 0.3,  # 調達遅延が発生する確率
    cost_variation: float = 0.2,  # 原価変動幅（±20%）
    verbose: bool = True,
) -> Dict[str, any]:
    """
    調達遅延と原価変動を考慮した連続調達シミュレーション

    Args:
        duration_days: シミュレーション期間（日数）
        delay_probability: 調達遅延発生確率
        cost_variation: 原価変動幅
        verbose: 詳細出力フラグ
    """
    logger.info(f"=== 連続調達シミュレーション開始 ({duration_days}日間) ===")

    simulation_results = {
        "simulation_period": duration_days,
        "procurement_orders": [],
        "completed_procurements": [],
        "delayed_orders": [],
        "restocking_tasks": [],
        "sales_events": [],
        "financial_summary": {},
        "issues": [],
    }

    start_time = datetime.now()

    # シミュレーション用の商品データと在庫を初期化
    await initialize_simulation_data()

    try:
        # 初期調達を実行（シミュレーション開始時のSTORAGE補充）
        initial_order = request_procurement(
            products=["cola_regular", "cola_diet", "water_mineral"],
            quantity={"cola_regular": 50, "cola_diet": 30, "water_mineral": 40},
            supplier_costs={"cola_regular": 90, "cola_diet": 85, "water_mineral": 70},
        )
        simulation_results["procurement_orders"].append(initial_order)

        if verbose:
            logger.info(f"初期調達発注: {initial_order['order_id']}")

        # 調達遅延シミュレーション（3日遅延）
        if random.random() < delay_probability:
            delayed_order = simulate_procurement_delay(initial_order, delay_days=3)
            simulation_results["delayed_orders"].append(delayed_order)
            logger.warning(f"調達遅延発生: {initial_order['order_id']} (+3日)")

        # 初期調達完了
        completion_result = complete_procurement(initial_order)
        if completion_result["success"]:
            simulation_results["completed_procurements"].append(
                completion_result["order"]
            )
            logger.info(
                f"初期調達完了: {len(completion_result['order']['completed_items'])}商品"
            )
        else:
            simulation_results["issues"].append(
                {
                    "type": "procurement_failure",
                    "order_id": initial_order["order_id"],
                    "error": completion_result["error"],
                }
            )

    except Exception as e:
        logger.error(f"初期調達エラー: {e}")
        simulation_results["issues"].append(
            {"type": "initialization_error", "error": str(e)}
        )
        return simulation_results

    # 日次ループ：調達遅延と原価変動を考慮
    for day in range(duration_days):
        current_date = start_time + timedelta(days=day)
        logger.info(f"--- Day {day + 1}: {current_date.date()} ---")

        try:
            # 在庫チェックと低在庫商品の特定
            low_stock_products = []
            for slot in inventory_service.vending_machine_slots.values():
                if slot.needs_restock() or slot.status.name == "OUT_OF_STOCK":
                    if slot.product_id not in low_stock_products:
                        low_stock_products.append(slot.product_id)

            # STORAGE在庫確認
            for product_id in low_stock_products:
                inventory_info = inventory_service.get_total_inventory(product_id)
                storage_stock = inventory_info["storage_stock"]

                if storage_stock > 0:
                    # STORAGEに在庫がある場合は補充タスク実行
                    restocking_result = execute_restocking_task(
                        f"auto_restock_{day}_{product_id}"
                    )
                    simulation_results["restocking_tasks"].append(restocking_result)

                    if verbose:
                        logger.info(
                            f"自動補充実行: {product_id} - 完了={len(restocking_result.get('completed_transfers', []))}件"
                        )

                else:
                    # STORAGEに在庫がない場合は調達依頼
                    base_costs = {
                        "cola_regular": 100,
                        "cola_diet": 95,
                        "water_mineral": 80,
                        "energy_drink": 120,
                        "snack_chips": 120,
                        "snack_chocolate": 110,
                    }

                    # 原価変動を適用
                    variable_costs = {}
                    for product, base_cost in base_costs.items():
                        variation = random.uniform(-cost_variation, cost_variation)
                        variable_costs[product] = base_cost * (1 + variation)

                    order = request_procurement(
                        products=[product_id],
                        quantity={product_id: 30},  # 通常発注量
                        supplier_costs=variable_costs,
                    )
                    simulation_results["procurement_orders"].append(order)

                    if verbose:
                        logger.info(
                            f"調達発注: {product_id} - 原価変動後: ¥{variable_costs[product_id]:.0f}"
                        )

                    # 調達遅延シミュレーション
                    if random.random() < delay_probability:
                        delay_days_sim = random.randint(1, 5)  # 1-5日の遅延
                        delayed_order = simulate_procurement_delay(
                            order, delay_days=delay_days_sim
                        )
                        simulation_results["delayed_orders"].append(delayed_order)

                        # 遅延中は補充不可
                        if verbose:
                            logger.warning(
                                f"調達遅延: {order['order_id']} (+{delay_days_sim}日)"
                            )
                        continue

                    # 調達完了（遅延なしの場合）
                    completion_result = complete_procurement(order)
                    if completion_result["success"]:
                        simulation_results["completed_procurements"].append(
                            completion_result["order"]
                        )
                        if verbose:
                            logger.info(f"調達完了: {order['order_id']}")

            # 販売シミュレーション実行
            sales_result = await simulate_purchase_events(
                sales_lambda=8.0,  # 平均8イベント/日
                verbose=verbose,
                period_name=f"Day {day + 1}",
            )
            simulation_results["sales_events"].append(
                {"day": day + 1, "date": current_date.isoformat(), **sales_result}
            )

            # 定期休憩（実際の時間経過をシミュレート）
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Day {day + 1} エラー: {e}")
            simulation_results["issues"].append(
                {"type": "daily_cycle_error", "day": day + 1, "error": str(e)}
            )

    # 最終財務サマリー生成
    try:
        financial_summary = journal_processor.get_accounting_summary()
        financial_summary["total_procurement_orders"] = len(
            simulation_results["procurement_orders"]
        )
        financial_summary["total_completed_procurements"] = len(
            simulation_results["completed_procurements"]
        )
        financial_summary["total_delayed_orders"] = len(
            simulation_results["delayed_orders"]
        )
        financial_summary["simulation_duration_days"] = duration_days

        simulation_results["financial_summary"] = financial_summary

        logger.info(f"=== シミュレーション完了 ===")
        logger.info(f"総調達件数: {len(simulation_results['procurement_orders'])}")
        logger.info(
            f"完了調達件数: {len(simulation_results['completed_procurements'])}"
        )
        logger.info(f"遅延調達件数: {len(simulation_results['delayed_orders'])}")
        logger.info(
            f"総売上: ¥{sum(event['total_revenue'] for event in simulation_results['sales_events']):,.0f}"
        )

    except Exception as e:
        logger.error(f"財務サマリー生成エラー: {e}")
        simulation_results["issues"].append(
            {"type": "financial_summary_error", "error": str(e)}
        )

    # 在庫状況レポート
    final_inventory = inventory_service.get_inventory_report()
    simulation_results["final_inventory"] = final_inventory

    logger.info(f"最終STORAGE在庫: {final_inventory['storage_slots']}スロット")
    logger.info(
        f"最終VENDING_MACHINE在庫: {final_inventory['vending_machine_slots']}スロット"
    )

    return simulation_results


async def run_procurement_simulation_demo():
    """連続調達シミュレーションのデモ実行"""
    print("=== 連続調達シミュレーション デモ ===")

    # 7日間のシミュレーション実行
    results = await simulate_continuous_procurement_cycle(
        duration_days=7,
        delay_probability=0.4,  # 40%の確率で遅延発生
        cost_variation=0.15,  # ±15%の原価変動
        verbose=True,
    )

    print("=== シミュレーション結果 ===")
    print(f"調達発注総数: {len(results['procurement_orders'])}")
    print(f"完了調達数: {len(results['completed_procurements'])}")
    print(f"遅延調達数: {len(results['delayed_orders'])}")
    print(f"補充タスク実行数: {len(results['restocking_tasks'])}")

    total_sales = sum(event["total_revenue"] for event in results["sales_events"])
    print(f"総売上: ¥{total_sales:,.0f}")

    if results["issues"]:
        print(f"発生した問題: {len(results['issues'])}件")

    print("=== シミュレーション完了 ===")


if __name__ == "__main__":
    asyncio.run(run_procurement_simulation_demo())
