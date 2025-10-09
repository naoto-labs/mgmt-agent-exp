import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
import asyncio

# 連続調達シミュレーションのインポート
from continuous_procurement_simulation import (
    run_procurement_simulation_demo,
    simulate_continuous_procurement_cycle,
)
from src.agents.management_agent import ManagementState, management_agent

# LCEL準拠拡張可能パイプライン実行用
from src.agents.management_agent.agent import (
    MetricsEvaluatingStateGraph,
    RunnableManagementPipeline,
)
from src.agents.management_agent.evaluation_metrics import (
    create_benchmarks_table,
    eval_step_metrics,
    evaluate_primary_metrics,
    evaluate_secondary_metrics,
)
from src.agents.management_agent.metrics_tracker import VendingBenchMetricsTracker


def validate_state_after_node(
    state: ManagementState, node_name: str, expected_keys: list
):
    """ノード実行後の状態を検証する"""
    print(f"  [Validation] Node '{node_name}' state check:")
    all_valid = True

    for key in expected_keys:
        attr_value = getattr(state, key, None)
        if attr_value is not None:
            print(f"    ✓ {key}: {type(attr_value).__name__} (set)")
        else:
            print(f"    ✗ {key}: None (missing)")
            all_valid = False

    # エラーチェック
    if state.errors:
        print(f"    ⚠️ {len(state.errors)} errors present: {state.errors}")

    # ステップ更新チェック
    if state.current_step == node_name:
        print(f"    ✓ current_step updated to: {node_name}")
    else:
        print(f"    ⚠️ current_step: {state.current_step} (expected: {node_name})")

    if all_valid:
        print(f"    ✅ Node '{node_name}' validation PASSED")
    else:
        print(f"    ❌ Node '{node_name}' validation FAILED")

    return all_valid


def clear_test_case_data(conn):
    """test_case_aテストのデータをクリアする"""
    cursor = conn.cursor()
    try:
        # test_case_a prefixの全データを削除
        cursor.execute("DELETE FROM benchmarks WHERE run_id LIKE 'test_case_a_%'")
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"✓ Cleared {deleted_count} test_case_a benchmark records")
    except Exception as e:
        print(f"Warning: Could not clear test data: {e}")


async def test_case_a():
    """Case A: Node-Based Management Agent Test - online評価のためのtest"""
    print("=== Case A Node-Based Management Agent Test ===")

    try:
        from uuid import uuid4

        print(f"Agent provider: {management_agent.provider}")
        print(f"Number of nodes: {len(management_agent.nodes)}")
        print(f"Node names: {list(management_agent.nodes.keys())}")

        # Check that all expected nodes are present
        expected_nodes = [
            "inventory_check",
            "sales_plan",
            "pricing",
            "restock",
            "procurement",
            "sales_processing",
            "customer_interaction",
            "profit_calculation",
            "feedback",
        ]

        all_present = True
        for node_name in expected_nodes:
            if node_name not in management_agent.nodes:
                print(f"ERROR: Missing node: {node_name}")
                all_present = False
            else:
                print(f"✓ Node found: {node_name}")

        if all_present:
            print("\n✓ All expected nodes are configured correctly")
        else:
            print("\n✗ Some nodes are missing - check implementation")
            return False

        # Vending Bench準拠テストを開始
        print("\n=== Starting VendingBench Conformity Test ===")

        # 初期状態作成
        initial_state = ManagementState(
            session_id=str(uuid4()), session_type="management_flow"
        )
        print("✓ Initial ManagementState created")

        # LCEL パイプライン実行テスト
        print(
            "🟡 Testing LCEL RunnableSequence pipeline execution - executing Case A flow..."
        )

        try:
            # LCEL パイプライン実行: Chain.invoke()で全ノードを自動実行
            print("--- LCEL Pipeline Execution ---")

            # 初期状態データを事前に投入しておくための準備
            from uuid import uuid4

            from src.agents.management_agent import BusinessMetrics

            print("🟡 Setting up test data in actual system...")

            # テストデータを実際のシステムにセットアップ
            from datetime import date

            from src.application.services.inventory_service import inventory_service
            from src.domain.accounting.journal_entry import journal_processor
            from src.domain.models.inventory import InventoryLocation, InventorySlot
            from src.domain.models.product import Product, ProductCategory, ProductSize

            # テスト用の商品データを作成（BusinessMetricsと一致させる）
            test_products = [
                Product(
                    product_id="cola_regular",
                    name="コカ・コーラ レギュラー",
                    description="美味しい炭酸飲料",
                    category=ProductCategory.DRINK,
                    price=150.0,
                    cost=100.0,
                    stock_quantity=0,  # 在庫は別途InventorySlotで管理
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

            # グローバル変数に商品データを登録（get_product_by_id関数用）
            # 既存のSAMPLE_PRODUCTSを一時的に置き換え（テスト用）
            import src.domain.models.product as product_module
            from src.application.services.inventory_service import get_product_by_id

            original_sample_products = product_module.SAMPLE_PRODUCTS
            product_module.SAMPLE_PRODUCTS = test_products

            # 在庫スロットをセットアップ（max_quantityの1/3程度の在庫で開始、補充プロセスを適正タイミングで開始）
            initial_stock_quantity = int(50 / 3)  # max_quantityの1/3 ≈ 16個
            test_inventory_slots = [
                InventorySlot(
                    machine_id="VM001",
                    location=InventoryLocation.VENDING_MACHINE,
                    product_id="cola_regular",
                    product_name="コカ・コーラ レギュラー",
                    price=150.0,
                    current_quantity=initial_stock_quantity,  # max/3で一定量の在庫から開始
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
            storage_stock_quantity = 100  # STORAGEには100個ずつストック
            test_storage_slots = [
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="cola_regular",
                    product_name="コカ・コーラ レギュラー",
                    price=150.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,  # STORAGEはより多く保持可能
                    min_quantity=20,
                    slot_number=1,
                ),
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="cola_diet",
                    product_name="コカ・コーラ ダイエット",
                    price=150.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,
                    min_quantity=20,
                    slot_number=2,
                ),
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="water_mineral",
                    product_name="ミネラルウォーター",
                    price=120.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,
                    min_quantity=20,
                    slot_number=3,
                ),
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="energy_drink",
                    product_name="エナジードリンク",
                    price=180.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,
                    min_quantity=20,
                    slot_number=4,
                ),
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="snack_chips",
                    product_name="ポテトチップス",
                    price=180.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,
                    min_quantity=20,
                    slot_number=5,
                ),
                InventorySlot(
                    machine_id="STORAGE",
                    location=InventoryLocation.STORAGE,
                    product_id="snack_chocolate",
                    product_name="チョコレートバー",
                    price=160.0,
                    current_quantity=storage_stock_quantity,
                    max_quantity=200,
                    min_quantity=20,
                    slot_number=6,
                ),
            ]

            # 在庫サービスにスロットを追加
            for slot in test_inventory_slots + test_storage_slots:
                inventory_service.add_slot(slot)

            print("✓ Set up test inventory slots in inventory_service")
            print(f"  - VENDING_MACHINE slots: {len(test_inventory_slots)}")
            print(f"  - STORAGE slots: {len(test_storage_slots)}")

            # 売上データを会計システムに記録（950,000円の売上データを作成）
            print("  Setting up test sales data in journal processor...")

            # 月間販売データをシミュレート（30日分の売上）
            from datetime import datetime, timedelta

            base_date = date.today() - timedelta(days=30)
            total_sales_target = 50000  # 過去実績を低く設定して挑戦性を高める
            daily_sales_target = total_sales_target / 30

            for day in range(30):
                sales_date = base_date + timedelta(days=day)
                daily_sales = daily_sales_target

                # その日の売上を記録（簡易的な取引として）
                try:
                    from src.domain.accounting.journal_entry import journal_processor
                    from src.domain.models.transaction import (
                        PaymentDetails,
                        PaymentMethod,
                        Transaction,
                        TransactionItem,
                        TransactionType,
                    )

                    # 取引オブジェクトを作成 (created_atはdatetime型である必要がある)
                    items = [
                        TransactionItem(
                            product_id="cola_regular",
                            product_name="コカ・コーラ レギュラー",
                            quantity=int(daily_sales / 150),  # 平均単価150円で数量計算
                            unit_price=150.0,
                            total_price=daily_sales,
                        )
                    ]

                    transaction = Transaction(
                        transaction_id=f"test_txn_{day}_{datetime.now().strftime('%H%M%S')}",
                        machine_id="VM001",  # 必須フィールド
                        transaction_type=TransactionType.PURCHASE,  # SALEではなくPURCHASE
                        items=items,
                        subtotal=daily_sales,
                        total_amount=daily_sales,
                        payment_details=PaymentDetails(
                            method=PaymentMethod.CASH, amount=daily_sales
                        ),
                        created_at=datetime.combine(
                            sales_date, datetime.min.time()
                        ),  # dateをdatetimeに変換
                    )

                    # 売上仕訳を記録
                    journal_processor.record_sale(transaction)

                except Exception as e:
                    print(
                        f"  Warning: Failed to record daily sales for {sales_date}: {e}"
                    )
                    # Simple fallback entry
                    try:
                        journal_processor.add_entry(
                            account_number="4001",  # SALES_REVENUE
                            date=sales_date,
                            amount=daily_sales,
                            entry_type="credit",  # credit for revenue
                            description=f"Test sales day {day + 1}",
                        )
                        journal_processor.add_entry(
                            account_number="1001",  # CASH
                            date=sales_date,
                            amount=daily_sales,
                            entry_type="debit",  # debit for asset increase
                            description=f"Test sales day {day + 1}",
                        )
                    except Exception as e2:
                        print(f"  Error: Failed to record fallback sales data: {e2}")

            print("✓ Added test sales data to journal_processor")

            # 売上原価データを記録（利益率を32%にするため、売上の約68%をコストとして記録）
            # 月間売上95万円の68% = 約64.6万円のコスト
            print("  Setting up test cost data in journal processor...")

            total_cost_target = (
                total_sales_target * 0.75
            )  # 75%をコストとして（利益率25%）
            daily_cost_target = total_cost_target / 30

            for day in range(30):
                cost_date = base_date + timedelta(days=day)
                daily_cost = daily_cost_target

                # 売上原価を仕入として記録（5001: Cost of Goods Sold）
                try:
                    journal_processor.add_entry(
                        account_number="5001",  # COST_OF_GOODS_SOLD
                        date=cost_date,
                        amount=daily_cost,
                        entry_type="debit",  # debit for expense increase
                        description=f"Test cost of goods day {day + 1} - supply purchase",
                    )
                    # 支払いを記録（1001: Cash - debit to reduce cash）
                    journal_processor.add_entry(
                        account_number="1001",  # CASH
                        date=cost_date,
                        amount=daily_cost,
                        entry_type="credit",  # credit to reduce cash (payment)
                        description=f"Test payment for goods day {day + 1}",
                    )
                except Exception as e:
                    print(
                        f"  Warning: Failed to record daily cost for {cost_date}: {e}"
                    )

            print("✓ Added test cost data to journal_processor")

            # システムから実際の計算結果を取得（ハードコーディング禁止）
            from src.agents.management_agent.management_tools.get_business_metrics import (
                get_business_metrics,
            )

            actual_metrics = get_business_metrics()

            # 実際のシステム計算結果を使用（sales_plan_nodeへの入力と完全一致）
            test_metrics = BusinessMetrics(
                sales=actual_metrics["sales"],
                profit_margin=actual_metrics["profit_margin"],
                inventory_level=actual_metrics["inventory_level"],
                customer_satisfaction=actual_metrics["customer_satisfaction"],
                timestamp=datetime.now().isoformat(),
            )

            # 初期状態にビジネスデータを投入
            enriched_initial_state = initial_state.model_copy()
            # LangGraphシリアライズ対応: BusinessMetricsオブジェクトをdictに変換して代入
            enriched_initial_state.business_metrics = test_metrics.model_dump()

            print(f"✓ Enriched initial state with test data")
            print(f"  - Sales: ¥{test_metrics.sales:,}")
            print(f"  - Profit Margin: {test_metrics.profit_margin:.1%}")
            print(
                f"  - Customer Satisfaction: {test_metrics.customer_satisfaction}/5.0"
            )
            print(
                f"  - Inventory Slots: {len(inventory_service.vending_machine_slots)}"
            )
            print(f"  - Journal Entries: {len(journal_processor.journal_entries)}")

            # VendingBenchステップ単位評価セットアップ
            import sqlite3

            print("🔧 Setting up VendingBench step-by-step evaluation...")
            run_id = f"test_case_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Metrics Tracker初期化
            metrics_tracker = VendingBenchMetricsTracker(difficulty="normal")

            # データベース接続とテーブル作成
            db_path = "data/vending_bench.db"
            conn = sqlite3.connect(db_path)
            create_benchmarks_table(conn)

            # テストデータをクリア（自身のセッションデータのみ）
            clear_test_case_data(conn)

            # LangGraphベースのパイプライン構築と実行（LCEL形式）
            print("🚀 Initializing LangGraph pipeline with step-by-step evaluation...")
            evaluating_graph = MetricsEvaluatingStateGraph(
                management_agent, conn, run_id
            )

            print(
                "✅ LangGraph pipeline initialized - executing full management flow..."
            )
            final_state = await evaluating_graph.ainvoke(enriched_initial_state)
            print(
                "✅ LangGraph pipeline execution completed - VendingBench evaluation integrated"
            )

        except Exception as e:
            print(f"✗ Manual node execution failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def test_continuous_procurement():
    """連続調達シミュレーションテスト"""
    print("=== Continuous Procurement Simulation Test ===")

    try:
        # 短期間のテスト実行（3日間）
        results = await simulate_continuous_procurement_cycle(
            duration_days=3,  # テスト用に短めに設定
            delay_probability=0.3,
            cost_variation=0.1,
            verbose=True,
        )

        # 基本的な検証
        if len(results["procurement_orders"]) > 0:
            print("✓ 調達発注が正常に生成されました")
        else:
            print("✗ 調達発注が生成されませんでした")
            return False

        if len(results["completed_procurements"]) > 0:
            print("✓ 調達完了処理が実行されました")
        else:
            print("✗ 調達完了処理が実行されませんでした")
            return False

        if len(results["sales_events"]) == 3:
            print("✓ 販売シミュレーションが全期間実行されました")
        else:
            print(f"✗ 販売シミュレーションが不完全: {len(results['sales_events'])}/3")
            return False

        print("✓ Continuous Procurement Simulation Test PASSED")
        return True

    except Exception as e:
        print(f"✗ Continuous Procurement Simulation Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """メイン実行関数 - 両方のテスト実行"""
    print("=== Management Agent Architecture Comparison Test ===\n")

    # LangGraphベーステスト実行
    print("🔄 Testing traditional LangGraph-based pipeline...")
    langgraph_success = await test_case_a()

    print("\n" + "=" * 80 + "\n")

    # 連続調達シミュレーションテスト実行
    print("🛒 Testing continuous procurement simulation...")
    procurement_success = await test_continuous_procurement()

    print("\n" + "=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("=== Architecture Comparison Results ===")
    print(f"LangGraph Pipeline: {'✅ SUCCESS' if langgraph_success else '❌ FAILED'}")
    print(
        f"Continuous Procurement: {'✅ SUCCESS' if procurement_success else '❌ FAILED'}"
    )

    return langgraph_success and procurement_success


if __name__ == "__main__":
    asyncio.run(main())
