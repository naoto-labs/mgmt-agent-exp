"""
continuous_multi_day_simulation.py - 調達遅延と原価変動を考慮した連続多日シミュレーション

test_case_aを参考にして、自動調達システムを統合した連続的なシミュレーションを実装。
在庫補充、調達、原価登録を完全自動化し、遅延・コスト変動を現実的にシミュレート。
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from uuid import uuid4

from src.agents.management_agent import ManagementState, management_agent
from src.agents.management_agent.agent import MetricsEvaluatingStateGraph
from src.agents.management_agent.evaluation_metrics import (
    create_benchmarks_table,
    eval_step_metrics,
    evaluate_primary_metrics,
    evaluate_secondary_metrics,
)
from src.agents.management_agent.metrics_tracker import VendingBenchMetricsTracker
from src.application.services.inventory_service import inventory_service
from src.domain.accounting.journal_entry import journal_processor
from src.domain.models.inventory import InventoryLocation, InventorySlot
from src.domain.models.product import Product, ProductCategory, ProductSize
from src.simulations.sales_simulation import simulate_purchase_events

logger = logging.getLogger(__name__)


async def setup_simulation_environment():
    """連続多日シミュレーション用の環境を初期化"""
    logger.info("🔧 連続多日シミュレーション環境を初期化...")

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

    # 在庫スロットを初期化（max_quantityの1/3程度の在庫で開始、補充プロセスを適正タイミングで開始）
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

    # 在庫サービスをクリアして再初期化
    inventory_service._slots = {}
    inventory_service._storage_slots = {}

    for slot in test_inventory_slots + test_storage_slots:
        inventory_service.add_slot(slot)

    logger.info("✅ 連続多日シミュレーション環境初期化完了")
    return test_products, test_inventory_slots


async def run_daily_management_cycle(
    state: ManagementState,
    day: int,
    run_id: str,
    conn,
    evaluating_graph: MetricsEvaluatingStateGraph,
    verbose: bool = True,
) -> ManagementState:
    """1日分のManagement Agentサイクルを実行

    Args:
        state: 現在のManagementState
        day: 日数（何日目か）
        run_id: 実行ID
        conn: データベース接続
        verbose: 詳細出力フラグ

    Returns:
        更新されたManagementState
    """
    if verbose:
        logger.info(f"🏪 Day {day + 1} - Management Agent実行開始")
        logger.info(
            f"現在の状態: session_id={state.session_id}, day_sequence={state.day_sequence}"
        )

    # ステップ1: 前日のデータをキャリーオーバー
    if hasattr(state, "previous_day_carry_over") and state.previous_day_carry_over:
        # 前日のfinal_reportを引き継ぎ
        if verbose:
            logger.info("📋 前日のデータを引き継ぎ中...")

    # 日次セッションを設定
    state.business_date = datetime.now().date() + timedelta(days=day)
    state.day_sequence = day + 1

    # 在庫補充と調達の自動実行
    if verbose and day == 0:
        logger.info("🔄 StateGraph実行中...")

    management_execution_state = await evaluating_graph.ainvoke(state)
    if isinstance(management_execution_state, dict):
        # ainvokeがdictを返した場合、stateを更新
        state.business_metrics = management_execution_state.get(
            "business_metrics", state.business_metrics
        )
        state.executed_actions = management_execution_state.get(
            "executed_actions", state.executed_actions
        )
        state.pending_procurements = management_execution_state.get(
            "pending_procurements", state.pending_procurements
        )
        state.inventory_analysis = management_execution_state.get(
            "inventory_analysis", state.inventory_analysis
        )
        state.sales_analysis = management_execution_state.get(
            "sales_analysis", state.sales_analysis
        )
        state.pricing_decision = management_execution_state.get(
            "pricing_decision", state.pricing_decision
        )
        state.restock_decision = management_execution_state.get(
            "restock_decision", state.restock_decision
        )
        state.procurement_decision = management_execution_state.get(
            "procurement_decision", state.procurement_decision
        )
        state.sales_processing = management_execution_state.get(
            "sales_processing", state.sales_processing
        )
        state.customer_interaction = management_execution_state.get(
            "customer_interaction", state.customer_interaction
        )
        state.profit_calculation = management_execution_state.get(
            "profit_calculation", state.profit_calculation
        )
        state.feedback = management_execution_state.get("feedback", state.feedback)
        state.final_report = management_execution_state.get(
            "final_report", state.final_report
        )
        state.errors = management_execution_state.get("errors", state.errors)
        # cumulative_kpisをコピー
        state.cumulative_kpis = management_execution_state.get(
            "cumulative_kpis", state.cumulative_kpis
        )
        # ManagementStateオブジェクトとして扱うため、state自体をmanagement_execution_stateに置き換え
        management_execution_state = state
    management_execution_state.day_sequence = day + 1

    if verbose:
        logger.info(f"✅ Day {day + 1} - Management Agent実行完了")

        # アクション実行結果表示
        executed_actions = management_execution_state.executed_actions
        procurement_actions = [
            a for a in executed_actions if "procurement" in a.get("type", "")
        ]
        restock_actions = [
            a for a in executed_actions if "restock" in a.get("type", "")
        ]

        logger.info(f"   総実行アクション数: {len(executed_actions)}")
        logger.info(f"   調達アクション数: {len(procurement_actions)}")
        logger.info(f"   在庫補充アクション数: {len(restock_actions)}")

        # 現在のpending_procurements状況を表示
        pending_count = len(management_execution_state.pending_procurements)
        logger.info(f"   未完了調達数: {pending_count}")

        # 売上データ確認
        sales_events = management_execution_state.actual_sales_events
        if sales_events:
            logger.info(f"   本日の売上イベント数: {len(sales_events)}")

    # ステップ3: 進行中調達の状態更新（遅延・完了処理）
    updated_state = await update_pending_procurements(
        management_execution_state, verbose
    )

    # ステップ4: 自然な販売活動をシミュレート（次日の準備）
    sales_simulation_result = await simulate_purchase_events(
        sales_lambda=5.0,  # 平均5イベント/日
        verbose=False,
        period_name=f"Day {day + 1} Sales",
    )

    if sales_simulation_result.get("successful_sales", 0) > 0:
        # 売上イベントをstateに記録
        sales_event = {
            "event_id": str(uuid4()),
            "simulation_date": (datetime.now() + timedelta(days=day)).isoformat(),
            "total_events": sales_simulation_result.get("total_events", 0),
            "successful_sales": sales_simulation_result.get("successful_sales", 0),
            "total_revenue": sales_simulation_result.get("total_revenue", 0),
            "conversion_rate": sales_simulation_result.get("conversion_rate", 0),
            "average_budget": sales_simulation_result.get("average_budget", 0),
        }
        updated_state.actual_sales_events.append(sales_event)

        if verbose:
            logger.info(
                f"💰 Day {day + 1} - 販売収入: ¥{sales_simulation_result['total_revenue']:,.0f}"
            )

            # ステップ5: 当日の主要指標を累積データに記録
            updated_state.primary_metrics_history.append(
                {
                    "day": day + 1,
                    "session_id": updated_state.session_id,
                    "profit_amount": updated_state.profit_amount
                    if hasattr(updated_state, "profit_amount")
                    else 0,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # ステップ6: その日のKPIデータをデータベースに保存
            try:
                # eval_step_metricsでKPIデータを保存
                kpi_result = eval_step_metrics(conn, run_id, day + 1, updated_state)

                if kpi_result["status"] == "success":
                    logger.info(f"✅ Day {day + 1} KPIデータ保存完了")
                else:
                    logger.warning(
                        f"⚠️ Day {day + 1} KPIデータ保存失敗: {kpi_result.get('error', 'unknown error')}"
                    )

            except Exception as kpi_error:
                logger.error(f"Day {day + 1} KPI保存エラー: {kpi_error}")
                results["issues"].append(
                    {"type": "kpi_save_error", "day": day + 1, "error": str(kpi_error)}
                )

            return updated_state


async def update_pending_procurements(
    state: ManagementState, verbose: bool = True
) -> ManagementState:
    """保留中の調達注文の状態を更新（遅延/完了処理）

    Args:
        state: 現在のManagementState
        verbose: 詳細出力フラグ

    Returns:
        更新されたManagementState
    """
    if not hasattr(state, "pending_procurements") or not state.pending_procurements:
        return state

    logger.info(f"🔄 保留中調達 {len(state.pending_procurements)}件 の状態更新")

    # 保留中調達のコピーを作成（イテレーション中に変更するため）
    pending_procurements = state.pending_procurements.copy()
    completed_procurements = []
    delayed_still = []

    for proc in pending_procurements:
        # 遅延処理
        if proc.get("delayed", False):
            remaining_delay = proc.get("remaining_delay", 0)
            remaining_delay -= 1

            if remaining_delay <= 0:
                # 遅延完了 → 実際の調達処理実行
                if verbose:
                    logger.info(f"⏰ 遅延調達完了: {proc['product']}")

                # 原価変動を適用して調達を完了
                from src.agents.management_agent.procurement_tools.request_procurement import (
                    register_procurement_cost,
                )

                base_cost = proc.get("base_cost", 100)
                cost_variation = proc.get("cost_variation", 0)
                actual_cost = base_cost * (1 + cost_variation)

                result = register_procurement_cost(
                    proc["product"], actual_cost, proc["quantity"]
                )

                if result["success"]:
                    # 在庫に追加
                    success = inventory_service.add_inventory(
                        product_name=proc["product"], quantity=proc["quantity"]
                    )

                    if success:
                        completed_procurements.append(
                            {
                                **proc,
                                "actual_cost": actual_cost,
                                "completion_date": datetime.now().isoformat(),
                                "inventory_updated": True,
                            }
                        )

                        if verbose:
                            logger.info(
                                f"✅ 遅延調達処理完了: {proc['product']} x{proc['quantity']} @ ¥{actual_cost:.0f}"
                            )
                    else:
                        logger.error(f"在庫追加失敗: {proc['product']}")
                else:
                    logger.error(f"原価登録失敗: {proc['product']}")

            else:
                # まだ遅延中
                proc["remaining_delay"] = remaining_delay
                delayed_still.append(proc)

                if verbose:
                    logger.info(
                        f"⏳ 調達引き続き遅延中: {proc['product']} ({remaining_delay}日残り)"
                    )

        else:
            # 非遅延 → 1日消費
            proc["days_waited"] = proc.get("days_waited", 0) + 1

            if proc["days_waited"] > 7:  # 最大待ち時間超過
                logger.warning(f"⚠️ 調達タイムアウト: {proc['product']}")
                continue

            delayed_still.append(proc)

    # アクションとして調達結果を記録
    for completed in completed_procurements:
        action = {
            "type": "completed_pending_procurement",
            "product": completed["product"],
            "quantity": completed["quantity"],
            "actual_cost": completed["actual_cost"],
            "original_delay": completed.get("delay_days", 0),
            "timestamp": datetime.now().isoformat(),
        }
        state.executed_actions.append(action)

    # State更新
    state.pending_procurements = delayed_still

    if verbose:
        logger.info(
            f"📊 調達状態更新完了 - 完了:{len(completed_procurements)}, 残延:{len(delayed_still)}"
        )

    return state


async def run_continuous_simulation(
    duration_days: int = 5,
    delay_probability: float = 0.3,
    cost_variation: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    test_case_aを参考にした連続多日調達シミュレーション

    Args:
        duration_days: シミュレーション期間（日数）
        delay_probability: 調達遅延発生確率
        cost_variation: 原価変動範囲（±cost_variation）
        verbose: 詳細出力フラグ

    Returns:
        シミュレーション結果
    """
    logger.info(f"=== 連続多日調達シミュレーション開始 ({duration_days}日間) ===")
    logger.info(
        f"調達遅延確率: {delay_probability:.1%}, コスト変動幅: ±{cost_variation:.1%}"
    )

    start_time = datetime.now()
    results = {
        "simulation_params": {
            "duration_days": duration_days,
            "delay_probability": delay_probability,
            "cost_variation": cost_variation,
            "start_time": start_time.isoformat(),
        },
        "daily_results": [],
        "cumulative_metrics": {},
        "procurement_analysis": {},
        "issues": [],
    }

    try:
        # ステップ1: 環境初期化
        await setup_simulation_environment()

        # ステップ2: データベース接続とテーブル設定
        import sqlite3

        db_path = "data/vending_bench.db"
        conn = sqlite3.connect(db_path)
        create_benchmarks_table(conn)

        run_id = f"continuous_simulation_{start_time.strftime('%Y%m%d_%H%M%S')}"

        # ステップ3: 初期ManagementState作成
        initial_state = ManagementState(
            session_id=f"{run_id}_initial",
            session_type="continuous_management_flow",
            day_sequence=1,
            delay_probability=delay_probability,
            cost_variation=cost_variation,
            cumulative_kpis={
                "total_profit": 0.0,
                "average_stockout_rate": 0.0,
                "customer_satisfaction_trend": [],
                "action_accuracy_history": [],
            },
        )

        # 初期ビジネスデータを設定
        metrics = management_agent.get_business_metrics()
        initial_state.business_metrics = metrics

        current_state = initial_state

        # ステップ4: StateGraphを一回だけ作成（トレース連続性のため）
        logger.info("🔧 StateGraphを初期化...")
        # 親トレースIDを設定してトレースの連続性を確保
        import uuid

        parent_trace_id = str(uuid.uuid4())
        logger.info(f"📊 親トレースIDを設定: {parent_trace_id}")

        evaluating_graph = MetricsEvaluatingStateGraph(
            management_agent, conn, run_id, parent_trace_id
        )
        evaluating_graph.set_parent_trace_id(parent_trace_id)  # トレースIDを設定
        logger.info("✅ StateGraph初期化完了")

        # ステップ5: 日次ループ実行
        for day in range(duration_days):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"📅 シミュレーション Day {day + 1} / {duration_days}")
            logger.info(f"{'=' * 50}")

            # 累積KPIデータをバックアップ（日次サイクル実行前の状態を保持）
            cumulative_backup = current_state.cumulative_kpis.copy()
            logger.debug(f"累積KPIバックアップ: {cumulative_backup}")

            # Management Agentの1日サイクル実行
            updated_state = await run_daily_management_cycle(
                current_state, day, run_id, conn, evaluating_graph, verbose
            )

            # 累積KPIデータを維持（各Nodeで更新された累積データを保持）
            # ノード実行により更新されたcumulative_kpisを優先的に使用 - 各ノードで正しく更新された累積利益を維持
            # バックアップ（初期状態）は無視し、各ノードで更新された値（特にtotal_profit）を保持
            logger.debug(f"累積KPI維持: {updated_state.cumulative_kpis}")

            current_state = updated_state

            # 日次結果を保存
            daily_result = {
                "day": day + 1,
                "session_id": current_state.session_id,
                "business_date": current_state.business_date.isoformat()
                if current_state.business_date
                else None,
                "executed_actions_count": len(current_state.executed_actions),
                "pending_procurements_count": len(current_state.pending_procurements),
                "sales_events_count": len(current_state.actual_sales_events),
                "errors_count": len(current_state.errors),
                "profit_amount": current_state.profit_amount
                if hasattr(current_state, "profit_amount")
                else 0,
                "cumulative_profit": current_state.cumulative_kpis.get(
                    "total_profit", 0
                ),
            }

            results["daily_results"].append(daily_result)

            # デバッグログ: 累積利益の詳細を確認
            logger.debug(
                f"Day {day + 1} - cumulative_kpis: {current_state.cumulative_kpis}"
            )
            logger.debug(
                f"Day {day + 1} - cumulative_profit from state: {current_state.cumulative_kpis.get('total_profit', 0)}"
            )
            logger.debug(
                f"Day {day + 1} - daily_result cumulative_profit: {daily_result['cumulative_profit']}"
            )

            if verbose:
                logger.info(f"📊 Day {day + 1} サマリー:")
                logger.info(
                    f"   実行アクション: {daily_result['executed_actions_count']}件"
                )
                logger.info(
                    f"   保留調達: {daily_result['pending_procurements_count']}件"
                )
                logger.info(f"   売上イベント: {daily_result['sales_events_count']}件")
                logger.info(f"   累積利益: ¥{daily_result['cumulative_profit']:,.0f}")

            # 定期休憩で安定した実行を確保
            await asyncio.sleep(0.1)

        # ステップ5: 最終集計
        final_state = current_state
        results["final_state"] = {
            "total_actions": len(final_state.executed_actions),
            "total_sales_events": len(final_state.actual_sales_events),
            "pending_procurements": len(final_state.pending_procurements),
            "total_errors": len(final_state.errors),
            "cumulative_profit": final_state.cumulative_kpis.get("total_profit", 0),
        }

        # 調達分析
        procurement_actions = [
            a
            for a in final_state.executed_actions
            if "procurement" in a.get("type", "")
        ]
        restock_actions = [
            a for a in final_state.executed_actions if "restock" in a.get("type", "")
        ]

        results["procurement_analysis"] = {
            "total_procurements": len(procurement_actions),
            "total_restocking": len(restock_actions),
            "pending_procurements": len(final_state.pending_procurements),
            "delayed_orders": len(
                [p for p in final_state.pending_procurements if p.get("delayed", False)]
            ),
            "completed_pending_orders": len(
                [p for p in procurement_actions if p.get("pending", False)]
            ),
        }

        results["cumulative_metrics"] = final_state.cumulative_kpis

        # 所要時間を計算
        end_time = datetime.now()
        results["execution_time"] = (end_time - start_time).total_seconds()

        logger.info(f"\n{'=' * 60}")
        logger.info("🎯 連続多日調達シミュレーション完了")
        logger.info(f"{'=' * 60}")
        logger.info(f"総実行日数: {duration_days}")
        logger.info(f"総アクション数: {results['final_state']['total_actions']}")
        logger.info(f"総売上イベント数: {results['final_state']['total_sales_events']}")
        logger.info(f"累積利益: ¥{results['final_state']['cumulative_profit']:,.0f}")
        logger.info(
            f"保留中調達: {results['procurement_analysis']['pending_procurements']}"
        )
        logger.info(f"実行時間: {results['execution_time']:.1f}秒")
        logger.info(f"{'=' * 60}")

        conn.close()

    except Exception as e:
        logger.error(f"連続シミュレーション実行エラー: {e}")
        results["issues"].append(
            {
                "type": "simulation_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )
        import traceback

        traceback.print_exc()

    return results


async def analyze_procurement_patterns(results: dict) -> dict:
    """調達パターンを分析"""
    logger.info("🔍 調達パターンの分析を開始")

    analysis = {
        "procurement_efficiency": {},
        "delay_analysis": {},
        "cost_variation_impact": {},
        "inventory_optimization": {},
    }

    try:
        daily_results = results.get("daily_results", [])
        procurement_analysis = results.get("procurement_analysis", {})

        # 調達効率分析
        total_procurements = procurement_analysis.get("total_procurements", 0)
        pending_procurements = procurement_analysis.get("pending_procurements", 0)

        if total_procurements > 0:
            completion_rate = (
                total_procurements - pending_procurements
            ) / total_procurements
            analysis["procurement_efficiency"] = {
                "total_procurements": total_procurements,
                "completed_procurements": total_procurements - pending_procurements,
                "pending_procurements": pending_procurements,
                "completion_rate": completion_rate,
                "assessment": "excellent"
                if completion_rate > 0.9
                else "good"
                if completion_rate > 0.7
                else "needs_improvement",
            }

            logger.info(".1%")

        # 遅延分析
        delayed_orders = procurement_analysis.get("delayed_orders", 0)
        if delayed_orders > 0:
            delay_ratio = (
                delayed_orders / total_procurements if total_procurements > 0 else 0
            )
            analysis["delay_analysis"] = {
                "delayed_orders": delayed_orders,
                "delay_ratio": delay_ratio,
                "impact_assessment": "minor"
                if delay_ratio < 0.1
                else "moderate"
                if delay_ratio < 0.3
                else "significant",
            }

            logger.info(".1%")

        # コスト変動影響分析
        # 実際の原価データを分析するため、procurement_actionsを詳細分析
        cost_variations = []
        for day_result in daily_results:
            if "procurement_details" in day_result:
                for proc in day_result["procurement_details"]:
                    if "actual_cost" in proc and "base_cost" in proc:
                        variation = (proc["actual_cost"] - proc["base_cost"]) / proc[
                            "base_cost"
                        ]
                        cost_variations.append(variation)

        if cost_variations:
            avg_variation = sum(cost_variations) / len(cost_variations)
            analysis["cost_variation_impact"] = {
                "total_cost_records": len(cost_variations),
                "average_variation": avg_variation,
                "cost_stability": "stable" if abs(avg_variation) < 0.05 else "volatile",
            }

            logger.info(
                f"原価変動影響: 平均変動率 {avg_variation:.1%} ({analysis['cost_variation_impact']['cost_stability']})"
            )

        logger.info("✅ 調達パターン分析完了")

    except Exception as e:
        logger.error(f"調達パターン分析エラー: {e}")
        analysis["error"] = str(e)

    return analysis


async def main():
    """メイン実行関数"""
    print("🚀 Continuous Multi-Day Procurement Simulation")
    print("=" * 60)

    # シミュレーション実行（5日間）
    simulation_results = await run_continuous_simulation(
        duration_days=10,
        delay_probability=0.3,  # 30%の確率で調達遅延
        cost_variation=0.1,  # ±10%の原価変動
        verbose=True,
    )

    # 調達パターン分析
    procurement_analysis = await analyze_procurement_patterns(simulation_results)

    print("\n" + "=" * 60)
    print("📊 SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    final_state = simulation_results.get("final_state", {})
    procurement = simulation_results.get("procurement_analysis", {})

    print(
        f"Total Duration:           {simulation_results['simulation_params']['duration_days']} days"
    )
    print(f"Total Actions:           {final_state.get('total_actions', 0)}")
    print(f"Total Sales Events:      {final_state.get('total_sales_events', 0)}")
    print(f"Cumulative Profit:       ¥{final_state.get('cumulative_profit', 0):,.0f}")
    print(f"Total Procurements:      {procurement.get('total_procurements', 0)}")
    print(f"Pending Procurements:    {procurement.get('pending_procurements', 0)}")
    print(f"Restocking Actions:      {procurement.get('total_restocking', 0)}")
    print(f"Timed Delayed Orders:    {procurement.get('delayed_orders', 0)}")
    print(
        f"Execution Time:          {simulation_results.get('execution_time', 0):.1f} seconds"
    )

    if procurement_analysis:
        print(f"\nProcurement Efficiency:")
        efficiency = procurement_analysis.get("procurement_efficiency", {})
        if efficiency:
            print(f"  Completion Rate:      {efficiency.get('completion_rate', 0):.1%}")
            print(f"  Assessment:           {efficiency.get('assessment', 'unknown')}")

        delay_info = procurement_analysis.get("delay_analysis", {})
        if delay_info:
            print(
                f"  Delay Impact:         {delay_info.get('impact_assessment', 'unknown')}"
            )

        cost_info = procurement_analysis.get("cost_variation_impact", {})
        if cost_info:
            print(
                f"  Cost Stability:       {cost_info.get('cost_stability', 'unknown')}"
            )

    print(f"\n✅ Continuous Multi-Day Procurement Simulation Completed Successfully!")
    print("=" * 60)

    # シミュレーション後にKPI時系列可視化を自動実行
    try:
        print("\n🎯 シミュレーション完了後にKPI時系列可視化を自動実行します...")
        from kpi_visualization import main as run_kpi_visualization

        run_kpi_visualization()
    except Exception as e:
        print(f"⚠️ KPI可視化実行中にエラーが発生しました: {e}")
        print("手動で python kpi_visualization.py を実行してください")

    return simulation_results, procurement_analysis


if __name__ == "__main__":
    asyncio.run(main())
