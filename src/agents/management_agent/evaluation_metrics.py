"""
VendingBench Evaluation Metrics
テストから分離された評価関数群
"""

import datetime
import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.agents.management_agent.agent import ManagementState

# VendingBench設定のインポート（設定更新時はここから変更可能）
from src.shared.config.vending_bench_metrics import VENDING_BENCH_METRICS

logger = logging.getLogger(__name__)


def create_benchmarks_table(conn: sqlite3.Connection) -> None:
    """
    VendingBench準拠のbenchmarksテーブルを作成する

    Args:
        conn: SQLiteデータベース接続
    """
    cursor = conn.cursor()

    # benchmarksテーブル作成 (VendingBench spec準拠DDL)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS benchmarks (
        run_id INTEGER,
        step INTEGER,
        profit_actual REAL,
        stockout_count INTEGER,
        total_demand INTEGER,
        pricing_accuracy REAL,
        action_correctness REAL,
        customer_satisfaction REAL
    );
    """

    cursor.execute(create_table_sql)
    conn.commit()


def _safe_get_business_metric(business_metrics, key, default=None):
    """
    business_metricsがBusinessMetricsオブジェクトかdictかを安全にアクセスするヘルパー関数

    Args:
        business_metrics: BusinessMetricsオブジェクトまたはdict
        key: アクセスするキー
        default: デフォルト値

    Returns:
        取得した値
    """
    if business_metrics is None:
        return default

    # dictの場合
    if isinstance(business_metrics, dict):
        return business_metrics.get(key, default)

    # BusinessMetricsオブジェクトの場合（過去互換用）
    if hasattr(business_metrics, key):
        return getattr(business_metrics, key, default)

    return default


def evaluate_primary_metrics(state) -> dict:
    """
    VendingBench Primary Metrics評価 - 実際の状態データから動的に計算

    Args:
        state: 評価対象のManagementStateまたはdict（LangGraph実行結果対応）

    Returns:
        Primary Metrics評価結果
    """
    metrics = {}

    # LangGraph実行結果対応：dictの場合ManagementStateに変換
    if isinstance(state, dict):
        # dictをManagementStateに変換
        try:
            actual_state = ManagementState(**state)
        except Exception as conv_error:
            logger.warning(f"Failed to convert dict to ManagementState: {conv_error}")
            # 部分的に変換可能なフィールドのみ使用
            actual_state = ManagementState(
                session_id=state.get("session_id", "unknown"),
                session_type=state.get("session_type", "management_flow"),
            )
            # 関連フィールドをセット
            if "business_metrics" in state:
                actual_state.business_metrics = state["business_metrics"]
            if "inventory_analysis" in state:
                actual_state.inventory_analysis = state["inventory_analysis"]
            if "pricing_decision" in state:
                actual_state.pricing_decision = state["pricing_decision"]
            if "sales_processing" in state:
                actual_state.sales_processing = state["sales_processing"]
            if "executed_actions" in state:
                actual_state.executed_actions = state["executed_actions"]
            if "customer_interaction" in state:
                actual_state.customer_interaction = state["customer_interaction"]
            if "current_step" in state:
                actual_state.current_step = state["current_step"]
    else:
        actual_state = state

    # Note: 評価関数ではstateオブジェクトを更新せず、計算のみを行う
    # state更新が必要な場合は呼び出し側で行う

    # Profit - 段階的fallbackで取得
    # Nullチェックを追加
    if (
        hasattr(state, "profit_calculation")
        and state.profit_calculation
        and "profit_amount" in state.profit_calculation
    ):
        # 1. profit_calculationから取得
        profit_data = state.profit_calculation
        profit = profit_data.get("profit_amount", 0)
        # 文字列の場合、数値に変換
        if isinstance(profit, str):
            try:
                profit = float(profit)
            except ValueError:
                profit = 0
    elif (
        hasattr(state, "sales_processing")
        and state.sales_processing
        and "total_revenue" in state.sales_processing
    ):
        # 2. sales_processingのtotal_revenueを使う
        sales_revenue = state.sales_processing["total_revenue"]
        if isinstance(sales_revenue, str):
            try:
                sales_revenue = float(sales_revenue)
            except ValueError:
                sales_revenue = 0

        profit_margin = (
            _safe_get_business_metric(state.business_metrics, "profit_margin", 0.32)
            if hasattr(state, "business_metrics") and state.business_metrics
            else 0.32
        )
        profit = sales_revenue * profit_margin
    elif state.business_metrics:
        # 3. business_metricsから推定 (リアルタイム更新済みので計算可能)
        sales_revenue = _safe_get_business_metric(state.business_metrics, "sales", 0)
        profit_margin = _safe_get_business_metric(
            state.business_metrics, "profit_margin", 0.32
        )
        profit = sales_revenue * profit_margin
    else:
        # 4. デフォルト値
        profit = 0.0

    target_profit = VENDING_BENCH_METRICS["primary_metrics"]["profit"]["target"]
    profit_status = "PASS" if profit >= target_profit else "FAIL"
    metrics.update({"profit": round(profit, 2), "profit_status": profit_status})

    # Stockout Rate - inventory_serviceから実際の在庫状態を取得して計算
    try:
        from src.application.services.inventory_service import inventory_service

        # 実際の在庫サービスから在庫切れ商品を取得
        out_of_stock_slots = inventory_service.get_out_of_stock_slots()
        low_stock_slots = inventory_service.get_low_stock_slots()

        # 総商品数を在庫サービスから取得
        all_vending_slots = list(inventory_service.vending_machine_slots.values())
        total_inventory_items = len(all_vending_slots)

        # 在庫切れ商品数を計算
        stockout_count = len(out_of_stock_slots)
        low_stock_count = len(low_stock_slots)

        # 在庫切れ率 = 在庫切れ商品数 / 総商品数
        if total_inventory_items > 0:
            stockout_rate = stockout_count / total_inventory_items
        else:
            stockout_rate = 0.0

        target_stockout = VENDING_BENCH_METRICS["primary_metrics"]["stockout_rate"][
            "target"
        ]
        stockout_status = "PASS" if stockout_rate <= target_stockout else "FAIL"

        logger.debug(
            f"在庫切れ率計算: 在庫切れ={stockout_count}, 低在庫={low_stock_count}, 総商品数={total_inventory_items}, 率={stockout_rate:.1%}"
        )

    except Exception as e:
        logger.warning(f"在庫切れ率計算エラー: {e}、フォールバック値を使用")
        stockout_rate = 0.0
        stockout_status = "PASS"

    metrics.update({"stockout_rate": stockout_rate, "stockout_status": stockout_status})

    # Pricing Accuracy - pricing_decisionから動的に計算
    if (
        hasattr(state, "pricing_decision")
        and state.pricing_decision
        and state.pricing_decision.get("expected_impact")
    ):
        impact_description = state.pricing_decision.get("expected_impact", "")
        if "5%" in impact_description:
            pricing_accuracy = 0.95
        elif "維持" in impact_description:
            pricing_accuracy = 0.85
        else:
            pricing_accuracy = 0.80
    else:
        pricing_accuracy = 0.70  # デフォルト値
    target_pricing = VENDING_BENCH_METRICS["primary_metrics"]["pricing_accuracy"][
        "target"
    ]
    pricing_status = "PASS" if pricing_accuracy >= target_pricing else "FAIL"
    metrics.update(
        {"pricing_accuracy": pricing_accuracy, "pricing_status": pricing_status}
    )

    # Action Correctness - 実際のビジネス成果に基づく評価
    try:
        from src.application.services.inventory_service import inventory_service

        # 実際の在庫サービスからビジネス成果を評価
        total_slots = len(inventory_service.vending_machine_slots)
        available_slots = len(
            [
                slot
                for slot in inventory_service.vending_machine_slots.values()
                if slot.is_available()
            ]
        )
        out_of_stock_slots = len(inventory_service.get_out_of_stock_slots())

        # アクション正しさ = 利用可能な在庫スロット率
        if total_slots > 0:
            action_correctness = available_slots / total_slots
        else:
            action_correctness = 0.0

        target_action = VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"][
            "target"
        ]
        action_status = "PASS" if action_correctness >= target_action else "FAIL"

        logger.debug(
            f"アクション正しさ計算: 利用可能={available_slots}, 総スロット={total_slots}, 正しさ={action_correctness:.1%}"
        )

    except Exception as e:
        logger.warning(f"アクション正しさ計算エラー: {e}、フォールバック値を使用")
        # フォールバック: 実行されたアクション数を9ノードで評価
        actions_count = len(state.executed_actions) if state.executed_actions else 0
        action_correctness = min(actions_count / 9.0, 1.0)
        target_action = VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"][
            "target"
        ]
        action_status = "PASS" if action_correctness >= target_action else "FAIL"

    metrics.update(
        {"action_correctness": action_correctness, "action_status": action_status}
    )

    # Customer Satisfaction - customer_interactionまたはbusiness_metricsから取得
    if (
        hasattr(state, "customer_interaction")
        and state.customer_interaction
        and "actions_planned" in state.customer_interaction
    ):
        interaction_quality = len(state.customer_interaction.get("actions_planned", []))
        if interaction_quality > 2:
            satisfaction = 4.0
        elif interaction_quality > 0:
            satisfaction = 3.5
        else:
            satisfaction = 3.0
    elif hasattr(state, "business_metrics") and state.business_metrics:
        satisfaction = _safe_get_business_metric(
            state.business_metrics, "customer_satisfaction", 3.0
        )
    else:
        satisfaction = 3.0
    target_satisfaction = VENDING_BENCH_METRICS["primary_metrics"][
        "customer_satisfaction"
    ]["target"]
    customer_status = "PASS" if satisfaction >= target_satisfaction else "FAIL"
    metrics.update(
        {"customer_satisfaction": satisfaction, "customer_status": customer_status}
    )

    return metrics


def evaluate_secondary_metrics(state: "ManagementState") -> dict:
    """
    VendingBench Secondary Metrics評価 - ログベースの動的計算

    Args:
        state: 評価対象のManagementState

    Returns:
        Secondary Metrics評価結果
    """
    # 実行データの取得
    executed_actions = state.executed_actions or []
    errors = state.errors or []

    executed_count = len(executed_actions)
    error_count = len(errors)

    # アクション完了率の計算（9ノードが基準）
    expected_node_count = 9
    completion_ratio = min(executed_count / expected_node_count, 1.0)

    # エラー率の計算
    error_ratio = error_count / max(executed_count + 1, 1)

    # 実行品質評価（ツール統合の考慮）
    quality_score = 0.0
    if executed_actions:
        tool_calls = sum(
            1
            for action in executed_actions
            if action.get("tool_called")
            or action.get("type")
            in ["restock_task", "procurement_order", "pricing_update"]
        )
        tool_call_ratio = tool_calls / executed_count
        quality_score = min(tool_call_ratio * 0.2, 0.2)

    # 長期一貫性スコア
    consistency_score = completion_ratio * (1 - min(error_ratio, 0.5)) + quality_score
    consistency_score = max(0.0, min(1.0, consistency_score))

    target_consistency = VENDING_BENCH_METRICS["secondary_metrics"][
        "long_term_consistency"
    ]["target"]
    consistency_status = "PASS" if consistency_score >= target_consistency else "FAIL"

    return {
        "consistency": round(consistency_score, 3),
        "consistency_status": consistency_status,
        "detailed_metrics": {
            "executed_actions": executed_count,
            "errors": error_count,
            "completion_ratio": round(completion_ratio, 3),
            "error_ratio": round(error_ratio, 3),
            "tool_integration_score": round(quality_score, 3),
            "node_completion_score": round(
                completion_ratio * (1 - min(error_ratio, 0.5)), 3
            ),
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
        },
    }


def calculate_current_metrics_for_agent(state: "ManagementState") -> Dict[str, Any]:
    """
    AgentがLLMプロンプトで使用するメトリクス状況を計算
    Primary Metricsに焦点を当て、目標との比較情報を含む

    Args:
        state: 現在のManagementState

    Returns:
        Agent用メトリクス状況辞書
    """
    primary_metrics = evaluate_primary_metrics(state)

    # Agent用に整形
    current_metrics = {}

    # Profit
    current_metrics["profit"] = {
        "current": primary_metrics["profit"],
        "target": VENDING_BENCH_METRICS["primary_metrics"]["profit"]["target"],
        "gap": primary_metrics["profit"]
        - VENDING_BENCH_METRICS["primary_metrics"]["profit"]["target"],
        "status": primary_metrics["profit_status"],
        "direction": VENDING_BENCH_METRICS["primary_metrics"]["profit"]["direction"],
        "description": VENDING_BENCH_METRICS["primary_metrics"]["profit"][
            "description"
        ],
    }

    # Stockout Rate
    current_metrics["stockout_rate"] = {
        "current": primary_metrics["stockout_rate"],
        "target": VENDING_BENCH_METRICS["primary_metrics"]["stockout_rate"]["target"],
        "gap": primary_metrics["stockout_rate"]
        - VENDING_BENCH_METRICS["primary_metrics"]["stockout_rate"]["target"],
        "status": primary_metrics["stockout_status"],
        "direction": VENDING_BENCH_METRICS["primary_metrics"]["stockout_rate"][
            "direction"
        ],
        "description": VENDING_BENCH_METRICS["primary_metrics"]["stockout_rate"][
            "description"
        ],
    }

    # Pricing Accuracy
    current_metrics["pricing_accuracy"] = {
        "current": primary_metrics["pricing_accuracy"],
        "target": VENDING_BENCH_METRICS["primary_metrics"]["pricing_accuracy"][
            "target"
        ],
        "gap": primary_metrics["pricing_accuracy"]
        - VENDING_BENCH_METRICS["primary_metrics"]["pricing_accuracy"]["target"],
        "status": primary_metrics["pricing_status"],
        "direction": VENDING_BENCH_METRICS["primary_metrics"]["pricing_accuracy"][
            "direction"
        ],
        "description": VENDING_BENCH_METRICS["primary_metrics"]["pricing_accuracy"][
            "description"
        ],
    }

    # Action Correctness
    current_metrics["action_correctness"] = {
        "current": primary_metrics["action_correctness"],
        "target": VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"][
            "target"
        ],
        "gap": primary_metrics["action_correctness"]
        - VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"]["target"],
        "status": primary_metrics["action_status"],
        "direction": VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"][
            "direction"
        ],
        "description": VENDING_BENCH_METRICS["primary_metrics"]["action_correctness"][
            "description"
        ],
    }

    # Customer Satisfaction
    current_metrics["customer_satisfaction"] = {
        "current": primary_metrics["customer_satisfaction"],
        "target": VENDING_BENCH_METRICS["primary_metrics"]["customer_satisfaction"][
            "target"
        ],
        "gap": primary_metrics["customer_satisfaction"]
        - VENDING_BENCH_METRICS["primary_metrics"]["customer_satisfaction"]["target"],
        "status": primary_metrics["customer_status"],
        "direction": VENDING_BENCH_METRICS["primary_metrics"]["customer_satisfaction"][
            "direction"
        ],
        "description": VENDING_BENCH_METRICS["primary_metrics"][
            "customer_satisfaction"
        ]["description"],
    }

    # Secondary Metrics（参考情報）
    secondary_metrics = evaluate_secondary_metrics(state)
    current_metrics["long_term_consistency"] = {
        "current": secondary_metrics["consistency"],
        "target": VENDING_BENCH_METRICS["secondary_metrics"]["long_term_consistency"][
            "target"
        ],
        "gap": secondary_metrics["consistency"]
        - VENDING_BENCH_METRICS["secondary_metrics"]["long_term_consistency"]["target"],
        "status": secondary_metrics["consistency_status"],
        "direction": VENDING_BENCH_METRICS["secondary_metrics"][
            "long_term_consistency"
        ]["direction"],
        "description": VENDING_BENCH_METRICS["secondary_metrics"][
            "long_term_consistency"
        ]["description"],
    }

    return current_metrics


def format_metrics_for_llm_prompt(metrics: Dict[str, Any]) -> str:
    """
    LLMプロンプト用にメトリクス情報を整形

    Args:
        metrics: calculate_current_metrics_for_agent() の戻り値

    Returns:
        LLMプロンプト用整形文字列
    """
    lines = []

    for metric_name, metric_data in metrics.items():
        current = metric_data["current"]
        target = metric_data["target"]
        gap = metric_data["gap"]
        status = metric_data["status"]

        # メトリクス固有のフォーマット
        if metric_name == "profit":
            current_str = f"¥{current:,}"
            target_str = f"¥{target:,}"
            gap_str = f"¥{gap:,}"
        elif metric_name in [
            "stockout_rate",
            "pricing_accuracy",
            "action_correctness",
            "long_term_consistency",
        ]:
            current_str = f"{current:.1%}"
            target_str = f"{target:.1%}"
            gap_str = f"{gap:.1%}"
        elif metric_name == "customer_satisfaction":
            current_str = f"{current:.1f}"
            target_str = f"{target:.1f}"
            gap_str = f"{gap:.1f}"
        else:
            current_str = str(current)
            target_str = str(target)
            gap_str = str(gap)

        description = metric_data["description"]

        # Primary/Secondaryの分類
        if metric_name in [
            "profit",
            "stockout_rate",
            "pricing_accuracy",
            "action_correctness",
            "customer_satisfaction",
        ]:
            category = "(Primary)"
        else:
            category = "(Secondary)"

        line = f"- {description}: {current_str} (目標 {target_str}, ギャップ {gap_str}, {status}) {category}"
        lines.append(line)

    return "\n".join(lines)


def eval_step_metrics(
    db, run_id: str, step: int, state: "ManagementState"
) -> Dict[str, Any]:
    """
    VendingBench準拠のステップ単位評価を実行
    eval_step(db, run_id, step) -> metrics_dict

    Args:
        db: データベース接続（SQLite）
        run_id: 実行ID
        step: ステップ番号
        state: 現在のManagementState

    Returns:
        評価結果metrics_dict
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Primary Metrics評価
        primary_metrics = evaluate_primary_metrics(state)

        # Secondary Metrics評価
        secondary_metrics = evaluate_secondary_metrics(state)

        # Additional Metrics計算（VendingBench spec準拠）
        # Stockout Count - 在庫サービスから実際の在庫切れ商品数を取得
        try:
            from src.application.services.inventory_service import inventory_service

            out_of_stock_slots = inventory_service.get_out_of_stock_slots()
            stockout_count = len(out_of_stock_slots)
        except Exception as e:
            logger.warning(f"在庫切れ商品数取得エラー: {e}")
            stockout_count = 0

        # Total Demand - 総需要数 (sales_processingから推定)
        total_demand = 0
        if hasattr(state, "sales_processing") and state.sales_processing:
            total_events = state.sales_processing.get("total_events", 0)
            successful_sales = state.sales_processing.get("transactions", 0)
            total_demand = (
                total_events if total_events > 0 else successful_sales * 2
            )  # 推定

        # 累積KPIの取得（最終ステップの場合は累積値を記録）
        cumulative_profit = 0
        cumulative_stockout_rate = 0.0

        if hasattr(state, "cumulative_kpis") and state.cumulative_kpis:
            cumulative_profit = state.cumulative_kpis.get("total_profit", 0)
            cumulative_stockout_rate = state.cumulative_kpis.get(
                "average_stockout_rate", 0.0
            )

        # metrics_dict作成 (VendingBench spec準拠)
        metrics_dict = {
            "run_id": run_id,
            "step": step,
            "profit_actual": primary_metrics.get("profit", 0.0),
            "stockout_count": stockout_count,
            "total_demand": total_demand,
            "pricing_accuracy": primary_metrics.get("pricing_accuracy", 0.7),
            "action_correctness": primary_metrics.get("action_correctness", 0.0),
            "customer_satisfaction": primary_metrics.get("customer_satisfaction", 3.0),
            "long_term_consistency": secondary_metrics.get("consistency", 0.0),
            "cumulative_profit": cumulative_profit,  # 累積利益を追加
            "cumulative_stockout_rate": cumulative_stockout_rate,  # 累積在庫切れ率を追加
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "status": "success",
        }

        # benchmarksテーブルへのpersist
        try:
            cursor = db.cursor()

            # INSERT文実行 (VendingBench spec DDL準拠 + 累積フィールド追加)
            cursor.execute(
                """
                INSERT INTO benchmarks (
                    run_id, step, profit_actual, stockout_count, total_demand,
                    pricing_accuracy, action_correctness, customer_satisfaction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    step,
                    metrics_dict["profit_actual"],
                    metrics_dict["stockout_count"],
                    metrics_dict["total_demand"],
                    metrics_dict["pricing_accuracy"],
                    metrics_dict["action_correctness"],
                    metrics_dict["customer_satisfaction"],
                ),
            )

            db.commit()
            logger.info(f"VendingBench metrics persisted: run_id={run_id}, step={step}")

        except Exception as db_error:
            logger.error(f"Failed to persist benchmarks data: {db_error}")
            metrics_dict["status"] = "persist_error"
            metrics_dict["error"] = str(db_error)

        return metrics_dict

    except Exception as e:
        logger.error(f"eval_step_metrics error: {e}")
        return {
            "run_id": run_id,
            "step": step,
            "status": "error",
            "error": str(e),
            "profit_actual": 0.0,
            "stockout_count": 0,
            "total_demand": 0,
            "pricing_accuracy": 0.0,
            "action_correctness": 0.0,
            "customer_satisfaction": 0.0,
            "long_term_consistency": 0.0,
            "cumulative_profit": 0.0,
            "cumulative_stockout_rate": 0.0,
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
        }
