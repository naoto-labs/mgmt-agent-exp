"""
continuous_multi_day_simulation.py - 連続多日経営管理シミュレーション

test_case_a.pyをベースとして、複数日にわたる連続経営管理を実行
メモリ引き継ぎと過去実績データ活用により、より高度な戦略的意思決定を実現
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.append(os.path.join(os.getcwd(), "src"))
import asyncio

from src.agents.management_agent import ManagementState, management_agent
from src.agents.management_agent.agent import MetricsEvaluatingStateGraph
from src.agents.management_agent.evaluation_metrics import (
    create_benchmarks_table,
    eval_step_metrics,
)
from src.agents.management_agent.metrics_tracker import VendingBenchMetricsTracker


async def prepare_next_day_state(
    final_state: ManagementState,
    next_day: int
) -> ManagementState:
    """
    前日の最終状態から翌日の初期状態を作成
    メモリ引き継ぎと過去実績データの維持を行う

    Args:
        final_state: 前日の最終状態
        next_day: 翌日の日付番号 (1, 2, ...)

    Returns:
        翌日の初期状態
    """
    print(f"📅 Day {next_day} 状態準備: 前日データ引き継ぎ")

    # 前日状態をコピー
    next_state = final_state.model_copy()

    # 日次管理更新
    next_state.day_sequence = next_day
    next_state.business_date = (
        final_state.business_date + timedelta(days=1)
        if final_state.business_date
        else datetime.now().date()
    )

    # 会話メモリの引き継ぎ
    next_state.memory_snapshot = final_state.memory_snapshot
    next_state.learned_patterns = final_state.learned_patterns

    # 過去洞察の積算（長期トレンド分析用）
    if final_state.feedback:
        insight_entry = {
            "day": final_state.day_sequence,
            "date": datetime.now().isoformat(),
            "strategic_insights": final_state.feedback.get("strategic_insights", {}),
            "tomorrow_priorities": final_state.feedback.get("tomorrow_priorities", []),
            "business_health": final_state.feedback.get("business_health", {}),
            "performance_summary": final_state.final_report.get("analyses_completed", {}),
        }
        next_state.historical_insights.append(insight_entry)

    # 前日データの引き継ぎ（Multi-day運用フィールド）
    next_state.previous_day_carry_over = {
        "day": final_state.day_sequence,
        "profit_amount": final_state.profit_calculation.get("profit_amount", 0)
        if final_state.profit_calculation
        else 0,
        "inventory_levels": (
            final_state.business_metrics.get("inventory_level", {})
            if final_state.business_metrics
            else {}
        ),
        "customer_satisfaction": (
            final_state.business_metrics.get("customer_satisfaction", 3.0)
            if final_state.business_metrics
            else 3.0
        ),
        "final_report": final_state.final_report or {},
        "executed_actions_count": len(final_state.executed_actions),
        "errors_count": len(final_state.errors),
    }

    # 累積KPIの継承
    # cumulative_kpisはincrementalに更新されるためそのまま引き継ぎ

    # 翌日開始状態にリセット
    next_state.current_step = "inventory_check"
    next_state.processing_status = "pending"
    next_state.business_metrics = None  # 当日データを再取得

    # 当日リセット（新規開始）
    next_state.actual_sales_events = []
    next_state.executed_actions = []
    next_state.errors = []

    # 各ノード分析結果のリセット（当日再分析）
    next_state.inventory_analysis = None
    next_state.sales_analysis = None
    next_state.financial_analysis = None
    next_state.pricing_decision = None
    next_state.restock_decision = None
    next_state.procurement_decision = None
    next_state.sales_processing = None
    next_state.customer_interaction = None
    next_state.profit_calculation = None
    next_state.feedback = None
    next_state.final_report = None

    return next_state


async def run_continuous_simulation(
    total_days: int = 7,
    initial_state: Optional[ManagementState] = None
) -> Dict[str, any]:
    """
    複数日にわたる連続経営管理シミュレーション

    Args:
        total_days: シミュレーション日数
        initial_state: 初期状態（Noneの場合は自動生成）

    Returns:
        シミュレーション全体の結果
    """
    print(f"=== 連続多日経営管理シミュレーション ({total_days}日間) ===")

    results = {
        "simulation_config": {
            "total_days": total_days,
            "start_date": datetime.now().isoformat(),
        },
        "daily_results": [],
        "cumulative_analytics": {},
        "issues": [],
        "overall_performance": {},
    }

    try:
        # 初期状態の設定（Day 1）
        if initial_state is None:
            current_state = ManagementState(
                session_id=f"continuous_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                session_type="multi_day_management",
                day_sequence=1,
                business_date=datetime.now().date(),
            )
        else:
            current_state = initial_state

        # Multi-day運用ための拡張状態初期化
        current_state.inventory_history = []
        current_state.sales_history = []
        current_state.performance_history = []

        print(f"✓ 初期状態作成完了: session_id={current_state.session_id}")

        # 日次ループ
        for current_day in range(1, total_days + 1):
            print(f"\n{'='*60}")
            print(f"🚀 Day {current_day}/{total_days} 実行開始")
            print(f"{'='*60}")

            try:
                # LangGraph パイプラインの初期化（日毎に再初期化）
                run_id = f"{current_state.session_id}_day{current_day}"
                db_path = "data/vending_bench.db"

                # VendingBench評価セットアップ
                conn = None
                evaluating_graph = None

                try:
                    import sqlite3

                    # データベース接続
                    conn = sqlite3.connect(db_path)
                    create_benchmarks_table(conn)

                    # 日次グラフ初期化
                    evaluating_graph = MetricsEvaluatingStateGraph(
                        management_agent, conn, run_id
                    )
                    print(f"✓ Day {current_day} LangGraphパイプライン初期化完了")

                except Exception as setup_error:
                    print(f"⚠️ LangGraphセットアップ警告: {setup_error}")
                    results["issues"].append({
                        "day": current_day,
                        "type": "graph_setup_warning",
                        "error": str(setup_error)
                    })

                # 9ノード実行（1日の意思決定サイクル）
                if evaluating_graph:
                    print(f"▶️ Day {current_day} 経営意思決定実行...")
                    final_state = await evaluating_graph.ainvoke(current_state)
                    print(f"✅ Day {current_day} 意思決定完了")
                else:
                    print(f"⚠️ Day {current_day} Graphなしで実行スキップ")
                    final_state = current_state

                # 当日の結果記録
                day_result = {
                    "day": current_day,
                    "date": final_state.business_date.isoformat() if final_state.business_date else None,
                    "session_id": run_id,
                    "processing_status": final_state.processing_status,
                    "executed_actions_count": len(final_state.executed_actions),
                    "errors_count": len(final_state.errors),
                    "inventory_items_count": len(final_state.inventory_history),
                    "sales_events_count": len(final_state.sales_history),
                    "performance_records_count": len(final_state.performance_history),
                }

                # 財務・業務指標の記録
                if final_state.profit_calculation:
                    day_result["profit_amount"] = final_state.profit_calculation.get("profit_amount", 0)
                    day_result["total_revenue"] = final_state.profit_calculation.get("total_revenue", 0)
                    day_result["profit_margin"] = final_state.profit_calculation.get("profit_margin", 0)

                if final_state.business_metrics:
                    day_result["customer_satisfaction"] = final_state.business_metrics.get("customer_satisfaction", 0)

                # Cumulative KPIの確認
                if hasattr(final_state, "cumulative_kpis") and final_state.cumulative_kpis:
                    day_result["cumulative_profit"] = final_state.cumulative_kpis.get("total_profit", 0)

                # 当日結果を蓄積
                results["daily_results"].append(day_result)

                print(f"📊 Day {current_day} 結果: 利益 ¥{day_result.get('profit_amount', 0):,.0f}, アクション {day_result['executed_actions_count']}件")

                # 翌日継続判定
                if current_day < total_days:
                    if final_state.processing_status == "completed":
                        # 翌日状態の準備
                        current_state = await prepare_next_day_state(final_state, current_day + 1)
                        print(f"🔄 Day {current_day + 1} 状態準備完了（メモリ・履歴引き継ぎ）")

                        # 過去実績蓄積の確認
                        print(f"   📈 歴データ: 在庫{len(current_state.inventory_history)}, 販売{len(current_state.sales_history)}, 業績{len(current_state.performance_history)}件")
                    else:
                        print(f"⚠️ Day {current_day} エラー終了のため継続中止")
                        break

                # クリーンアップ
                if conn:
                    conn.close()

                # 定期休憩
                await asyncio.sleep(0.5)

            except Exception as day_error:
                print(f"❌ Day {current_day} エラー: {day_error}")
                results["issues"].append({
                    "day": current_day,
                    "type": "day_execution_error",
                    "error": str(day_error)
                })

                # エラー時も継続は試みるが、状態は維持
                if current_day < total_days:
                    try:
                        current_state = await prepare_next_day_state(final_state, current_day + 1)
                        print(f"⚠️ Day {current_day} エラー後、次の状態準備")
                    except:
                        break

        # シミュレーション完了後の全体分析
        print(f"\n{'='*60}")
        print("🧪 シミュレーション完了 - 全体分析")
        print(f"{'='*60}")

        # 全体パフォーマンスの計算
        if results["daily_results"]:
            total_days_completed = len(results["daily_results"])
            total_profit = sum(r.get("profit_amount", 0) for r in results["daily_results"])
            total_actions = sum(r["executed_actions_count"] for r in results["daily_results"])
            avg_customer_satisfaction = sum(
                r.get("customer_satisfaction", 3.0) for r in results["daily_results"]
            ) / len(results["daily_results"])

            results["overall_performance"] = {
                "total_days_completed": total_days_completed,
                "total_profit": total_profit,
                "average_daily_profit": total_profit / total_days_completed if total_days_completed > 0 else 0,
                "total_actions_executed": total_actions,
                "average_customer_satisfaction": round(avg_customer_satisfaction, 2),
                "errors_total": sum(r["errors_count"] for r in results["daily_results"]),
                "completion_rate": total_days_completed / total_days * 100,
            }

            # トレンド分析
            profit_trend = [r.get("profit_amount", 0) for r in results["daily_results"]]
            results["cumulative_analytics"]["profit_trend"] = profit_trend

            if len(profit_trend) > 1:
                results["cumulative_analytics"]["profit_trend_analysis"] = {
                    "increasing_days": sum(1 for i in range(1, len(profit_trend)) if profit_trend[i] > profit_trend[i-1]),
                    "best_day": profit_trend.index(max(profit_trend)) + 1,
                    "worst_day": profit_trend.index(min(profit_trend)) + 1,
                }

            print("📈 全体パフォーマンスサマリー:")
            print(f"   完了日数: {total_days_completed}/{total_days}")
            print(f"   総利益: ¥{total_profit:,.0f}")
            print(f"   1日平均利益: ¥{results['overall_performance']['average_daily_profit']:,.0f}")
            print(f"   総アクション数: {total_actions}")
            print(f"   平均顧客満足度: {avg_customer_satisfaction:.2f}")

        # 完了情報
        results["simulation_config"]["end_date"] = datetime.now().isoformat()
        results["simulation_config"]["actual_duration_days"] = len(results["daily_results"])

        print(f"✅ 連続多日シミュレーション完了 - {len(results['daily_results'])}日分のデータ生成")

    except Exception as e:
        print(f"❌ シミュレーション全体エラー: {e}")
        results["issues"].append({
            "type": "simulation_error",
            "error": str(e)
        })

    return results


async def demo_continuous_simulation():
    """連続多日経営管理のデモ実行"""
    print("=== 連続多日経営管理シミュレーション デモ ===")

    # 3日間のテスト実行
    results = await run_continuous_simulation(total_days=3)

    print("\n=== シミュレーション結果 ===")
    print(f"実行日数: {len(results['daily_results'])}")
    if results['daily_results']:
        for day_result in results['daily_results']:
            print(f"Day {day_result['day']}: 利益 ¥{day_result.get('profit_amount', 0):,.0f}, アクション {day_result['executed_actions_count']}件")


if __name__ == "__main__":
    asyncio.run(demo_continuous_simulation())</content>
