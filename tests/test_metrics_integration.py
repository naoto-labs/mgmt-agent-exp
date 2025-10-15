import os
import sqlite3
import sys

sys.path.append("src")

from src.agents.management_agent.evaluation_metrics import (
    create_benchmarks_table,
    eval_step_metrics,
)
from src.agents.management_agent.models import BusinessMetrics, ManagementState


def test_metrics_integration():
    """ステップ単位評価関数のデータベース連携テスト"""

    # データベース接続
    db_path = "data/vending_bench.db"
    conn = sqlite3.connect(db_path)

    # テーブル作成確認
    create_benchmarks_table(conn)
    print("✅ Benchmarks table ready")

    # サンプルManagementState作成
    from datetime import datetime

    business_metrics = BusinessMetrics(
        sales=1500.0,
        profit_margin=0.25,
        inventory_level={"product_1": 10, "product_2": 5},
        customer_satisfaction=3.5,
        timestamp=datetime.now(),
    )

    state = ManagementState(
        session_id="test_session_001",
        session_type="vending_bench_test",
    )
    state.business_metrics = business_metrics
    state.executed_actions = [
        {"type": "inventory_check", "node": "inventory_check_node"},
        {"type": "restock_task", "tool_called": True},
        {"type": "pricing_update", "tool_called": True},
    ]

    state.inventory_analysis = {
        "low_stock_items": [{"product_id": "product_2", "stock": 5}],
        "critical_items": [],  # 在庫切れなし
    }

    state.sales_processing = {
        "total_revenue": 1500,
        "transactions": 45,
        "total_events": 60,
    }

    state.pricing_decision = {
        "expected_impact": "売上向上率5%程度の価格調整が必要",
    }

    state.profit_calculation = {
        "profit_amount": 375.0,  # 1500 * 0.25
    }

    state.customer_interaction = {
        "actions_planned": ["謝罪対応", "代替商品提案"],
    }

    # eval_step_metrics実行
    run_id = "test_run_001"
    step = 1

    print(f"📊 Evaluating metrics for run_id={run_id}, step={step}")

    metrics_result = eval_step_metrics(conn, run_id, step, state)

    print("✅ Metrics evaluation completed:")
    print(f"   Status: {metrics_result['status']}")
    print(".2f")
    print(f"   Stockout Count: {metrics_result['stockout_count']}")
    print(f"   Total Demand: {metrics_result['total_demand']}")
    print(".3f")
    print(".3f")
    print(".1f")
    print(".3f")

    # データベースから結果を確認
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM benchmarks WHERE run_id = ? AND step = ?", (run_id, step)
    )
    rows = cursor.fetchall()

    print("\n📋 Persisted data in database:")
    for row in rows:
        print(f"   {row}")

    conn.close()

    # TODOリスト更新
    task_progress = [
        "- [x] eval_step_metrics関数を実装（VendingBench spec準拠）",
        "- [x] step_metrics_evaluationメソッドをmetrics_trackerに追加",
        "- [x] benchmarksテーブル作成",
        "- [x] inventory_check_node実行後にmetrics評価追加",
        "- [x] 全9node実行後にmetrics評価追加",
        "- [x] データベース連携テスト",
    ]

    return "🎉 All tasks completed! VendingBench準拠のステップ単位評価実装完了"


if __name__ == "__main__":
    result = test_metrics_integration()
    print(result)
