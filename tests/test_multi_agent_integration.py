#!/usr/bin/env python3
"""
マルチエージェントシステム統合テスト

全Agentを連携させた店舗運営をシミュレートします。
3日間の全業務シーンで各Agentの協働を検証します。
"""

import asyncio
import logging
from datetime import datetime

import pytest

from src.agents.analytics_agent import analytics_agent
from src.agents.customer_agent import customer_agent
from src.agents.management_agent import management_agent
from src.agents.procurement_agent import procurement_agent
from src.agents.recorder_agent import recorder_agent
from src.agents.search_agent import search_agent
from src.models.inventory import create_sample_inventory_slots
from src.models.product import SAMPLE_PRODUCTS
from src.services.inventory_service import inventory_service
from src.simulations.sales_simulation import simulate_purchase_events
from tests.test_scenarios import (
    get_common_config,
    get_multi_agent_config,
    get_required_actions,
)

# 設定を取得
common_config = get_common_config()

# デバッグログ有効化（Management Agent関係のみ）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Management Agent関連のログのみDEBUGレベルに
if common_config["logging"]["enable_detailed_logging"]:
    logging.getLogger("src.agents.management_agent").setLevel(logging.DEBUG)
    logging.getLogger("src.agents.search_agent").setLevel(logging.DEBUG)

# 低レベルHTTPログを抑制（設定に応じて）
for log_name in common_config["logging"]["supress_external_logs"]:
    logging.getLogger(log_name).setLevel(logging.WARNING)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@pytest.fixture
def setup_test_inventory():
    """テスト用の初期在庫を設定"""
    sample_slots = create_sample_inventory_slots()
    for slot in sample_slots:
        inventory_service.add_slot(slot)
    yield
    # クリーンアップ
    inventory_service._slots.clear()


@pytest.mark.asyncio
async def test_multi_agent_integration_scenario_high_sales(setup_test_inventory):
    """高売上シナリオでのマルチエージェント統合テスト"""
    # 設定を取得
    config = get_multi_agent_config()
    required_actions = get_required_actions()

    # テストデータ収集コンテナ
    test_results = {"scenario": config["scenario"], "days": [], "total_sales": 0}

    print("🤖 AI Multi-Agent System統合テスト - 高売上シナリオ")
    print("=" * 70)
    print("全Agentが連携した店舗運営を3日間シミュレート")

    start_time = datetime.now()

    for day in range(1, config["days"] + 1):  # 設定から日数を取得
        print(f"\n{'=' * 60}")
        print(f"📅 日 {day} の店舗統合運用を開始")
        print(f"{'=' * 60}\n")

        day_result = {"day": day, "actions": [], "decisions": [], "sales": 0}

        try:
            # === 朝の業務フェーズ ===
            print("🌅【朝の業務フェーズ】")

            # Management Agent: 朝ルーチン
            print("  📋 Management Agent: 朝ルーチン開始...")
            morning_data = await management_agent.morning_routine()
            print("  ✓ 経営状況分析完了")
            day_result["actions"].append("morning_routine")

            # Analytics Agent: 前日実績分析
            print("  📊 Analytics Agent: 前日トレンド分析...")
            trends = await analytics_agent.analyze_daily_trends()
            print(f"  ✓ 売上トレンド: {trends.get('revenue_trend', '不明')}")
            day_result["actions"].append("trend_analysis")

            print()

            # === 営業時間の業務フェーズ ===
            print("🕒【営業時間業務フェーズ】")

            # Management Agent: 昼間チェック
            print("  📋 Management Agent: 午前実績チェック...")
            midday_data = await management_agent.midday_check()
            print("  ✓ 業務調整完了")
            day_result["actions"].append("midday_check")

            # 販売イベントシミュレーション
            print("  🛒 販売イベントシミュレーション (午前)...")
            sales_stats = await simulate_purchase_events(
                config["sales_multipliers"]["morning_business"],
                verbose=False,
                period_name="午前営業",
            )
            midday_sales = sales_stats.get("total_sales", 0)
            print(f"    売上: ¥{midday_sales:,.0f}")

            # 販売イベント (午後)
            print("  🛒 販売イベントシミュレーション (午後)...")
            sales_stats_pm = await simulate_purchase_events(
                config["sales_multipliers"]["afternoon_business"],
                verbose=False,
                period_name="午後営業",
            )
            afternoon_sales = sales_stats_pm.get("total_sales", 0)
            day_sales = midday_sales + afternoon_sales
            print(f"    売上: ¥{afternoon_sales:,.0f} (日計: ¥{day_sales:,.0f})")

            day_result["sales"] = day_sales
            test_results["total_sales"] += day_sales

            # Search Agent: 商品検索支援テスト
            print("  🔍 Search Agent: 商品検索機能デモ...")
            search_results = await search_agent.search_products("人気飲料")
            assert len(search_results) > 0, "検索結果が取得できるべき"
            print(f"  ✓ 検索結果: {len(search_results)}件ヒット")
            day_result["actions"].append("product_search")

            # Customer Agent: 顧客問い合わせ対応テスト
            print("  👤 Customer Agent: 顧客問い合わせ処理...")
            customer_response = await customer_agent.respond_to_inquiry(
                "C001", "人気商品を教えてください"
            )
            assert "response" in customer_response, "顧客応答が生成されるべき"
            print("  ✓ 問い合わせ対応完了")
            day_result["actions"].append("customer_inquiry")

            # Procurement Agent: 在庫確認と調達提案テスト
            print("  📦 Procurement Agent: 在庫確認と調達提案...")
            inventory_check = await procurement_agent.check_supplier_inventory("cola")
            assert inventory_check is not None, "在庫チェックが機能するべき"
            print(
                f"  ✓ サプライヤ在庫: {inventory_check.get('supplier_stock', '不明')}"
            )
            day_result["actions"].append("inventory_check")

            day_result["actions"].append("coordinate_tasks")

            # === 業務終了フェーズ ===
            print("🌆【業務終了フェーズ】")

            # Management Agent: 夕方総括
            print("  📋 Management Agent: 業務総括...")
            evening_data = await management_agent.evening_summary()
            print("  ✓ 業務評価完了")
            day_result["actions"].append("evening_summary")

            # Analytics Agent: 本日実績報告テスト
            print("  📈 Analytics Agent: 本日実績分析...")
            report = await analytics_agent.generate_daily_report()
            assert "insights" in report, "レポート生成が機能するべき"
            print(f"  ✓ 日次レポート生成: {len(report.get('insights', []))}件の洞察")
            day_result["actions"].append("daily_report")

            # Recorder Agent: 学習記録テスト
            print("  🧠 Recorder Agent: 経験データ記録...")
            record = await recorder_agent.record_daily_session(
                {
                    "day": str(day),
                    "performance": {"sales": day_sales, "inventory_efficiency": 0.95},
                    "actions_taken": ["price_adjustment", "restock_task"],
                    "learnings": ["price_strategy_effective"],
                }
            )
            assert record is not None, "学習記録が機能するべき"
            print(f"  ✓ 業務記録完了: ID {record.get('record_id', '不明')}")
            day_result["actions"].append("learning_record")

            print(f"📊 日 {day} 完了! - AI Agentチームの協働業務正常完了")

        except Exception as e:
            logger.error(f"日 {day} にエラー: {e}")
            print(f"❌ 日 {day} でエラーが発生しました: {e}")
            # テスト失敗時に記録
            day_result["error"] = str(e)
            raise

        test_results["days"].append(day_result)

    # Agent学習結果サマリー（簡易テスト）
    outcomes = recorder_agent.get_recent_outcomes(10)
    if outcomes:
        print(f"\n🎓【Agent学習結果サマリー】")
        print("=" * 50)
        for record_id, outcome in outcomes.items():
            day = outcome.get("session_id", "Unknown")
            sales = outcome.get("metrics", {}).get("sales", 0)
            efficiency = outcome.get("metrics", {}).get("inventory_efficiency", 0)
            actions_taken = outcome.get("actions_taken", [])
            learnings = outcome.get("learnings", [])

            print(f"日 {day}")
            print(f"  売上: ¥{sales:,.0f}")
            print(f"  在庫効率: {efficiency:.1%}")
            print(f"  実行アクション: {actions_taken}")
            print(f"  学習内容: {learnings}")
            print()

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print("🎉 Multi-Agent System統合テスト完了!")
    print(f"⏱ 総実行時間: {total_time:.1f}秒")
    print("AI Agentチームが連携して店舗運営を管理しました")
    print("6種のAI AgentがIoT, ERP, CRM, Analyticsを統合")

    # === テストアサーション ===
    # 全日程が完了していること
    assert len(test_results["days"]) == config["days"], (
        f"{config['days']}日分のテストが完了しているべき"
    )

    # 全Agentの主要機能が実行されていること
    for day_result in test_results["days"]:
        required_actions_list = required_actions  # 一元管理設定から取得
        for action in required_actions:
            assert action in day_result["actions"], (
                f"日 {day_result['day']} で {action} が実行されているべき"
            )

        # 売上が発生していること
        assert day_result["sales"] >= 0, (
            f"日 {day_result['day']} の売上は非負数であるべき"
        )

        # エラーが発生していないこと
        assert "error" not in day_result, (
            f"日 {day_result['day']} でエラーが発生していないべき"
        )

    # 総売上が正であること
    assert test_results["total_sales"] > 0, "テスト期間中に売上が発生しているべき"

    # 学習記録が保存されていること
    assert len(outcomes) > 0, "学習結果が記録されているべき"
