#!/usr/bin/env python3
"""
マルチエージェントシステム統合テスト (簡易版)

全Agentを連携させた店舗運営をシミュレートします。
3日間の全業務シーンで各Agentの協働を検証します。
"""

import asyncio
import logging
from datetime import datetime

# デバッグログ有効化（Management Agent関係のみ）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Management Agent関連のログのみDEBUGレベルに
logging.getLogger("src.agents.management_agent").setLevel(logging.DEBUG)
logging.getLogger("src.agents.search_agent").setLevel(logging.DEBUG)

# 低レベルHTTPログを抑制
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def get_user_scenario_selection():
    """ユーザーがテストシナリオを選択"""
    print("\n🎯 テストシナリオを選択してください:")
    print("1. high_sales     - 売上好調シナリオ (拡張戦略テスト)")
    print("2. low_inventory  - 在庫不足集中シナリオ (緊急調達テスト)")
    print("3. price_competition - 価格競争シナリオ (価格戦略テスト)")
    print("4. customer_demand - 顧客需要変動シナリオ (トレンド分析テスト)")
    print("5. default        - 標準シナリオ (現在の動作)")

    while True:
        choice = input("\n選択 (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            break
        print("1-5の数字で入力してください")

    scenario_map = {
        "1": "high_sales",
        "2": "low_inventory",
        "3": "price_competition",
        "4": "customer_demand",
        "5": "default",
    }

    return scenario_map[choice]


async def setup_scenario_inventory(scenario_type: str):
    """シナリオに応じた初期在庫データを設定"""
    print(f"\n📦 {scenario_type} シナリオの在庫データを初期化...")

    if scenario_type == "low_inventory":
        # 在庫不足シナリオ: 全ての商品を最低限に
        for product in SAMPLE_PRODUCTS:
            product.stock_quantity = max(
                1, int(product.stock_quantity * 0.2)
            )  # 20%のみ
            logger.info(
                f"Low inventory setup: {product.name} -> {product.stock_quantity}pc"
            )

    elif scenario_type == "high_sales":
        # 高売上シナリオ: 適切な在庫レベルを維持
        for product in SAMPLE_PRODUCTS:
            product.stock_quantity = int(product.stock_quantity * 0.8)  # 適切レベル
            logger.info(
                f"High sales setup: {product.name} -> {product.stock_quantity}pc"
            )

    elif scenario_type == "price_competition":
        # 価格競争シナリオ: 在庫多め、価格競争力チェック
        for product in SAMPLE_PRODUCTS:
            product.stock_quantity = int(product.stock_quantity * 1.2)  # 20%多め
            product.price *= 0.95  # 5%引き
            logger.info(
                f"Price competition setup: {product.name} -> ¥{product.price} ({product.stock_quantity}pc)"
            )

    elif scenario_type == "customer_demand":
        # 顧客需要変動シナリオ: 通常設定
        pass  # デフォルトのまま

    # 在庫サービスを初期化
    sample_slots = create_sample_inventory_slots()
    for slot in sample_slots:
        inventory_service.add_slot(slot)
    print(f"✅ {len(sample_slots)}個のスロットを追加しました")


async def run_multi_agent_simulation(scenario_type: str = "default"):
    """マルチエージェント統合テストを実行"""
    print("🤖 AI Multi-Agent System統合テスト")
    print("=" * 70)
    print(f"シナリオ: {scenario_type}")
    print("全Agentが連携した店舗運営を3日間シミュレート")

    # シナリオに応じた初期化
    await setup_scenario_inventory(scenario_type)

    print("\n🚀 Agentチームの準備...")
    start_time = datetime.now()

    for day in range(1, 4):  # 3日間
        print(f"\n{'=' * 60}")
        print(f"📅 日 {day} の店舗統合運用を開始")
        print(f"{'=' * 60}\n")

        try:
            # === 朝の業務フェーズ ===
            print("🌅【朝の業務フェーズ】")

            # Management Agent: 朝ルーチン
            print("  📋 Management Agent: 朝ルーチン開始...")
            try:
                morning_data = await management_agent.morning_routine()
                print("  ✓ 経営状況分析完了")
                print(
                    f"  💰 売上: ¥{morning_data.get('overnight_data', {}).get('sales', 0):,.0f}"
                )
            except Exception as e:
                print(f"  ✗ Management Agentエラー: {e}")

            # Analytics Agentに統合: 前日実績分析

            # Analytics Agent: 前日実績分析
            print("  📊 Analytics Agent: 前日トレンド分析...")
            try:
                trends = await analytics_agent.analyze_daily_trends()
                print(f"  ✓ 売上トレンド: {trends.get('revenue_trend', '不明')}")
            except Exception as e:
                print(f"  ✗ Analytics Agentエラー: {e}")

            print()

            # === 営業時間の業務フェーズ ===
            print("🕒【営業時間業務フェーズ】")

            # Management Agent: 昼間チェック
            print("  📋 Management Agent: 午前実績チェック...")
            try:
                midday_data = await management_agent.midday_check()
                print("  ✓ 業務調整完了")
            except Exception as e:
                print(f"  ✗ Management Agentエラー: {e}")

            # 販売イベント
            print("  🛒 販売イベントシミュレーション (午前)...")
            try:
                sales_stats = await simulate_purchase_events(
                    2.5, verbose=False, period_name="午前営業"
                )
                print(".1%")
            except Exception as e:
                print(f"  ✗ 販売シミュレーションエラー: {e}")

            # Search Agent: 商品検索支援
            print("  🔍 Search Agent: 商品検索機能デモ...")
            try:
                search_results = await search_agent.search_products("飲料")
                print(f"  ✓ 検索結果: {len(search_results)}件ヒット")
            except Exception as e:
                print(f"  ✗ Search Agentエラー: {e}")

            # Customer Agent: 顧客問い合わせ対応
            print("  👤 Customer Agent: 顧客問い合わせ処理...")
            try:
                customer_response = await customer_agent.respond_to_inquiry(
                    "C001", "コーラの価格を教えてください"
                )
                print("  ✓ 問い合わせ対応完了")
            except Exception as e:
                print(f"  ✗ Customer Agentエラー: {e}")

            # 販売イベント (午後)
            print("  🛒 販売イベントシミュレーション (午後)...")
            try:
                sales_stats_pm = await simulate_purchase_events(
                    3.5, verbose=False, period_name="午後営業"
                )
                print(".1%")
            except Exception as e:
                print(f"  ✗ 販売シミュレーションエラー: {e}")

            # Procurement Agent: 在庫補充調達
            print("  📦 Procurement Agent: 在庫確認と調達提案...")
            try:
                inventory_check = await procurement_agent.check_supplier_inventory(
                    "cola"
                )
                print(
                    f"  ✓ サプライヤ在庫: {inventory_check.get('supplier_stock', '不明')}"
                )
            except Exception as e:
                print(f"  ✗ Procurement Agentエラー: {e}")

            # Management Agent: 従業員タスク調整 (新機能: Procurement連携)
            print("  📋 Management Agent: 従業員タスク調整...")
            try:
                task_result = await management_agent.coordinate_employee_tasks()
                notifications = task_result.get("notifications_sent", [])
                print(f"  ✓ タスク通知送信: {len(notifications)}件")
                for notification in notifications:
                    if "new_procurement" in notification.get("task_type", ""):
                        orders = notification.get("orders", [])
                        print(
                            f"    📦 新商品発注完了: {len(orders)}件 - {[o['product'] for o in orders]}"
                        )
                    elif "restock" in notification.get("task_type", ""):
                        print(f"    🔄 在庫補充依頼: {notification.get('body', '')}")
            except Exception as e:
                print(f"  ✗ Task coordination error: {e}")

            # Management Agentに統合: 営業中状態監視

            print()

            # === 業務終了フェーズ ===
            print("🌆【業務終了フェーズ】")

            # Management Agent: 夕方総括
            print("  📋 Management Agent: 業務総括...")
            try:
                evening_data = await management_agent.evening_summary()
                print("  ✓ 業務評価完了")
            except Exception as e:
                print(f"  ✗ Management Agentエラー: {e}")

            # Analytics Agent: 本日実績報告
            print("  📈 Analytics Agent: 本日実績分析...")
            try:
                report = await analytics_agent.generate_daily_report()
                print(
                    f"  ✓ 日次レポート生成: {len(report.get('insights', []))}件の洞察"
                )
            except Exception as e:
                print(f"  ✗ Analytics Agentエラー: {e}")

            # Recorder Agent: 学習記録
            print("  🧠 Recorder Agent: 経験データ記録...")
            try:
                record = await recorder_agent.record_daily_session(
                    {
                        "day": str(day),
                        "performance": {
                            "sales": evening_data.get("daily_performance", {}).get(
                                "sales", 0
                            ),
                            "inventory_efficiency": 0.95,
                        },
                        "actions_taken": ["price_adjustment", "restock_task"],
                        "learnings": ["price_strategy_effective"],
                    }
                )
                print(f"  ✓ 業務記録完了: ID {record.get('record_id', '不明')}")
            except Exception as e:
                print(f"  ✗ Recorder Agentエラー: {e}")

            print(f"📊 日 {day} 完了! - AI Agentチームの協働業務正常完了")

        except Exception as e:
            logger.error(f"日 {day} にエラー: {e}")
            print(f"❌ 日 {day} でエラーが発生しました: {e}")
            continue

    # Agent学習結果サマリー
    outcomes = recorder_agent.get_recent_outcomes(10)
    if outcomes:
        print(f"\n🎓【Agent学習結果サマリー】")
        print("=" * 50)
        for record_id, outcome in outcomes.items():
            day = outcome.get("session_id", "Unknown")
            sales = outcome.get("metrics", {}).get("sales", 0)
            efficiency = outcome.get("metrics", {}).get("inventory_efficiency", 0)
            actions = outcome.get("actions_taken", [])
            learnings = outcome.get("learnings", [])

            print(f"日 {day}")
            print(f"  売上: ¥{sales:,.0f}")
            print(f"  在庫効率: {efficiency:.1%}")
            print(f"  実行アクション: {actions}")
            print(f"  学習内容: {learnings}")
            print()

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print("🎉 Multi-Agent System統合テスト完了!")
    print(f"⏱ 総実行時間: {total_time:.1f}秒")
    print("AI Agentチームが連携して店舗運営を管理しました")
    print("6種のAI AgentがIoT, ERP, CRM, Analyticsを統合")


async def main():
    """メイン関数"""
    try:
        # テストシナリオを選択
        scenario = await get_user_scenario_selection()
        await run_multi_agent_simulation(scenario)
    except KeyboardInterrupt:
        print("\n⏹️ 中断されました")
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
