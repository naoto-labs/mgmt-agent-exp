#!/usr/bin/env python3
"""
3日間店舗運営シミュレーション (簡易版)

Management Agentを使った3日間の店舗運営をシミュレートします。
各日で朝・昼・夕の業務ルーチンを実行します。
"""

import asyncio
import logging
from datetime import datetime

from src.agents.management_agent import SessionBasedManagementAgent
from src.models.inventory import create_sample_inventory_slots
from src.services.inventory_service import inventory_service
from src.simulations.sales_simulation import simulate_purchase_events

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)


async def run_three_day_simulation():
    """3日間シミュレーションを実行"""
    print("🏪 AIエージェント自律型自動販売機 3日間運用シミュレーション")
    print("=" * 60)
    print("各日の業務ルーチンをAIエージェントが自律的に実行します")

    print("\n")
    print("🏪 在庫データを初期化...")
    sample_slots = create_sample_inventory_slots()
    for slot in sample_slots:
        inventory_service.add_slot(slot)
    print(f"✅ {len(sample_slots)}個のスロットを追加しました")
    print()

    agent = SessionBasedManagementAgent(provider="openai")
    start_time = datetime.now()

    for day in range(1, 4):  # 3日間
        print(f"\n{'=' * 50}")
        print(f"📅 日 {day} の店舗運用を開始")
        print(f"{'=' * 50}\n")

        try:
            # 朝のルーチン
            print("🌅【朝のルーチン (9:00)】")
            morning_result = await agent.morning_routine()
            print("✓ 完了\n")

            # 結果表示
            metrics = morning_result.get('overnight_data', {})
            decision = morning_result.get('decisions', {})
            print("📈 朝の状況分析:")
            sales_value = metrics.get('sales', 0)
            print(f"  💰 売上データ確認: {sales_value:,.0f}円")
            inventory_level = metrics.get('inventory_level', '')
            print(f"  📦 在庫レベル: {inventory_level}")
            satisfaction = metrics.get('customer_satisfaction', 0)
            print(f"  😊 顧客満足度: {satisfaction:.1f}/5.0")
            print(f"📋 AI意思決定: {decision.get('decision', '不明')[:80]}...")
            actions = decision.get('actions', [])
            print(f"💰 推奨アクション: {', '.join(actions[:3])}")
            print()

            # 午前の販売イベント
            await simulate_purchase_events(1.8, verbose=False, period_name="午前")

            # 昼のチェック
            print("☀️【昼のチェック (12:00)】")
            midday_result = await agent.midday_check()
            print("✓ 完了\n")

            # 結果表示
            midday_metrics = midday_result.get('metrics', {})
            midday_analysis = midday_result.get('analysis', {})
            midday_decision = midday_result.get('decisions', {})
            print("📊 午前実績:")
            midday_sales = midday_metrics.get('sales', 0)
            print(f"  💰 売上: {midday_sales:,.0f}円")
            analysis_text = midday_analysis.get('analysis', '')[:80]
            print(f"  📊 財務分析: {analysis_text}...")
            decision_text = midday_decision.get('decision', '不明')[:80]
            print(f"📋 AI意思決定: {decision_text}...")
            print()

            # 午後の販売イベント
            await simulate_purchase_events(3.0, verbose=False, period_name="午後")

            # 夕方の総括
            print("🌆【夕方の総括 (17:00)】")
            evening_result = await agent.evening_summary()
            print("✓ 完了\n")

            # 夕方の販売イベント
            await simulate_purchase_events(2.0, verbose=False, period_name="夕方")

            # 結果表示
            daily_perf = evening_result.get('daily_performance', {})
            inventory_status = evening_result.get('inventory_status', {})
            lessons = evening_result.get('lessons_learned', [])
            print("📈 本日の総合実績:")
            final_sales = daily_perf.get('sales', 0)
            print(f"  💰 売上総額: ¥{final_sales:,.0f}円")
            inv_status = inventory_status.get('status', 'unknown')
            print(f"  📦 在庫ステータス: {inv_status}")
            low_stock = inventory_status.get('low_stock_items', [])
            print(f"  ⚠️ 在庫低レベル商品: {low_stock}")
            inventory_level = daily_perf.get('inventory_level', {})
            total_inventory = sum(inventory_level.values())
            print(f"  📊 現在の総在庫数: {total_inventory}個")
            print(f"  📋 商品別在庫: {inventory_level}")
            print(f"📚 学んだ教訓: {', '.join(lessons[:3])}")
            print()

            print(f"📊 日 {day} 完了!")

        except Exception as e:
            logger.error(f"日 {day} にエラー: {e}")
            print(f"❌ 日 {day} でエラーが発生しました: {e}")
            continue

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print("🎉 3日間店舗運用シミュレーション完了!")
    print(f"⏱ 総実行時間: {total_time:.1f}秒")
    print("AIエージェントが自律的に店舗運営を管理しました")


async def main():
    """メイン関数"""
    try:
        await run_three_day_simulation()
    except KeyboardInterrupt:
        print("\n⏹️ 中断されました")
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
