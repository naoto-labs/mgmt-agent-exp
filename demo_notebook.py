"""
AI自動販売機シミュレーター デモノートブック

このスクリプトは、Jupyter Notebookや対話型環境で
システムの機能をテスト・デモするために使用します。
"""

import asyncio
import json
from datetime import datetime

# システムコンポーネントのインポート
from src.models.product import SAMPLE_PRODUCTS
from src.services.payment_service import payment_service
from src.services.inventory_service import inventory_service
from src.agents.search_agent import search_agent
from src.accounting.journal_entry import journal_processor
from src.analytics.event_tracker import event_tracker

def print_header(title):
    """ヘッダーを表示"""
    print(f"\n{'='*50}")
    print(f"🤖 {title}")
    print(f"{'='*50}")

def print_section(title):
    """セクションを表示"""
    print(f"\n📋 {title}")
    print("-" * 30)

async def demo_system_overview():
    """システム概要のデモ"""
    print_header("システム概要")

    print("✅ システムコンポーネント:")
    print(f"   • 商品数: {len(SAMPLE_PRODUCTS)}")
    print(f"   • 決済サービス: {payment_service.__class__.__name__}")
    print(f"   • 在庫サービス: {inventory_service.__class__.__name__}")
    print(f"   • 検索エージェント: {search_agent.__class__.__name__}")
    print(f"   • 会計システム: {journal_processor.__class__.__name__}")
    print(f"   • イベント追跡: {event_tracker.__class__.__name__}")

    # システム健全性チェック
    health = await event_tracker.get_system_health_score()
    print(f"\n🏥 システム健全性スコア: {health:.2".2f"

async def demo_products():
    """商品管理のデモ"""
    print_header("商品管理")

    print("📦 登録商品一覧:")
    for i, product in enumerate(SAMPLE_PRODUCTS, 1):
        print(f"   {i}. {product.name} - ¥{product.price} (在庫: {product.stock_quantity})")

    print("\n💰 価格分析:")
    total_value = sum(p.price * p.stock_quantity for p in SAMPLE_PRODUCTS)
    avg_price = sum(p.price for p in SAMPLE_PRODUCTS) / len(SAMPLE_PRODUCTS)
    print(f"   • 総在庫価値: ¥{total_value:,}",.0f"   • 平均価格: ¥{avg_price:.0".0f"

async def demo_payment_simulation():
    """決済シミュレーションのデモ"""
    print_header("決済シミュレーション")

    from src.models.transaction import PaymentMethod

    # 各種決済方法をテスト
    payment_methods = [
        (PaymentMethod.CASH, 500),
        (PaymentMethod.CARD, 1000),
        (PaymentMethod.COUPON, 800)
    ]

    print("💳 決済方法別テスト:")
    for method, amount in payment_methods:
        try:
            result = await payment_service.process_payment(amount, method)
            status = "✅ 成功" if result.success else "❌ 失敗"
            print(f"   • {method.value}: ¥{amount} - {status}")

            if result.success and result.payment_id:
                print(f"     決済ID: {result.payment_id}")

        except Exception as e:
            print(f"   • {method.value}: ¥{amount} - ❌ エラー: {e}")

    # クーポン決済テスト
    print("\n🎫 クーポン決済テスト:")
    try:
        coupon_result = await payment_service.process_payment(
            1000, PaymentMethod.COUPON, coupon_code="DISCOUNT10"
        )
        status = "✅ 成功" if coupon_result.success else "❌ 失敗"
        print(f"   • DISCOUNT10: {status}")
    except Exception as e:
        print(f"   • DISCOUNT10: ❌ エラー: {e}")

async def demo_sales_model():
    """販売シミュレーションモデルのデモ"""
    print_header("販売シミュレーションモデル")

    print("📊 需要予測テスト:")
    for product in SAMPLE_PRODUCTS:
        # 現在の時間で需要予測
        current_time = datetime.now()
        demand = payment_service.sales_model.predict_demand(
            product.product_id, current_time.hour, current_time.weekday()
        )
        print(f"   • {product.name}: 予測需要 {demand:.2".2f"

    print("\n💰 価格最適化テスト:")
    for product in SAMPLE_PRODUCTS:
        optimal_price = payment_service.sales_model.calculate_optimal_price(
            product.product_id, product.cost
        )
        margin = ((optimal_price - product.cost) / product.cost) * 100
        print(f"   • {product.name}: 最適価格 ¥{optimal_price:.0".0f"(マージン {margin:.1".1f"")

    print("\n📈 市場シナリオ分析:")
    scenarios = ["normal", "economic_boom", "recession"]
    for scenario in scenarios:
        result = payment_service.simulate_market_scenario(scenario)
        print(f"   • {scenario}: {len(result['recommendations'])}個の推奨事項")

async def demo_inventory_management():
    """在庫管理のデモ"""
    print_header("在庫管理")

    try:
        # 在庫サマリー取得
        summary = inventory_service.get_inventory_summary()
        print("📦 在庫サマリー:"        print(f"   • 総スロット数: {summary.total_slots}")
        print(f"   • アクティブスロット: {summary.active_slots}")
        print(f"   • 在庫切れスロット: {summary.out_of_stock_slots}")
        print(f"   • 総商品数: {summary.total_products}")

    except Exception as e:
        print(f"❌ 在庫管理エラー: {e}")

async def demo_accounting():
    """会計システムのデモ"""
    print_header("会計システム")

    try:
        # 試算表取得
        trial_balance = journal_processor.get_trial_balance()
        print("📊 試算表:"        print(f"   • 借方合計: ¥{trial_balance['total_debit']:,}",.0f"        print(f"   • 貸方合計: ¥{trial_balance['total_credit']:,}",.0f"
        print(f"   • 勘定科目数: {len(trial_balance['accounts'])}")

        # 残高確認
        cash_balance = journal_processor.get_account_balance("1000")  # 現金
        sales_balance = journal_processor.get_account_balance("4000")  # 売上高
        print("
💰 勘定残高:"        print(f"   • 現金: ¥{cash_balance:,}",.0f"        print(f"   • 売上高: ¥{sales_balance:,}",.0f"
    except Exception as e:
        print(f"❌ 会計システムエラー: {e}")

async def demo_analytics():
    """分析システムのデモ"""
    print_header("分析システム")

    try:
        # イベント統計
        event_stats = event_tracker.get_event_stats()
        print("📈 イベント統計:"        print(f"   • 総イベント数: {event_stats.get('total_events', 0)}")
        print(f"   • イベントタイプ数: {len(event_stats.get('event_types', {}))}")

        # システム健全性
        health_score = event_tracker.get_system_health_score()
        print(f"\n🏥 システム健全性スコア: {health_score:.2".2f"

        # 最近のイベント
        recent_events = event_tracker.get_recent_events(5)
        print(f"\n📅 最近のイベント ({len(recent_events)}件):")
        for event in recent_events[-3:]:  # 最新3件
            print(f"   • {event.get('timestamp', 'N/A')} - {event.get('event_type', 'unknown')}")

    except Exception as e:
        print(f"❌ 分析システムエラー: {e}")

async def demo_advanced_features():
    """高度な機能のデモ"""
    print_header("高度な機能")

    print("🔍 高度な販売分析:")
    try:
        # 現実的な販売シミュレーション
        sale_result = payment_service.simulate_realistic_sale('drink_cola', quantity=3)
        print("   • 予測需要: {sale_result['predicted_demand']".2f"}")
        print(f"   • 実際販売数: {sale_result['actual_quantity']}")
        print(f"   • 顧客満足度: {sale_result['customer_satisfaction']".2f"}")
        print(f"   • 総売上: ¥{sale_result['total_amount']}")

        # 需要予測
        forecast = payment_service.get_demand_forecast('drink_cola', days=3)
        print("
📊 3日間需要予測:"        print(f"   • 総予測需要: {forecast['summary']['total_predicted_demand']".2f"}")
        print(f"   • ピーク時間帯: {forecast['summary']['peak_demand_hour']}時")

    except Exception as e:
        print(f"❌ 高度な機能エラー: {e}")

def run_all_demos():
    """全デモを実行"""
    print("🚀 AI自動販売機シミュレーター デモ開始")
    print("=" * 60)

    async def main():
        await demo_system_overview()
        await demo_products()
        await demo_payment_simulation()
        await demo_sales_model()
        await demo_inventory_management()
        await demo_accounting()
        await demo_analytics()
        await demo_advanced_features()

        print(f"\n{'='*60}")
        print("🎉 デモ完了！")
        print("💡 詳細な使い方は README.md を参照してください。")
        print("🌐 Web UI: http://localhost:8000")
        print("📚 APIドキュメント: http://localhost:8000/docs")
        print(f"{'='*60}")

    # asyncio.run()が使用できない環境の場合のフォールバック
    try:
        asyncio.run(main())
    except RuntimeError:
        # 既にイベントループが実行中の場合
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # ネストされたイベントループで実行
            asyncio.create_task(main())
        else:
            loop.run_until_complete(main())

if __name__ == "__main__":
    run_all_demos()
