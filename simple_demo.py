"""
AI自動販売機シミュレーター シンプルデモ

このスクリプトは、システムの機能をテスト・デモするために使用します。
"""

import asyncio

from src.accounting.journal_entry import journal_processor
from src.agents.customer_agent import customer_agent
from src.agents.search_agent import search_agent
from src.analytics.event_tracker import event_tracker
from src.models.product import SAMPLE_PRODUCTS
from src.services.inventory_service import inventory_service
from src.services.payment_service import payment_service


async def main():
    print("🤖 AI自動販売機シミュレーター デモ")
    print("=" * 50)

    # 1. システム概要
    print("\n📋 システム概要:")
    print(f"   • 商品数: {len(SAMPLE_PRODUCTS)}")
    print("   • 決済サービス: シミュレーションモデル統合済み")
    print("   • 在庫サービス: 統合管理システム稼働中")
    print("   • 検索エージェント: AI価格比較機能有効")
    print("   • 顧客エージェント: AI顧客対応機能有効")
    print("   • 会計システム: 複式簿記自動処理機能有効")
    print("   • イベント追跡: リアルタイム監視システム稼働中")

    # 2. 商品一覧
    print("\n📦 登録商品一覧:")
    for i, product in enumerate(SAMPLE_PRODUCTS, 1):
        print(
            f"   {i}. {product.name} - ¥{product.price} (在庫: {product.stock_quantity})"
        )

    # 3. 決済シミュレーション
    print("\n💳 決済シミュレーション:")
    from src.models.transaction import PaymentMethod

    # 現金決済テスト
    result = await payment_service.process_payment(500, PaymentMethod.CASH)
    print(f"   • 現金決済 ¥500: {'✅ 成功' if result.success else '❌ 失敗'}")

    # カード決済テスト
    result = await payment_service.process_payment(1000, PaymentMethod.CARD)
    print(f"   • カード決済 ¥1000: {'✅ 成功' if result.success else '❌ 失敗'}")

    # 4. AIエージェントテスト
    print("\n🤖 AIエージェントテスト:")

    # 検索エージェントの統計と実テスト
    try:
        # 実検索テスト（コカ・コーラ）
        print("   • 検索エージェント検索テスト（コカ・コーラ）...")
        search_results = await search_agent.search_products(
            "コカ・コーラ", max_results=3
        )
        print(f"     - 検索結果数: {len(search_results)}")
        if search_results:
            for i, result in enumerate(search_results[:3], 1):
                price_str = f"¥{result['price']:.0f}" if result["price"] else "価格不明"
                print(
                    f"       {i}. {result['title'][:50]}... - {price_str} - {result['source']}"
                )

        # 統計表示
        search_stats = search_agent.get_search_stats()
        print(f"   • 検索履歴: {search_stats.get('total_searches', 0)}回")
    except Exception as e:
        print(f"   • 検索エージェント: エラー - {e}")

    # 顧客エージェントの統計
    try:
        # 顧客エージェントの基本情報確認
        print("   • 顧客エージェント: AIモデル連携済み")
        # AIモデル連携テスト
        from src.ai.model_manager import model_manager

        print(f"     - プライマリモデル: {model_manager.primary_model}")
        print(f"     - 利用可能モデル数: {len(model_manager.models)}")
    except Exception as e:
        print(f"   • 顧客エージェント: エラー - {e}")

    # 5. 販売シミュレーションモデル
    print("\n🏪 販売シミュレーションモデル:")

    # 需要予測
    for product in SAMPLE_PRODUCTS:
        current_time = asyncio.get_event_loop().time()
        demand = payment_service.sales_model.predict_demand(
            product.product_id,
            14,
            3,  # 午後2時、木曜日
        )
        print(f"   • {product.name}: 予測需要 {demand:.2f}")

    # 価格最適化
    for product in SAMPLE_PRODUCTS:
        optimal_price = payment_service.sales_model.calculate_optimal_price(
            product.product_id, product.cost
        )
        print(f"   • {product.name}: 最適価格 ¥{optimal_price:.0f}")

    # 5. 在庫管理
    print("\n📦 在庫管理:")
    try:
        summary = inventory_service.get_inventory_summary()
        print(f"   • 総スロット数: {summary.total_slots}")
        print(f"   • アクティブスロット: {summary.active_slots}")
        print(f"   • 在庫切れスロット: {summary.out_of_stock_slots}")
    except Exception as e:
        print(f"   • 在庫情報取得エラー: {e}")

    # 6. 会計システム
    print("\n💰 会計システム:")
    try:
        trial_balance = journal_processor.get_trial_balance()
        print(f"   • 借方合計: ¥{trial_balance['total_debit']:,}")
        print(f"   • 貸方合計: ¥{trial_balance['total_credit']:,}")
        print(f"   • 勘定科目数: {len(trial_balance['accounts'])}")
    except Exception as e:
        print(f"   • 会計情報取得エラー: {e}")

    # 7. 高度な分析
    print("\n📊 高度な分析:")
    try:
        # 現実的な販売シミュレーション
        sale_result = payment_service.simulate_realistic_sale("drink_cola", quantity=2)
        print(f"   • 予測需要: {sale_result['predicted_demand']:.2f}")
        print(f"   • 実際販売数: {sale_result['actual_quantity']}")
        print(f"   • 総売上: ¥{sale_result['total_amount']}")

        # 需要予測
        forecast = payment_service.get_demand_forecast("drink_cola", days=2)
        print(
            f"   • 2日間総需要予測: {forecast['summary']['total_predicted_demand']:.2f}"
        )

        # 市場シナリオ
        scenario = payment_service.simulate_market_scenario("economic_boom")
        print(f"   • 好景気シナリオ: {len(scenario['recommendations'])}個の推奨事項")

    except Exception as e:
        print(f"   • 高度分析エラー: {e}")

    print(f"\n{'=' * 50}")
    print("🎉 デモ完了！")
    print("💡 詳細な使い方は README.md を参照してください。")
    print("🌐 Web UI: http://localhost:8000")
    print("📚 APIドキュメント: http://localhost:8000/docs")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
