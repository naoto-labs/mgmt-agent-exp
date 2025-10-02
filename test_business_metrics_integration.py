"""
BusinessMetricsの統合テスト
SessionBasedManagementAgentが各種システムと正しく連携できているかを確認
"""

import asyncio
import logging

from src.agents.management_agent import management_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_business_metrics_integration():
    """BusinessMetricsが各種システムと連携しているかテスト"""
    print("\n=== BusinessMetrics統合テスト開始 ===\n")

    try:
        # 1. ビジネスメトリクスを取得
        print("1. ビジネスメトリクスを取得中...")
        metrics = management_agent.get_business_metrics()

        print("\n取得したメトリクス:")
        print(f"  - 売上: ¥{metrics.get('sales', 0):,.2f}")
        print(f"  - 利益率: {metrics.get('profit_margin', 0):.1%}")
        print(f"  - 在庫状況: {metrics.get('inventory_level', {})}")
        print(f"  - 顧客満足度: {metrics.get('customer_satisfaction', 0):.2f}/5.0")

        # 在庫統計情報
        if "inventory_status" in metrics:
            inv_status = metrics["inventory_status"]
            print(f"\n  在庫統計:")
            print(f"    - 総スロット数: {inv_status.get('total_slots', 0)}")
            print(f"    - 低在庫: {inv_status.get('low_stock_count', 0)}")
            print(f"    - 在庫切れ: {inv_status.get('out_of_stock_count', 0)}")

        # 売上統計情報
        if "sales_stats" in metrics:
            sales_stats = metrics["sales_stats"]
            print(f"\n  売上統計:")
            print(f"    - 取引数: {sales_stats.get('transaction_count', 0)}")
            print(f"    - 成功率: {sales_stats.get('success_rate', 0):.1%}")

        # エラーチェック
        if "error" in metrics:
            print(f"\n  ⚠️ エラー: {metrics['error']}")
            return False

        print("\n✅ ビジネスメトリクス取得成功")

        # 2. システム連携の確認
        print("\n2. システム連携の確認...")

        # 在庫サービスとの連携確認
        from src.services.inventory_service import inventory_service

        print(
            f"  - 在庫サービス: VM slots = {len(inventory_service.vending_machine_slots)}"
        )
        print(
            f"  - 在庫サービス: Storage slots = {len(inventory_service.storage_slots)}"
        )

        # 決済サービスとの連携確認
        from src.services.payment_service import PaymentService

        payment_service = PaymentService()
        payment_stats = payment_service.get_payment_stats()
        print(
            f"  - 決済サービス: 総売上 = ¥{payment_stats.get('total_revenue', 0):,.2f}"
        )

        # 管理会計との連携確認
        from src.accounting.management_accounting import management_analyzer

        print(f"  - 管理会計: analyzer = {type(management_analyzer).__name__}")

        print("\n✅ システム連携確認完了")

        # 3. データの整合性チェック
        print("\n3. データの整合性チェック...")

        has_inventory = bool(metrics.get("inventory_level"))
        has_sales = metrics.get("sales", 0) >= 0
        has_profit = metrics.get("profit_margin", 0) >= 0
        has_satisfaction = 0 <= metrics.get("customer_satisfaction", 0) <= 5

        print(f"  - 在庫データあり: {has_inventory}")
        print(f"  - 売上データ有効: {has_sales}")
        print(f"  - 利益率データ有効: {has_profit}")
        print(f"  - 顧客満足度有効: {has_satisfaction}")

        all_valid = has_inventory and has_sales and has_profit and has_satisfaction

        if all_valid:
            print("\n✅ データの整合性確認完了")
        else:
            print("\n⚠️ 一部のデータに問題がある可能性があります")

        return all_valid

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_financial_analysis():
    """財務分析機能のテスト"""
    print("\n=== 財務分析テスト開始 ===\n")

    try:
        print("財務パフォーマンスを分析中...")
        analysis = await management_agent.analyze_financial_performance()

        print("\n財務分析結果:")
        print(f"  - 分析: {analysis.get('analysis', 'N/A')}")
        print(f"  - 推奨事項数: {len(analysis.get('recommendations', []))}")

        if "recommendations" in analysis:
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"    {i}. {rec}")

        print("\n✅ 財務分析完了")
        return True

    except Exception as e:
        print(f"\n❌ 財務分析エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_inventory_check():
    """在庫状況確認のテスト"""
    print("\n=== 在庫状況確認テスト開始 ===\n")

    try:
        print("在庫状況を確認中...")
        status = await management_agent.check_inventory_status()

        print("\n在庫状況:")
        print(f"  - ステータス: {status.get('status', 'N/A')}")
        print(f"  - 低在庫商品: {status.get('low_stock_items', [])}")
        print(f"  - 発注必要: {status.get('reorder_needed', [])}")

        print("\n✅ 在庫状況確認完了")
        return True

    except Exception as e:
        print(f"\n❌ 在庫確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("=" * 80)
    print("SessionBasedManagementAgent - BusinessMetrics統合テスト")
    print("=" * 80)

    # 同期テスト
    result1 = test_business_metrics_integration()

    # 非同期テスト
    result2 = asyncio.run(test_financial_analysis())
    result3 = asyncio.run(test_inventory_check())

    # 結果サマリ
    print("\n" + "=" * 80)
    print("テスト結果サマリ")
    print("=" * 80)
    print(f"  1. ビジネスメトリクス取得: {'✅ 成功' if result1 else '❌ 失敗'}")
    print(f"  2. 財務分析: {'✅ 成功' if result2 else '❌ 失敗'}")
    print(f"  3. 在庫状況確認: {'✅ 成功' if result3 else '❌ 失敗'}")

    all_passed = result1 and result2 and result3
    print(f"\n総合結果: {'✅ すべて成功' if all_passed else '⚠️ 一部失敗'}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
