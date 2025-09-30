import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.payment_service import (
    PaymentService,
    PaymentSimulator,
    PaymentStatus,
    PaymentError,
    PaymentResult,
    RefundResult
)
from src.models.transaction import PaymentMethod

class TestPaymentSimulator:
    """決済シミュレーターのテスト"""

    @pytest.fixture
    def payment_simulator(self):
        """決済シミュレーターのフィクスチャ"""
        return PaymentSimulator()

    @pytest.mark.asyncio
    async def test_simulate_payment_success(self, payment_simulator):
        """決済シミュレーション成功テスト"""
        # 成功率を100%に設定
        payment_simulator.success_rate = 1.0

        result = await payment_simulator.simulate_payment(1000.0, PaymentMethod.CARD)

        assert result.success is True
        assert result.status == PaymentStatus.COMPLETED
        assert result.payment_id is not None
        assert result.fee > 0  # 手数料が計算されている

    @pytest.mark.asyncio
    async def test_simulate_payment_failure(self, payment_simulator):
        """決済シミュレーション失敗テスト"""
        # 成功率を0%に設定
        payment_simulator.success_rate = 0.0

        result = await payment_simulator.simulate_payment(1000.0, PaymentMethod.CARD)

        assert result.success is False
        assert result.status == PaymentStatus.FAILED
        assert result.error_code is not None
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_simulate_payment_with_processing_delay(self, payment_simulator):
        """決済シミュレーション処理遅延テスト"""
        start_time = datetime.now()

        # 処理遅延を1-2秒に設定
        payment_simulator.processing_delay = (1.0, 2.0)
        payment_simulator.success_rate = 1.0

        result = await payment_simulator.simulate_payment(1000.0, PaymentMethod.CARD)

        elapsed_time = (datetime.now() - start_time).total_seconds()

        assert result.success is True
        assert elapsed_time >= 1.0  # 最低1秒の遅延が発生

    def test_generate_payment_id(self, payment_simulator):
        """決済ID生成テスト"""
        payment_id = payment_simulator._generate_payment_id()

        assert payment_id.startswith("sim_")
        assert len(payment_id) > 10  # 十分な長さがある

    def test_get_error_message(self, payment_simulator):
        """エラーメッセージ取得テスト"""
        message = payment_simulator._get_error_message(PaymentError.INSUFFICIENT_FUNDS, PaymentMethod.CARD)
        assert "残高が不足" in message

        message = payment_simulator._get_error_message(PaymentError.CARD_DECLINED, PaymentMethod.CARD)
        assert "カードが拒否" in message

class TestPaymentService:
    """決済サービスのテスト"""

    @pytest.fixture
    def payment_service(self):
        """決済サービスのフィクスチャ"""
        return PaymentService()

    @pytest.mark.asyncio
    async def test_process_card_payment_success(self, payment_service):
        """カード決済処理成功テスト"""
        # シミュレーターをモック
        with patch.object(payment_service.simulator, 'simulate_payment', new_callable=AsyncMock) as mock_simulate:
            mock_result = PaymentResult(
                success=True,
                status=PaymentStatus.COMPLETED,
                payment_id="test_payment_123",
                fee=30.0
            )
            mock_simulate.return_value = mock_result

            result = await payment_service.process_payment(1000.0, PaymentMethod.CARD)

            assert result.success is True
            assert result.payment_id == "test_payment_123"
            assert result.fee == 30.0

    @pytest.mark.asyncio
    async def test_process_cash_payment(self, payment_service):
        """現金決済処理テスト"""
        result = await payment_service.process_payment(500.0, PaymentMethod.CASH)

        assert result.success is True
        assert result.status == PaymentStatus.COMPLETED
        assert result.payment_id is not None

    @pytest.mark.asyncio
    async def test_process_coupon_payment_valid(self, payment_service):
        """有効クーポン決済テスト"""
        result = await payment_service.process_payment(
            1000.0,
            PaymentMethod.COUPON,
            coupon_code="DISCOUNT10"
        )

        assert result.success is True
        assert result.status == PaymentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_coupon_payment_invalid(self, payment_service):
        """無効クーポン決済テスト"""
        result = await payment_service.process_payment(
            1000.0,
            PaymentMethod.COUPON,
            coupon_code="INVALID_COUPON"
        )

        assert result.success is False
        assert "無効なクーポンコード" in result.error_message

    @pytest.mark.asyncio
    async def test_process_payment_zero_amount(self, payment_service):
        """ゼロ金額決済テスト"""
        result = await payment_service.process_payment(0.0, PaymentMethod.CARD)

        assert result.success is False
        assert "0より大きくなければなりません" in result.error_message

    @pytest.mark.asyncio
    async def test_refund_payment_success(self, payment_service):
        """返金処理成功テスト"""
        # 取引をモック
        mock_transaction = MagicMock()
        mock_transaction.total_amount = 1000.0

        with patch.object(payment_service, '_get_transaction', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_transaction

            # シミュレーターのスリープをモック
            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await payment_service.refund_payment("test_transaction_123")

                assert result.success is True
                assert result.refund_id is not None

    @pytest.mark.asyncio
    async def test_refund_payment_transaction_not_found(self, payment_service):
        """取引が見つからない場合の返金テスト"""
        with patch.object(payment_service, '_get_transaction', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            result = await payment_service.refund_payment("invalid_transaction")

            assert result.success is False
            assert "取引が見つかりません" in result.error_message

    @pytest.mark.asyncio
    async def test_refund_payment_amount_exceeds(self, payment_service):
        """返金額が取引金額を超える場合のテスト"""
        mock_transaction = MagicMock()
        mock_transaction.total_amount = 500.0

        with patch.object(payment_service, '_get_transaction', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_transaction

            result = await payment_service.refund_payment("test_transaction", amount=1000.0)

            assert result.success is False
            assert "返金金額が取引金額を超えています" in result.error_message

    def test_log_transaction(self, payment_service):
        """取引ログ記録テスト"""
        initial_count = len(payment_service.transaction_log)

        payment_service._log_transaction({
            "type": "test",
            "amount": 1000,
            "timestamp": datetime.now()
        })

        assert len(payment_service.transaction_log) == initial_count + 1

        # ログ制限テスト（100件制限）
        payment_service.transaction_log = [
            {"id": i} for i in range(100)
        ]

        payment_service._log_transaction({"type": "test"})
        assert len(payment_service.transaction_log) == 100  # 最新100件に制限

    def test_get_payment_stats(self, payment_service):
        """決済統計取得テスト"""
        # ログをクリア
        payment_service.transaction_log = []

        # サンプルログを追加
        payment_service.transaction_log = [
            {
                "type": "payment",
                "amount": 1000,
                "result": PaymentResult(success=True, status=PaymentStatus.COMPLETED, fee=30.0)
            },
            {
                "type": "payment",
                "amount": 2000,
                "result": PaymentResult(success=False, status=PaymentStatus.FAILED)
            }
        ]

        stats = payment_service.get_payment_stats()

        assert stats["total_transactions"] == 2
        assert stats["successful_transactions"] == 1
        assert stats["failed_transactions"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["total_amount"] == 1000
        assert stats["total_fees"] == 30.0

    def test_get_recent_transactions(self, payment_service):
        """最近の取引取得テスト"""
        # ログをクリア
        payment_service.transaction_log = []

        # サンプルログを追加
        for i in range(5):
            payment_service._log_transaction({
                "type": "payment",
                "amount": 1000 + i * 100,
                "timestamp": datetime.now()
            })

        recent = payment_service.get_recent_transactions(limit=3)

        assert len(recent) == 3
        assert recent[0]["amount"] == 1400  # 最新のものから

    def test_simulate_realistic_sale(self, payment_service):
        """現実的な販売シミュレーションテスト"""
        sale_result = payment_service.simulate_realistic_sale("drink_cola", quantity=2)

        assert "product_id" in sale_result
        assert "predicted_demand" in sale_result
        assert "actual_quantity" in sale_result
        assert "unit_price" in sale_result
        assert "total_amount" in sale_result
        assert "customer_satisfaction" in sale_result
        assert "simulation_model" in sale_result

        # 価格が正の値であることを確認
        assert sale_result["unit_price"] > 0
        assert sale_result["total_amount"] > 0

        # 顧客満足度が0-1の範囲であることを確認
        assert 0 <= sale_result["customer_satisfaction"] <= 1

    def test_get_demand_forecast(self, payment_service):
        """需要予測テスト"""
        forecast = payment_service.get_demand_forecast("drink_cola", days=3)

        assert forecast["product_id"] == "drink_cola"
        assert forecast["forecast_period_days"] == 3
        assert "forecast_data" in forecast
        assert "summary" in forecast

        # 予測データが適切な構造であることを確認
        assert len(forecast["forecast_data"]) == 3
        assert "predicted_demand" in forecast["forecast_data"][0]
        assert "expected_sales" in forecast["forecast_data"][0]

        # サマリー統計が計算されていることを確認
        assert "total_predicted_demand" in forecast["summary"]
        assert "peak_demand_hour" in forecast["summary"]

    def test_simulate_market_scenario(self, payment_service):
        """市場シナリオシミュレーションテスト"""
        scenario_result = payment_service.simulate_market_scenario("economic_boom")

        assert scenario_result["scenario"] == "economic_boom"
        assert "scenario_parameters" in scenario_result
        assert "market_trends" in scenario_result
        assert "scenario_impact" in scenario_result
        assert "recommendations" in scenario_result

        # シナリオパラメータが適切であることを確認
        params = scenario_result["scenario_parameters"]
        assert "demand_change" in params
        assert "price_sensitivity" in params

        # 市場トレンドが全商品に対して計算されていることを確認
        trends = scenario_result["market_trends"]
        expected_products = ["drink_cola", "drink_tea", "snack_chips", "snack_chocolate"]
        for product in expected_products:
            assert product in trends

    def test_get_advanced_analytics(self, payment_service):
        """高度な分析データ取得テスト"""
        # 取引ログを追加してテストデータを準備
        payment_service.transaction_log = [
            {
                "type": "payment",
                "amount": 1000,
                "timestamp": datetime.now()
            }
        ]

        analytics = payment_service.get_advanced_analytics()

        assert "hourly_statistics" in analytics
        assert "product_analytics" in analytics
        assert "market_trends" in analytics
        assert "inventory_efficiency" in analytics
        assert "sales_model_version" in analytics

        # 商品分析が全商品に対して存在することを確認
        product_analytics = analytics["product_analytics"]
        expected_products = ["drink_cola", "drink_tea", "snack_chips", "snack_chocolate"]
        for product in expected_products:
            assert product in product_analytics
            assert "popularity_score" in product_analytics[product]
            assert "optimal_price" in product_analytics[product]

    def test_sales_model_demand_prediction(self, payment_service):
        """販売モデル需要予測テスト"""
        # 異なる時間帯での予測をテスト
        morning_demand = payment_service.sales_model.predict_demand("drink_cola", 8, 1)  # 月曜朝8時
        afternoon_demand = payment_service.sales_model.predict_demand("drink_cola", 14, 1)  # 月曜午後2時

        # 午後の方が需要が高いはず
        assert afternoon_demand > morning_demand

        # 需要が適切な範囲内であることを確認
        assert 0 <= morning_demand <= 2.0
        assert 0 <= afternoon_demand <= 2.0

    def test_customer_behavior_simulation(self, payment_service):
        """顧客行動シミュレーションテスト"""
        behavior = payment_service.sales_model.simulate_customer_behavior("drink_cola")

        required_keys = [
            "purchase_probability",
            "satisfaction_score",
            "repeat_purchase_probability",
            "price_sensitivity",
            "promotion_responsiveness"
        ]

        for key in required_keys:
            assert key in behavior

        # 確率が適切な範囲内であることを確認
        assert 0 <= behavior["purchase_probability"] <= 1
        assert 0 <= behavior["satisfaction_score"] <= 1
        assert 0 <= behavior["repeat_purchase_probability"] <= 1

    def test_optimal_pricing(self, payment_service):
        """最適価格計算テスト"""
        cost_price = 100.0
        optimal_price = payment_service.sales_model.calculate_optimal_price("drink_cola", cost_price)

        # コスト価格より高い価格が設定されていることを確認（マージン考慮）
        assert optimal_price > cost_price

        # 価格が現実的な範囲内であることを確認
        assert optimal_price <= cost_price * 3  # 最大3倍まで

        # 価格が10円単位であることを確認
        assert optimal_price % 10 == 0

    def test_bulk_purchase_discount(self, payment_service):
        """まとめ買い割引テスト"""
        base_price = 150.0

        # 単品購入（割引なし）
        single_price = payment_service.sales_model.simulate_bulk_purchase_discount(base_price, 1)
        assert single_price == base_price

        # 3個購入（5%割引）
        triple_price = payment_service.sales_model.simulate_bulk_purchase_discount(base_price, 3)
        assert triple_price < base_price
        assert triple_price == base_price * 0.95  # 5%割引

        # 5個購入（10%割引）
        bulk_price = payment_service.sales_model.simulate_bulk_purchase_discount(base_price, 5)
        assert bulk_price == base_price * 0.9  # 10%割引

    def test_inventory_turnover_calculation(self, payment_service):
        """在庫回転率計算テスト"""
        turnover_rate = payment_service.sales_model.calculate_inventory_turnover(
            "drink_cola", 100, 5.0  # 在庫100個、1日平均販売5個
        )

        # 年間販売量 = 5 * 365 = 1825
        # 在庫回転率 = 1825 / 100 = 18.25
        expected_rate = 5.0 * 365 / 100
        assert abs(turnover_rate - expected_rate) < 0.01

    def test_seasonal_demand_simulation(self, payment_service):
        """季節需要シミュレーションテスト"""
        # 夏（7月）と冬（1月）の需要を比較
        summer_demand = payment_service.sales_model.simulate_seasonal_demand("drink_cola", 7)
        winter_demand = payment_service.sales_model.simulate_seasonal_demand("drink_cola", 1)

        # 夏の方が需要が高いはず
        assert summer_demand > winter_demand

        # 需要が適切な範囲内であることを確認
        assert summer_demand > 0
        assert winter_demand > 0

    def test_market_scenario_recommendations(self, payment_service):
        """市場シナリオ推奨事項テスト"""
        # 好景気シナリオ
        boom_scenario = payment_service.simulate_market_scenario("economic_boom")
        boom_recommendations = boom_scenario["recommendations"]

        # 不景気シナリオ
        recession_scenario = payment_service.simulate_market_scenario("recession")
        recession_recommendations = recession_scenario["recommendations"]

        # 両方のシナリオで推奨事項が生成されていることを確認
        assert len(boom_recommendations) > 0
        assert len(recession_recommendations) > 0

        # シナリオによって推奨事項が異なることを確認
        assert boom_recommendations != recession_recommendations

    def test_inventory_efficiency_rating(self, payment_service):
        """在庫効率性評価テスト"""
        # 高い回転率
        high_rating = payment_service._rate_inventory_efficiency(15)
        assert high_rating == "非常に効率的"

        # 標準的な回転率
        normal_rating = payment_service._rate_inventory_efficiency(6)
        assert normal_rating == "標準的"

        # 低い回転率
        low_rating = payment_service._rate_inventory_efficiency(1)
        assert low_rating == "非常に非効率的"

# パラメータ化テスト
@pytest.mark.parametrize("payment_method,expected_success", [
    (PaymentMethod.CASH, True),      # 現金は常に成功
    (PaymentMethod.COUPON, True),    # 有効クーポンは成功
    (PaymentMethod.CARD, None),      # カードはシミュレーションによる
    (PaymentMethod.MOBILE, None),    # モバイルはシミュレーションによる
])
@pytest.mark.asyncio
async def test_process_payment_methods(payment_method, expected_success, payment_service):
    """決済方法別テスト"""
    if payment_method == PaymentMethod.COUPON:
        # クーポンの場合は有効なコードを指定
        result = await payment_service.process_payment(1000.0, payment_method, coupon_code="DISCOUNT10")
    else:
        result = await payment_service.process_payment(1000.0, payment_method)

    if expected_success is True:
        assert result.success is True
    elif expected_success is False:
        assert result.success is False

# エラーハンドリングテスト
@pytest.mark.asyncio
async def test_payment_service_error_handling():
    """決済サービスのエラーハンドリングテスト"""
    service = PaymentService()

    # シミュレーターで例外が発生する場合
    with patch.object(service.simulator, 'simulate_payment', new_callable=AsyncMock) as mock_simulate:
        mock_simulate.side_effect = Exception("シミュレーターエラー")

        result = await service.process_payment(1000.0, PaymentMethod.CARD)

        assert result.success is False
        assert "シミュレーターエラー" in result.error_message

# 統計テスト
def test_payment_stats_empty_log(payment_service):
    """空ログの統計テスト"""
    payment_service.transaction_log = []

    stats = payment_service.get_payment_stats()

    assert stats["total_transactions"] == 0
    assert stats["success_rate"] == 0.0

# 返金テスト
@pytest.mark.asyncio
async def test_refund_service_integration(payment_service):
    """返金サービスの統合テスト"""
    # まず決済を実行
    payment_result = await payment_service.process_payment(1000.0, PaymentMethod.CARD)

    if payment_result.success:
        # 返金を実行
        refund_result = await payment_service.refund_payment(payment_result.payment_id)

        # 返金は高い成功率で成功するはず
        assert refund_result.success is True or "失敗" in refund_result.error_message

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
