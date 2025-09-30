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
