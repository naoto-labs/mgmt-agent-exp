import asyncio
import random
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.models.transaction import PaymentMethod, PaymentDetails, Transaction
from src.config.settings import settings

logger = logging.getLogger(__name__)

class PaymentStatus(str, Enum):
    """決済ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDING = "refunding"
    REFUNDED = "refunded"

class PaymentError(str, Enum):
    """決済エラー種別"""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_DECLINED = "card_declined"
    NETWORK_ERROR = "network_error"
    INVALID_CARD = "invalid_card"
    EXPIRED_CARD = "expired_card"
    SYSTEM_ERROR = "system_error"

@dataclass
class PaymentResult:
    """決済結果"""
    success: bool
    status: PaymentStatus
    payment_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None
    fee: float = 0.0

@dataclass
class RefundResult:
    """返金結果"""
    success: bool
    refund_id: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

class PaymentSimulator:
    """決済シミュレーター"""

    def __init__(self):
        self.success_rate = 0.95  # 決済成功率（95%）
        self.processing_delay = (1.0, 3.0)  # 処理時間範囲（秒）
        self.error_patterns = {
            PaymentError.INSUFFICIENT_FUNDS: 0.3,
            PaymentError.CARD_DECLINED: 0.25,
            PaymentError.NETWORK_ERROR: 0.2,
            PaymentError.INVALID_CARD: 0.15,
            PaymentError.EXPIRED_CARD: 0.1
        }

    def _generate_payment_id(self) -> str:
        """決済IDを生成"""
        return f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

    def _get_random_error(self) -> PaymentError:
        """ランダムなエラーを取得"""
        errors = list(self.error_patterns.keys())
        probabilities = list(self.error_patterns.values())
        return random.choices(errors, weights=probabilities)[0]

    def _simulate_processing_delay(self) -> float:
        """処理遅延をシミュレート"""
        min_delay, max_delay = self.processing_delay
        return random.uniform(min_delay, max_delay)

    async def simulate_payment(self, amount: float, method: PaymentMethod) -> PaymentResult:
        """決済をシミュレート"""
        logger.info(f"決済シミュレーション開始: 金額={amount}, 方法={method}")

        # 処理遅延をシミュレート
        delay = self._simulate_processing_delay()
        await asyncio.sleep(delay)

        # 決済成功/失敗を決定
        success = random.random() < self.success_rate

        if success:
            payment_id = self._generate_payment_id()
            result = PaymentResult(
                success=True,
                status=PaymentStatus.COMPLETED,
                payment_id=payment_id,
                processed_at=datetime.now(),
                fee=amount * 0.029 + 30  # Stripe手数料をシミュレート（2.9% + 30円）
            )
            logger.info(f"決済成功: {payment_id}")
        else:
            error = self._get_random_error()
            result = PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_code=error.value,
                error_message=self._get_error_message(error, method),
                processed_at=datetime.now()
            )
            logger.warning(f"決済失敗: {error.value} - {result.error_message}")

        return result

    def _get_error_message(self, error: PaymentError, method: PaymentMethod) -> str:
        """エラーメッセージを取得"""
        messages = {
            PaymentError.INSUFFICIENT_FUNDS: "残高が不足しています",
            PaymentError.CARD_DECLINED: "カードが拒否されました",
            PaymentError.NETWORK_ERROR: "ネットワークエラーが発生しました",
            PaymentError.INVALID_CARD: "カード情報が無効です",
            PaymentError.EXPIRED_CARD: "カードの有効期限が切れています",
            PaymentError.SYSTEM_ERROR: "システムエラーが発生しました"
        }
        return messages.get(error, "不明なエラー")

class PaymentService:
    """決済サービス"""

    def __init__(self):
        self.simulator = PaymentSimulator()
        self.transaction_log: List[Dict[str, Any]] = []

    async def process_payment(self, amount: float, method: PaymentMethod, **kwargs) -> PaymentResult:
        """決済を処理"""
        logger.info(f"決済処理開始: 金額={amount}, 方法={method}")

        try:
            # 入力検証
            if amount <= 0:
                raise ValueError("決済金額は0より大きくなければなりません")

            if method not in [PaymentMethod.CARD, PaymentMethod.MOBILE]:
                # 現金とクーポンは即時成功として処理
                if method == PaymentMethod.CASH:
                    return PaymentResult(
                        success=True,
                        status=PaymentStatus.COMPLETED,
                        payment_id=self.simulator._generate_payment_id(),
                        processed_at=datetime.now()
                    )
                elif method == PaymentMethod.COUPON:
                    return await self._process_coupon_payment(amount, **kwargs)

            # カード/モバイル決済のシミュレーション
            result = await self.simulator.simulate_payment(amount, method)

            # ログ記録
            self._log_transaction({
                "type": "payment",
                "amount": amount,
                "method": method,
                "result": result,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            logger.error(f"決済処理エラー: {e}")
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message=str(e),
                processed_at=datetime.now()
            )

    async def _process_coupon_payment(self, amount: float, coupon_code: str = None) -> PaymentResult:
        """クーポン決済を処理"""
        if not coupon_code:
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="クーポンコードが必要です",
                processed_at=datetime.now()
            )

        # クーポン検証をシミュレート（簡易版）
        valid_coupons = {
            "DISCOUNT10": 0.1,
            "DISCOUNT20": 0.2,
            "FREE": 1.0
        }

        if coupon_code not in valid_coupons:
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="無効なクーポンコードです",
                processed_at=datetime.now()
            )

        discount_rate = valid_coupons[coupon_code]
        discount_amount = amount * discount_rate

        return PaymentResult(
            success=True,
            status=PaymentStatus.COMPLETED,
            payment_id=self.simulator._generate_payment_id(),
            processed_at=datetime.now(),
            fee=0.0  # クーポンは手数料なし
        )

    async def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> RefundResult:
        """決済を返金"""
        logger.info(f"返金処理開始: 取引ID={transaction_id}")

        try:
            # 取引の検証（実際の実装ではデータベースから取得）
            transaction = await self._get_transaction(transaction_id)
            if not transaction:
                return RefundResult(
                    success=False,
                    error_message="取引が見つかりません",
                    processed_at=datetime.now()
                )

            # 返金金額の決定（指定がない場合は全額）
            refund_amount = amount if amount is not None else transaction.total_amount

            if refund_amount > transaction.total_amount:
                return RefundResult(
                    success=False,
                    error_message="返金金額が取引金額を超えています",
                    processed_at=datetime.now()
                )

            # 返金処理のシミュレーション
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)

            # 返金成功をシミュレート（高い成功率）
            success = random.random() < 0.98

            if success:
                refund_id = f"refund_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
                result = RefundResult(
                    success=True,
                    refund_id=refund_id,
                    processed_at=datetime.now()
                )
                logger.info(f"返金成功: {refund_id}")
            else:
                result = RefundResult(
                    success=False,
                    error_message="返金処理に失敗しました",
                    processed_at=datetime.now()
                )
                logger.warning(f"返金失敗: {transaction_id}")

            # ログ記録
            self._log_transaction({
                "type": "refund",
                "transaction_id": transaction_id,
                "refund_amount": refund_amount,
                "result": result,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            logger.error(f"返金処理エラー: {e}")
            return RefundResult(
                success=False,
                error_message=str(e),
                processed_at=datetime.now()
            )

    async def _get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """取引を取得（実際の実装ではデータベースから）"""
        # 簡易的な実装：ログから検索
        for log_entry in reversed(self.transaction_log):
            if (log_entry["type"] == "payment" and
                log_entry.get("payment_id") == transaction_id):
                return Transaction(
                    transaction_id=transaction_id,
                    machine_id="VM001",
                    items=[],  # 簡易版のため空
                    subtotal=0,
                    total_amount=log_entry["amount"]
                )
        return None

    def _log_transaction(self, log_entry: Dict[str, Any]):
        """取引ログを記録"""
        self.transaction_log.append(log_entry)

        # ログ数の制限（最新100件のみ保持）
        if len(self.transaction_log) > 100:
            self.transaction_log = self.transaction_log[-100:]

        logger.debug(f"取引ログ記録: {log_entry}")

    def get_payment_stats(self) -> Dict[str, Any]:
        """決済統計を取得"""
        if not self.transaction_log:
            return {"total_transactions": 0, "success_rate": 0.0}

        payment_logs = [log for log in self.transaction_log if log["type"] == "payment"]

        if not payment_logs:
            return {"total_transactions": 0, "success_rate": 0.0}

        successful = sum(1 for log in payment_logs if log["result"].success)
        total = len(payment_logs)

        return {
            "total_transactions": total,
            "successful_transactions": successful,
            "failed_transactions": total - successful,
            "success_rate": successful / total,
            "total_amount": sum(log["amount"] for log in payment_logs if log["result"].success),
            "total_fees": sum(log["result"].fee for log in payment_logs if log["result"].success)
        }

    def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の取引を取得"""
        return self.transaction_log[-limit:] if self.transaction_log else []

# グローバルインスタンス
payment_service = PaymentService()
