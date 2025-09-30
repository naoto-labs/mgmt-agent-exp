from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid

class PaymentMethod(str, Enum):
    """決済方法"""
    CASH = "cash"
    CARD = "card"
    MOBILE = "mobile"
    COUPON = "coupon"
    FREE = "free"  # テスト用

class TransactionStatus(str, Enum):
    """取引ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDING = "refunding"
    REFUNDED = "refunded"

class TransactionType(str, Enum):
    """取引タイプ"""
    PURCHASE = "purchase"
    REFUND = "refund"
    CANCELLATION = "cancellation"

class PaymentDetails(BaseModel):
    """決済詳細"""
    method: PaymentMethod
    amount: float = Field(..., gt=0)
    currency: str = "JPY"
    card_last_four: Optional[str] = None
    card_brand: Optional[str] = None
    payment_id: Optional[str] = None  # 外部決済サービスID
    fee: float = 0.0  # 決済手数料

class TransactionItem(BaseModel):
    """取引商品項目"""
    product_id: str
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)
    total_price: float = Field(..., gt=0)
    discount_amount: float = 0.0

    @validator("total_price", always=True)
    def calculate_total(cls, v, values):
        """合計金額の自動計算"""
        if v == 0 and "quantity" in values and "unit_price" in values:
            return values["quantity"] * values["unit_price"]
        return v

class Transaction(BaseModel):
    """取引モデル"""

    # 基本情報
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str = Field(..., min_length=1)
    customer_id: Optional[str] = None
    session_id: Optional[str] = None

    # 取引情報
    type: TransactionType = TransactionType.PURCHASE
    status: TransactionStatus = TransactionStatus.PENDING
    items: List[TransactionItem] = Field(..., min_items=1)

    # 金額情報
    subtotal: float = Field(..., ge=0)
    tax_amount: float = 0.0
    discount_amount: float = 0.0
    total_amount: float = Field(..., ge=0)

    # 決済情報
    payment_details: Optional[PaymentDetails] = None

    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # メタデータ
    notes: Optional[str] = None
    error_message: Optional[str] = None
    refund_reason: Optional[str] = None

    class Config:
        """Pydantic設定"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator("total_amount", always=True)
    def calculate_total_amount(cls, v, values):
        """合計金額の自動計算"""
        if v == 0 and "subtotal" in values:
            subtotal = values["subtotal"]
            tax = values.get("tax_amount", 0)
            discount = values.get("discount_amount", 0)
            return subtotal + tax - discount
        return v

    @validator("status")
    def validate_status_transition(cls, v, values):
        """ステータス遷移の検証"""
        if "status" in values:
            current_status = values["status"]
            valid_transitions = {
                TransactionStatus.PENDING: [TransactionStatus.PROCESSING, TransactionStatus.CANCELLED],
                TransactionStatus.PROCESSING: [TransactionStatus.COMPLETED, TransactionStatus.FAILED],
                TransactionStatus.COMPLETED: [TransactionStatus.REFUNDING],
                TransactionStatus.REFUNDING: [TransactionStatus.REFUNDED],
                TransactionStatus.FAILED: [TransactionStatus.PENDING],  # 再試行可能
                TransactionStatus.CANCELLED: [],  # 最終状態
                TransactionStatus.REFUNDED: []   # 最終状態
            }

            if v not in valid_transitions.get(current_status, []):
                raise ValueError(f"無効なステータス遷移: {current_status} -> {v}")

        return v

    def can_process_payment(self) -> bool:
        """決済処理可能かチェック"""
        return (
            self.status == TransactionStatus.PENDING and
            self.payment_details is not None and
            self.total_amount > 0
        )

    def can_complete(self) -> bool:
        """取引完了可能かチェック"""
        return (
            self.status == TransactionStatus.PROCESSING and
            self.payment_details is not None
        )

    def can_refund(self) -> bool:
        """返金可能かチェック"""
        return (
            self.status == TransactionStatus.COMPLETED and
            self.type == TransactionType.PURCHASE
        )

    def mark_as_processing(self):
        """処理中にステータス変更"""
        if self.can_process_payment():
            self.status = TransactionStatus.PROCESSING
            self.updated_at = datetime.now()

    def mark_as_completed(self):
        """完了にステータス変更"""
        if self.can_complete():
            self.status = TransactionStatus.COMPLETED
            self.completed_at = datetime.now()
            self.updated_at = datetime.now()

    def mark_as_failed(self, error_message: str):
        """失敗にステータス変更"""
        self.status = TransactionStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now()

    def mark_as_refunded(self, reason: str):
        """返金済みにステータス変更"""
        if self.can_refund():
            self.status = TransactionStatus.REFUNDED
            self.refund_reason = reason
            self.updated_at = datetime.now()

    def add_payment_details(self, payment_details: PaymentDetails):
        """決済詳細を追加"""
        self.payment_details = payment_details
        self.updated_at = datetime.now()

    def calculate_subtotal(self) -> float:
        """小計を計算"""
        return sum(item.total_price for item in self.items)

    def calculate_tax(self, tax_rate: float = 0.1) -> float:
        """税額を計算"""
        return self.subtotal * tax_rate

    def apply_discount(self, discount_amount: float):
        """割引を適用"""
        if discount_amount > 0 and discount_amount <= self.subtotal:
            self.discount_amount = discount_amount
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "transaction_id": self.transaction_id,
            "machine_id": self.machine_id,
            "customer_id": self.customer_id,
            "session_id": self.session_id,
            "type": self.type,
            "status": self.status,
            "items": [item.dict() for item in self.items],
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "discount_amount": self.discount_amount,
            "total_amount": self.total_amount,
            "payment_details": self.payment_details.dict() if self.payment_details else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "error_message": self.error_message,
            "refund_reason": self.refund_reason,
            "can_process_payment": self.can_process_payment(),
            "can_complete": self.can_complete(),
            "can_refund": self.can_refund()
        }

class TransactionSummary(BaseModel):
    """取引サマリ"""
    transaction_id: str
    machine_id: str
    customer_id: Optional[str]
    total_amount: float
    status: TransactionStatus
    created_at: datetime
    item_count: int

    @classmethod
    def from_transaction(cls, transaction: Transaction) -> "TransactionSummary":
        """Transactionからサマリを作成"""
        return cls(
            transaction_id=transaction.transaction_id,
            machine_id=transaction.machine_id,
            customer_id=transaction.customer_id,
            total_amount=transaction.total_amount,
            status=transaction.status,
            created_at=transaction.created_at,
            item_count=len(transaction.items)
        )

# サンプル取引データ（開発・テスト用）
def create_sample_transaction() -> Transaction:
    """サンプル取引データ作成"""
    from src.models.product import SAMPLE_PRODUCTS

    items = [
        TransactionItem(
            product_id=SAMPLE_PRODUCTS[0].product_id,
            product_name=SAMPLE_PRODUCTS[0].name,
            quantity=2,
            unit_price=SAMPLE_PRODUCTS[0].price,
            total_price=SAMPLE_PRODUCTS[0].price * 2
        )
    ]

    payment_details = PaymentDetails(
        method=PaymentMethod.CARD,
        amount=300.0,
        card_last_four="1234",
        card_brand="visa",
        payment_id="pi_test_1234567890"
    )

    transaction = Transaction(
        machine_id="VM001",
        customer_id="customer_001",
        items=items,
        subtotal=300.0,
        tax_amount=30.0,
        total_amount=330.0,
        payment_details=payment_details
    )

    return transaction
