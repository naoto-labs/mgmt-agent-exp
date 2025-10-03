import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class InventoryStatus(str, Enum):
    """在庫ステータス"""

    NORMAL = "normal"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    OVERSTOCK = "overstock"
    EXPIRED = "expired"
    DAMAGED = "damaged"


class InventoryLocation(str, Enum):
    """在庫場所"""

    VENDING_MACHINE = "vending_machine"
    STORAGE = "storage"
    WAREHOUSE = "warehouse"


class InventorySlot(BaseModel):
    """在庫スロットモデル"""

    # 基本情報
    slot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str = Field(..., min_length=1)
    location: InventoryLocation = InventoryLocation.VENDING_MACHINE

    # 商品情報
    product_id: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1)
    price: float = Field(default=0.0, gt=0)  # 商品価格（円）

    # 在庫情報
    current_quantity: int = Field(default=0, ge=0)
    max_quantity: int = Field(default=100, gt=0)
    min_quantity: int = Field(default=5, ge=0)

    # スロット情報
    slot_number: Optional[int] = None  # 自販機の物理スロット番号
    row_position: Optional[int] = None
    column_position: Optional[int] = None

    # ステータスとメタデータ
    status: InventoryStatus = InventoryStatus.NORMAL
    temperature_controlled: bool = False
    expiry_date: Optional[datetime] = None

    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_restocked: Optional[datetime] = None

    # 統計情報
    total_dispensed: int = Field(default=0, ge=0)
    total_restocked: int = Field(default=0, ge=0)
    last_dispensed: Optional[datetime] = None

    class Config:
        """Pydantic設定"""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @validator("status", always=True)
    def update_status_based_on_quantity(cls, v, values):
        """数量に基づくステータス更新"""
        if (
            "current_quantity" in values
            and "min_quantity" in values
            and "max_quantity" in values
        ):
            current = values["current_quantity"]
            min_qty = values["min_quantity"]
            max_qty = values["max_quantity"]

            if current == 0:
                return InventoryStatus.OUT_OF_STOCK
            elif current <= min_qty:
                return InventoryStatus.LOW_STOCK
            elif current > max_qty:
                return InventoryStatus.OVERSTOCK
            else:
                return InventoryStatus.NORMAL
        return v

    def is_available(self) -> bool:
        """利用可能かチェック"""
        return (
            self.status not in [InventoryStatus.OUT_OF_STOCK, InventoryStatus.DAMAGED]
            and self.current_quantity > 0
        )

    def needs_restock(self) -> bool:
        """補充が必要かチェック"""
        return self.current_quantity <= self.min_quantity

    def can_restock(self) -> bool:
        """補充可能かチェック"""
        return self.current_quantity < self.max_quantity

    def get_restock_quantity(self) -> int:
        """推奨補充数量を取得"""
        if not self.can_restock():
            return 0
        return min(
            self.max_quantity - self.current_quantity,
            self.max_quantity // 2,  # 最大在庫の50%を目安に補充
        )

    def dispense(self, quantity: int = 1) -> bool:
        """商品を排出"""
        if quantity <= 0 or quantity > self.current_quantity:
            return False

        self.current_quantity -= quantity
        self.total_dispensed += quantity
        self.last_dispensed = datetime.now()
        self.updated_at = datetime.now()

        return True

    def restock(self, quantity: int) -> bool:
        """在庫を補充"""
        if quantity <= 0:
            return False

        old_quantity = self.current_quantity
        self.current_quantity += quantity
        self.total_restocked += quantity
        self.last_restocked = datetime.now()
        self.updated_at = datetime.now()

        # 在庫変更ログ（後でログシステムに統合）
        print(
            f"在庫補充: {self.product_name} - {old_quantity} -> {self.current_quantity}"
        )
        return True

    def get_utilization_rate(self) -> float:
        """在庫利用率を取得"""
        if self.max_quantity == 0:
            return 0.0
        return (self.current_quantity / self.max_quantity) * 100

    def get_turnover_rate(self, days: int = 30) -> float:
        """回転率を取得（日次）"""
        if days <= 0:
            return 0.0
        return self.total_dispensed / days

    def is_expired(self) -> bool:
        """有効期限切れかチェック"""
        if not self.expiry_date:
            return False
        return datetime.now() > self.expiry_date

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "slot_id": self.slot_id,
            "machine_id": self.machine_id,
            "location": self.location,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "current_quantity": self.current_quantity,
            "max_quantity": self.max_quantity,
            "min_quantity": self.min_quantity,
            "slot_number": self.slot_number,
            "row_position": self.row_position,
            "column_position": self.column_position,
            "status": self.status,
            "temperature_controlled": self.temperature_controlled,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_restocked": self.last_restocked.isoformat()
            if self.last_restocked
            else None,
            "total_dispensed": self.total_dispensed,
            "total_restocked": self.total_restocked,
            "last_dispensed": self.last_dispensed.isoformat()
            if self.last_dispensed
            else None,
            "is_available": self.is_available(),
            "needs_restock": self.needs_restock(),
            "can_restock": self.can_restock(),
            "restock_quantity": self.get_restock_quantity(),
            "utilization_rate": self.get_utilization_rate(),
            "turnover_rate": self.get_turnover_rate(),
            "is_expired": self.is_expired(),
        }


class InventorySummary(BaseModel):
    """在庫サマリ"""

    machine_id: str
    total_slots: int
    active_slots: int
    out_of_stock_slots: int
    low_stock_slots: int
    total_value: float
    last_updated: datetime

    @classmethod
    def from_slots(
        cls, machine_id: str, slots: List[InventorySlot]
    ) -> "InventorySummary":
        """スロットリストからサマリを作成"""
        total_value = sum(
            slot.current_quantity * 100  # 仮の単価（後で商品情報から取得）
            for slot in slots
        )

        return cls(
            machine_id=machine_id,
            total_slots=len(slots),
            active_slots=len([s for s in slots if s.is_available()]),
            out_of_stock_slots=len(
                [s for s in slots if s.status == InventoryStatus.OUT_OF_STOCK]
            ),
            low_stock_slots=len(
                [s for s in slots if s.status == InventoryStatus.LOW_STOCK]
            ),
            total_value=total_value,
            last_updated=datetime.now(),
        )


class RestockPlan(BaseModel):
    """補充計画"""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total_cost: float = 0.0
    priority: str = "normal"  # normal, high, urgent

    def add_item(self, slot: InventorySlot, quantity: int):
        """補充項目を追加"""
        if quantity > 0:
            item = {
                "slot_id": slot.slot_id,
                "product_id": slot.product_id,
                "product_name": slot.product_name,
                "current_quantity": slot.current_quantity,
                "restock_quantity": quantity,
                "estimated_cost": quantity * 100,  # 仮の単価（後で商品情報から取得）
            }
            self.items.append(item)
            self.total_cost += item["estimated_cost"]

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "plan_id": self.plan_id,
            "machine_id": self.machine_id,
            "created_at": self.created_at.isoformat(),
            "items": self.items,
            "total_cost": self.total_cost,
            "priority": self.priority,
            "item_count": len(self.items),
        }


# サンプル在庫データ（開発・テスト用）
def create_sample_inventory_slots() -> List[InventorySlot]:
    """サンプル在庫スロットデータ作成"""
    from src.models.product import SAMPLE_PRODUCTS

    slots = []
    for i, product in enumerate(SAMPLE_PRODUCTS):
        slot = InventorySlot(
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id=product.product_id,
            product_name=product.name,
            price=product.price,
            current_quantity=product.stock_quantity,
            max_quantity=product.max_stock_quantity,
            min_quantity=product.min_stock_quantity,
            slot_number=i + 1,
            row_position=1,
            column_position=i + 1,
        )
        slots.append(slot)

    return slots
