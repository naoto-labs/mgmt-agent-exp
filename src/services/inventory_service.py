import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.models.inventory import (
    InventorySlot,
    InventoryStatus,
    InventoryLocation,
    InventorySummary,
    RestockPlan
)
from src.models.product import Product
from src.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class InventoryAlert:
    """在庫アラート"""
    slot_id: str
    product_id: str
    product_name: str
    alert_type: str  # "low_stock", "out_of_stock", "expired", "overstock"
    severity: str    # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime

class InventoryService:
    """在庫管理サービス"""

    def __init__(self):
        self.vending_machine_slots: Dict[str, InventorySlot] = {}
        self.storage_slots: Dict[str, InventorySlot] = {}
        self.alerts: List[InventoryAlert] = []
        self.alert_history: List[InventoryAlert] = []

    def add_slot(self, slot: InventorySlot):
        """在庫スロットを追加"""
        if slot.location == InventoryLocation.VENDING_MACHINE:
            self.vending_machine_slots[slot.slot_id] = slot
        elif slot.location == InventoryLocation.STORAGE:
            self.storage_slots[slot.slot_id] = slot
        else:
            logger.warning(f"未知の場所を持つスロット: {slot.location}")

        logger.info(f"在庫スロットを追加: {slot.product_name} ({slot.location})")

    def remove_slot(self, slot_id: str) -> bool:
        """在庫スロットを削除"""
        if slot_id in self.vending_machine_slots:
            del self.vending_machine_slots[slot_id]
            logger.info(f"自販機スロットを削除: {slot_id}")
            return True
        elif slot_id in self.storage_slots:
            del self.storage_slots[slot_id]
            logger.info(f"保管庫スロットを削除: {slot_id}")
            return True
        return False

    def get_slot(self, slot_id: str) -> Optional[InventorySlot]:
        """スロットを取得"""
        return (
            self.vending_machine_slots.get(slot_id) or
            self.storage_slots.get(slot_id)
        )

    def get_product_slots(self, product_id: str) -> List[InventorySlot]:
        """商品の全スロットを取得"""
        all_slots = list(self.vending_machine_slots.values()) + list(self.storage_slots.values())
        return [slot for slot in all_slots if slot.product_id == product_id]

    def get_total_inventory(self, product_id: str) -> Dict[str, Any]:
        """商品の総在庫を取得"""
        slots = self.get_product_slots(product_id)

        vending_stock = sum(slot.current_quantity for slot in slots if slot.location == InventoryLocation.VENDING_MACHINE)
        storage_stock = sum(slot.current_quantity for slot in slots if slot.location == InventoryLocation.STORAGE)

        return {
            "product_id": product_id,
            "vending_machine_stock": vending_stock,
            "storage_stock": storage_stock,
            "total_stock": vending_stock + storage_stock,
            "slot_count": len(slots)
        }

    def is_product_available(self, product_id: str) -> bool:
        """商品が利用可能かチェック"""
        slots = self.get_product_slots(product_id)
        return any(slot.is_available() for slot in slots)

    def dispense_product(self, product_id: str, quantity: int = 1) -> Tuple[bool, str]:
        """商品を排出"""
        # 自販機内のスロットから優先的に排出
        vending_slots = [slot for slot in self.get_product_slots(product_id)
                        if slot.location == InventoryLocation.VENDING_MACHINE and slot.is_available()]

        if not vending_slots:
            return False, "商品が利用できません"

        # 在庫の多いスロットから優先的に使用
        target_slot = max(vending_slots, key=lambda s: s.current_quantity)

        if target_slot.dispense(quantity):
            logger.info(f"商品排出成功: {target_slot.product_name} x{quantity}")

            # Record COGS in journal
            try:
                product = get_product_by_id(product_id)
                cost_per_unit = product.price * 0.7 if product else 100  # Assume cost is 70% of selling price
                total_cost = quantity * cost_per_unit
                from src.accounting.journal_entry import journal_processor
                from datetime import date
                journal_processor.add_entry("5001", date.today(), total_cost, "debit", f"COGS - {product_id} x{quantity}")
            except Exception as e:
                logger.error(f"Journal entry for COGS error: {e}")

            self._check_alerts(target_slot)
            return True, f"{target_slot.product_name}を{quantity}個排出しました"
        else:
            return False, "商品の排出に失敗しました"

    def restock_slot(self, slot_id: str, quantity: int) -> Tuple[bool, str]:
        """スロットを補充"""
        slot = self.get_slot(slot_id)
        if not slot:
            return False, "スロットが見つかりません"

        if not slot.can_restock():
            return False, "スロットは満杯です"

        if slot.restock(quantity):
            logger.info(f"在庫補充成功: {slot.product_name} +{quantity}")
            self._check_alerts(slot)
            return True, f"{slot.product_name}を{quantity}個補充しました"
        else:
            return False, "在庫の補充に失敗しました"

    def transfer_to_vending_machine(self, product_id: str, quantity: int) -> Tuple[bool, str]:
        """保管庫から自販機へ商品を移動"""
        # 保管庫のスロットを確認
        storage_slots = [slot for slot in self.get_product_slots(product_id)
                        if slot.location == InventoryLocation.STORAGE and slot.current_quantity >= quantity]

        if not storage_slots:
            return False, "保管庫に十分な在庫がありません"

        # 自販機の空きスロットを確認
        vending_slots = [slot for slot in self.get_product_slots(product_id)
                        if slot.location == InventoryLocation.VENDING_MACHINE and slot.can_restock()]

        if not vending_slots:
            return False, "自販機に空きスロットがありません"

        # 保管庫から減算
        source_slot = storage_slots[0]
        if not source_slot.dispense(quantity):
            return False, "保管庫からの商品移動に失敗しました"

        # 自販機に追加
        target_slot = vending_slots[0]
        if not target_slot.restock(quantity):
            # 失敗した場合は保管庫に戻す（ロールバック）
            source_slot.restock(quantity)
            return False, "自販機への商品移動に失敗しました"

        logger.info(f"在庫移動成功: {source_slot.product_name} x{quantity} (保管庫→自販機)")
        return True, f"{source_slot.product_name}を{quantity}個自販機へ移動しました"

    def get_inventory_summary(self) -> InventorySummary:
        """在庫サマリを取得"""
        all_slots = list(self.vending_machine_slots.values()) + list(self.storage_slots.values())
        return InventorySummary.from_slots("VM001", all_slots)

    def get_low_stock_slots(self, threshold: Optional[int] = None) -> List[InventorySlot]:
        """低在庫スロットを取得"""
        if threshold is None:
            threshold = 5  # デフォルトの閾値

        low_stock_slots = []

        for slot in self.vending_machine_slots.values():
            if slot.current_quantity <= threshold and slot.current_quantity > 0:
                low_stock_slots.append(slot)

        return low_stock_slots

    def get_out_of_stock_slots(self) -> List[InventorySlot]:
        """在庫切れスロットを取得"""
        out_of_stock_slots = []

        for slot in self.vending_machine_slots.values():
            if slot.status == InventoryStatus.OUT_OF_STOCK:
                out_of_stock_slots.append(slot)

        return out_of_stock_slots

    def generate_restock_plan(self) -> RestockPlan:
        """補充計画を生成"""
        plan = RestockPlan(machine_id="VM001")

        # 在庫切れ商品を優先的に補充
        out_of_stock = self.get_out_of_stock_slots()
        for slot in out_of_stock:
            quantity = slot.get_restock_quantity()
            if quantity > 0:
                plan.add_item(slot, quantity)
                plan.priority = "urgent"

        # 低在庫商品を補充
        low_stock = self.get_low_stock_slots()
        for slot in low_stock:
            quantity = slot.get_restock_quantity()
            if quantity > 0:
                plan.add_item(slot, quantity)
                if plan.priority == "normal":
                    plan.priority = "high"

        logger.info(f"補充計画を生成: {len(plan.items)}項目, 優先度={plan.priority}")
        return plan

    def _check_alerts(self, slot: InventorySlot):
        """スロットの状態に基づいてアラートをチェック"""
        alerts_to_remove = []
        new_alerts = []

        # 既存アラートの確認
        for alert in self.alerts:
            if alert.slot_id == slot.slot_id:
                # アラートが解決されたかチェック
                if self._should_remove_alert(alert, slot):
                    alerts_to_remove.append(alert)
                else:
                    # アラートの深刻度を更新
                    alert.severity = self._calculate_alert_severity(slot)

        # 新しいアラートの生成
        new_alert = self._generate_alert(slot)
        if new_alert:
            new_alerts.append(new_alert)

        # アラートの更新
        for alert in alerts_to_remove:
            self.alerts.remove(alert)
            self.alert_history.append(alert)

        for alert in new_alerts:
            self.alerts.append(alert)

        # アラート履歴の制限（最新100件）
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

    def _should_remove_alert(self, alert: InventoryAlert, slot: InventorySlot) -> bool:
        """アラートを削除すべきかチェック"""
        if alert.alert_type == "out_of_stock" and slot.status != InventoryStatus.OUT_OF_STOCK:
            return True
        elif alert.alert_type == "low_stock" and slot.current_quantity > slot.min_quantity:
            return True
        elif alert.alert_type == "expired" and not slot.is_expired():
            return True
        return False

    def _calculate_alert_severity(self, slot: InventorySlot) -> str:
        """アラートの深刻度を計算"""
        if slot.status == InventoryStatus.OUT_OF_STOCK:
            return "critical"
        elif slot.current_quantity == 0:
            return "high"
        elif slot.current_quantity <= slot.min_quantity // 2:
            return "medium"
        else:
            return "low"

    def _generate_alert(self, slot: InventorySlot) -> Optional[InventoryAlert]:
        """新しいアラートを生成"""
        if slot.status == InventoryStatus.OUT_OF_STOCK:
            return InventoryAlert(
                slot_id=slot.slot_id,
                product_id=slot.product_id,
                product_name=slot.product_name,
                alert_type="out_of_stock",
                severity="critical",
                message=f"{slot.product_name}の在庫が切れました",
                timestamp=datetime.now()
            )
        elif slot.needs_restock():
            return InventoryAlert(
                slot_id=slot.slot_id,
                product_id=slot.product_id,
                product_name=slot.product_name,
                alert_type="low_stock",
                severity="medium",
                message=f"{slot.product_name}の在庫が少なくなっています（残り{slot.current_quantity}個）",
                timestamp=datetime.now()
            )
        elif slot.is_expired():
            return InventoryAlert(
                slot_id=slot.slot_id,
                product_id=slot.product_id,
                product_name=slot.product_name,
                alert_type="expired",
                severity="high",
                message=f"{slot.product_name}の有効期限が切れています",
                timestamp=datetime.now()
            )

        return None

    def get_alerts(self) -> List[InventoryAlert]:
        """現在のアラートを取得"""
        return self.alerts.copy()

    def get_alert_summary(self) -> Dict[str, Any]:
        """アラートサマリを取得"""
        if not self.alerts:
            return {"total_alerts": 0, "by_severity": {}, "by_type": {}}

        by_severity = {}
        by_type = {}

        for alert in self.alerts:
            # 深刻度別集計
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            # タイプ別集計
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "by_severity": by_severity,
            "by_type": by_type,
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0)
        }

    def get_inventory_report(self) -> Dict[str, Any]:
        """在庫レポートを取得"""
        summary = self.get_inventory_summary()
        alerts = self.get_alert_summary()

        return {
            "summary": summary.dict(),
            "alerts": alerts,
            "vending_machine_slots": len(self.vending_machine_slots),
            "storage_slots": len(self.storage_slots),
            "total_products": len(set(
                slot.product_id for slot in
                list(self.vending_machine_slots.values()) + list(self.storage_slots.values())
            )),
            "generated_at": datetime.now().isoformat()
        }

    def optimize_inventory_levels(self) -> Dict[str, Any]:
        """在庫レベルを最適化"""
        recommendations = []

        for slot in self.vending_machine_slots.values():
            if slot.location == InventoryLocation.VENDING_MACHINE:
                # 自販機の最適在庫レベルを計算
                optimal_level = self._calculate_optimal_stock_level(slot)

                if slot.current_quantity < optimal_level * 0.5:
                    recommendations.append({
                        "slot_id": slot.slot_id,
                        "product_name": slot.product_name,
                        "current_quantity": slot.current_quantity,
                        "recommended_quantity": optimal_level,
                        "action": "restock",
                        "priority": "high" if slot.current_quantity == 0 else "medium"
                    })

        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }

    def _calculate_optimal_stock_level(self, slot: InventorySlot) -> int:
        """最適在庫レベルを計算"""
        # 簡易的な計算：最大在庫の70-80%を最適レベルとする
        return int(slot.max_quantity * 0.75)

    def get_inventory_by_location(self) -> Dict[str, List[InventorySlot]]:
        """場所別在庫を取得"""
        return {
            "vending_machine": list(self.vending_machine_slots.values()),
            "storage": list(self.storage_slots.values())
        }

    def get_expiring_products(self, days_ahead: int = 7) -> List[InventorySlot]:
        """有効期限切れ間近の商品を取得"""
        expiring_slots = []
        future_date = datetime.now() + timedelta(days=days_ahead)

        for slot in self.vending_machine_slots.values():
            if (slot.expiry_date and
                slot.expiry_date <= future_date and
                slot.current_quantity > 0):
                expiring_slots.append(slot)

        return expiring_slots

# グローバルインスタンス
inventory_service = InventoryService()

def get_product_by_id(product_id: str) -> Optional[Product]:
    """商品IDで商品を取得"""
    from src.models.product import SAMPLE_PRODUCTS
    for p in SAMPLE_PRODUCTS:
        if p.product_id == product_id:
            return p
    return None
