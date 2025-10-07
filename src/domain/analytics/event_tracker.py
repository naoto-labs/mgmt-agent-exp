import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.shared.config.settings import settings

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """イベントタイプ"""

    # 販売関連
    SALE = "sale"
    PAYMENT = "payment"
    REFUND = "refund"

    # 在庫関連
    INVENTORY_CHECK = "inventory_check"
    RESTOCK = "restock"
    DISPENSE = "dispense"
    INVENTORY_ADJUSTMENT = "inventory_adjustment"

    # 顧客関連
    CUSTOMER_INTERACTION = "customer_interaction"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"

    # AI関連
    AI_DECISION = "ai_decision"
    AI_ERROR = "ai_error"
    MODEL_SWITCH = "model_switch"

    # システム関連
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR = "error"
    WARNING = "warning"


class EventSeverity(str, Enum):
    """イベント深刻度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemEvent:
    """システムイベント"""

    event_id: str
    event_type: EventType
    severity: EventSeverity
    timestamp: datetime
    source: str  # イベント発生元（例: "payment_service", "inventory_service"）
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    machine_id: str = field(default="VM001")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "message": self.message,
            "data": self.data,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "machine_id": self.machine_id,
        }


class EventTracker:
    """イベント追跡クラス"""

    def __init__(self):
        self.events: List[SystemEvent] = []
        self.max_events = 10000  # 最大保持イベント数
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.storage_dir = "data/events"
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """ストレージディレクトリを確保"""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _generate_event_id(self) -> str:
        """イベントIDを生成"""
        return f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000}"

    def track_event(
        self,
        event_type: EventType,
        source: str,
        message: str,
        severity: EventSeverity = EventSeverity.MEDIUM,
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SystemEvent:
        """イベントを記録"""
        event = SystemEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            source=source,
            message=message,
            data=data or {},
            user_id=user_id,
            session_id=session_id,
        )

        # イベントをリストに追加
        self.events.append(event)

        # イベント数の制限
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

        # イベントハンドラーを実行
        self._trigger_event_handlers(event)

        # 永続化ストレージに保存
        self._save_event_to_storage(event)

        logger.info(f"イベント記録: {event.event_type.value} - {event.message}")
        return event

    def _trigger_event_handlers(self, event: SystemEvent):
        """イベントハンドラーを実行"""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"イベントハンドラー実行エラー: {e}")

    def _save_event_to_storage(self, event: SystemEvent):
        """イベントをストレージに保存"""
        try:
            # 日別ファイルに保存
            date_str = event.timestamp.strftime("%Y%m%d")
            file_path = f"{self.storage_dir}/events_{date_str}.jsonl"

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

        except Exception as e:
            logger.error(f"イベント保存エラー: {e}")

    def add_event_handler(self, event_type: EventType, handler: Callable):
        """イベントハンドラーを追加"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"イベントハンドラーを追加: {event_type.value}")

    def remove_event_handler(self, event_type: EventType, handler: Callable):
        """イベントハンドラーを削除"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"イベントハンドラーを削除: {event_type.value}")
            except ValueError:
                pass

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SystemEvent]:
        """イベントを取得"""
        filtered_events = self.events

        # タイプでフィルタリング
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        # 時間範囲でフィルタリング
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        # 最新順にソートして制限
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]

    def get_recent_events(self, hours: int = 24) -> List[SystemEvent]:
        """最近のイベントを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return self.get_events(start_time=cutoff_time)

    def get_event_stats(self) -> Dict[str, Any]:
        """イベント統計を取得"""
        if not self.events:
            return {"total_events": 0, "events_by_type": {}, "events_by_severity": {}}

        # タイプ別統計
        events_by_type = {}
        for event in self.events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1

        # 深刻度別統計
        events_by_severity = {}
        for event in self.events:
            severity = event.severity.value
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1

        # 時間別統計（直近24時間）
        recent_events = self.get_recent_events(24)
        recent_by_type = {}
        for event in recent_events:
            event_type = event.event_type.value
            recent_by_type[event_type] = recent_by_type.get(event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "recent_events_24h": len(recent_events),
            "recent_events_by_type": recent_by_type,
            "oldest_event": self.events[0].timestamp.isoformat()
            if self.events
            else None,
            "newest_event": self.events[-1].timestamp.isoformat()
            if self.events
            else None,
        }

    def get_daily_sales_data(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """日別売上データを取得"""
        daily_data = {}

        # 売上イベントを抽出
        sales_events = [
            e
            for e in self.events
            if e.event_type == EventType.SALE and start_date <= e.timestamp <= end_date
        ]

        for event in sales_events:
            date_key = event.timestamp.date().isoformat()
            if date_key not in daily_data:
                daily_data[date_key] = {
                    "date": date_key,
                    "revenue": 0,
                    "transaction_count": 0,
                }

            # データから金額を取得（簡易版）
            revenue = event.data.get("amount", 0)
            daily_data[date_key]["revenue"] += revenue
            daily_data[date_key]["transaction_count"] += 1

        return list(daily_data.values())

    def search_events(
        self, query: str, event_type: Optional[EventType] = None
    ) -> List[SystemEvent]:
        """イベントを検索"""
        results = []

        for event in self.events:
            if event_type and event.event_type != event_type:
                continue

            # メッセージとデータで検索
            search_text = (event.message + str(event.data)).lower()
            if query.lower() in search_text:
                results.append(event)

        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_error_events(self, hours: int = 24) -> List[SystemEvent]:
        """エラーイベントを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        error_events = [
            e
            for e in self.events
            if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
            and e.timestamp >= cutoff_time
        ]
        return sorted(error_events, key=lambda x: x.timestamp, reverse=True)

    def get_system_health_score(self) -> float:
        """システム健全性スコアを計算（0-1）"""
        if not self.events:
            return 1.0

        recent_events = self.get_recent_events(24)

        if not recent_events:
            return 1.0

        # エラーイベントの割合を計算
        error_events = [
            e
            for e in recent_events
            if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]
        ]

        error_rate = len(error_events) / len(recent_events)

        # スコア = 1 - エラー率（エラー率が高いほどスコアが低い）
        health_score = max(0.0, 1.0 - error_rate)

        return health_score

    def export_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json",
    ) -> str:
        """イベントをエクスポート"""
        events = self.get_events(None, start_date, end_date, limit=10000)

        if format.lower() == "json":
            events_data = [event.to_dict() for event in events]

            return json.dumps(
                {
                    "events": events_data,
                    "total_events": len(events_data),
                    "export_period": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None,
                    },
                    "exported_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )

        else:
            raise ValueError(f"未対応のエクスポート形式: {format}")

    def clear_old_events(self, days_to_keep: int = 30):
        """古いイベントを削除"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        original_count = len(self.events)

        self.events = [e for e in self.events if e.timestamp >= cutoff_date]

        removed_count = original_count - len(self.events)
        logger.info(f"古いイベントを削除: {removed_count}件削除")

        return removed_count


# グローバルインスタンス
event_tracker = EventTracker()


# 便利なイベント記録関数
def track_sale_event(
    transaction_id: str,
    amount: float,
    product_name: str = "",
    user_id: Optional[str] = None,
):
    """売上イベントを記録"""
    message = f"商品販売: {product_name} - 金額: ¥{amount:,}"
    data = {
        "transaction_id": transaction_id,
        "amount": amount,
        "product_name": product_name,
    }

    return event_tracker.track_event(
        EventType.SALE, "vending_service", message, EventSeverity.LOW, data, user_id
    )


def track_payment_event(
    payment_id: str,
    amount: float,
    method: str,
    success: bool,
    user_id: Optional[str] = None,
):
    """決済イベントを記録"""
    severity = EventSeverity.LOW if success else EventSeverity.MEDIUM
    message = (
        f"決済処理: {method} - 金額: ¥{amount:,} - {'成功' if success else '失敗'}"
    )
    data = {
        "payment_id": payment_id,
        "amount": amount,
        "method": method,
        "success": success,
    }

    return event_tracker.track_event(
        EventType.PAYMENT, "payment_service", message, severity, data, user_id
    )


def track_inventory_event(
    event_type: EventType,
    product_name: str,
    quantity: int,
    slot_id: str = "",
    user_id: Optional[str] = None,
):
    """在庫イベントを記録"""
    message = f"在庫操作: {product_name} - 数量: {quantity}"
    data = {"product_name": product_name, "quantity": quantity, "slot_id": slot_id}

    return event_tracker.track_event(
        event_type, "inventory_service", message, EventSeverity.LOW, data, user_id
    )


def track_customer_interaction(
    session_id: str,
    customer_id: str,
    interaction_type: str,
    message: str = "",
    user_id: Optional[str] = None,
):
    """顧客対話を記録"""
    full_message = f"顧客対話: {interaction_type} - {message[:50]}..."
    data = {
        "session_id": session_id,
        "interaction_type": interaction_type,
        "message": message,
    }

    return event_tracker.track_event(
        EventType.CUSTOMER_INTERACTION,
        "customer_agent",
        full_message,
        EventSeverity.LOW,
        data,
        customer_id,
        session_id,
    )


def track_ai_event(
    event_type: EventType,
    model_name: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
):
    """AI関連イベントを記録"""
    full_message = f"AIイベント: {model_name} - {message}"
    event_data = data or {}
    event_data["model_name"] = model_name

    return event_tracker.track_event(
        event_type, "ai_model_manager", full_message, EventSeverity.MEDIUM, event_data
    )


def track_system_event(
    event_type: EventType,
    message: str,
    severity: EventSeverity = EventSeverity.MEDIUM,
    data: Optional[Dict[str, Any]] = None,
):
    """システムイベントを記録"""
    return event_tracker.track_event(event_type, "system", message, severity, data)
