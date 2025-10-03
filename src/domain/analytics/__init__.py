"""
アナリティクモジュールパッケージ

このパッケージには、分析システムが含まれます。
"""

from .event_tracker import (
    EventTracker,
    SystemEvent,
    EventType,
    EventSeverity,
    event_tracker,
    track_sale_event,
    track_payment_event,
    track_inventory_event,
    track_customer_interaction,
    track_ai_event,
    track_system_event
)

__all__ = [
    "EventTracker",
    "SystemEvent",
    "EventType",
    "EventSeverity",
    "event_tracker",
    "track_sale_event",
    "track_payment_event",
    "track_inventory_event",
    "track_customer_interaction",
    "track_ai_event",
    "track_system_event",
]
