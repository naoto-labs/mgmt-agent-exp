"""
APIモジュールパッケージ

このパッケージには、REST APIエンドポイントが含まれます。
"""

from .vending import router as vending_router
from .tablet import router as tablet_router
from .procurement import router as procurement_router

__all__ = [
    "vending_router",
    "tablet_router",
    "procurement_router",
]
