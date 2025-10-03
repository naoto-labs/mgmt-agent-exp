"""
Infrastructure Layer - Technical Implementations
技術実装と外部システム統合
"""

# AI Infrastructure
# API Infrastructure
from . import ai, api

# AI Model Management
from .ai.model_manager import model_manager
from .api.procurement import router as procurement_router
from .api.tablet import router as tablet_router

# API Routers
from .api.vending import router as vending_router

__all__ = [
    # AI Infrastructure
    "model_manager",
    # API Infrastructure
    "vending_router",
    "tablet_router",
    "procurement_router",
]
