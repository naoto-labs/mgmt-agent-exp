"""
Infrastructure Layer - Technical Implementations
技術実装と外部システム統合
"""

# AI Infrastructure
# API Infrastructure
from . import ai, api


# AI Model Management (循環インポート防止のため遅延インポート)
def get_model_manager():
    """model_managerを遅延インポートして取得"""
    from .ai.model_manager import model_manager

    return model_manager


from .api.procurement import router as procurement_router
from .api.tablet import router as tablet_router

# API Routers
from .api.vending import router as vending_router

__all__ = [
    # API Infrastructure
    "vending_router",
    "tablet_router",
    "procurement_router",
]
