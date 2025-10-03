"""
データモデルパッケージ

このパッケージには、自動販売機システムの全データモデルが含まれます。
"""

# 商品関連モデル
from .product import (
    Product,
    ProductCategory,
    ProductStatus,
    ProductSize,
    ProductCategoryInfo,
    PRODUCT_CATEGORIES,
    SAMPLE_PRODUCTS
)

# 取引関連モデル
from .transaction import (
    Transaction,
    TransactionStatus,
    TransactionType,
    PaymentMethod,
    PaymentDetails,
    TransactionItem,
    TransactionSummary,
    create_sample_transaction
)

# 在庫関連モデル
from .inventory import (
    InventorySlot,
    InventoryStatus,
    InventoryLocation,
    InventorySummary,
    RestockPlan,
    create_sample_inventory_slots
)

__all__ = [
    # Product models
    "Product",
    "ProductCategory",
    "ProductStatus",
    "ProductSize",
    "ProductCategoryInfo",
    "PRODUCT_CATEGORIES",
    "SAMPLE_PRODUCTS",

    # Transaction models
    "Transaction",
    "TransactionStatus",
    "TransactionType",
    "PaymentMethod",
    "PaymentDetails",
    "TransactionItem",
    "TransactionSummary",
    "create_sample_transaction",

    # Inventory models
    "InventorySlot",
    "InventoryStatus",
    "InventoryLocation",
    "InventorySummary",
    "RestockPlan",
    "create_sample_inventory_slots",
]
