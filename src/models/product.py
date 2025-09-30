from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class ProductCategory(str, Enum):
    """商品カテゴリ"""
    DRINK = "drink"
    SNACK = "snack"
    FOOD = "food"
    OTHER = "other"

class ProductStatus(str, Enum):
    """商品ステータス"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"

class ProductSize(str, Enum):
    """商品サイズ"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

class Product(BaseModel):
    """商品モデル"""

    # 基本情報
    product_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    category: ProductCategory
    price: float = Field(..., gt=0)
    cost: float = Field(..., gt=0)  # 原価

    # 在庫情報
    stock_quantity: int = Field(default=0, ge=0)
    max_stock_quantity: int = Field(default=100, gt=0)
    min_stock_quantity: int = Field(default=5, ge=0)

    # 商品属性
    size: ProductSize = ProductSize.MEDIUM
    weight_grams: Optional[float] = Field(default=None, gt=0)
    calories: Optional[int] = Field(default=None, ge=0)
    ingredients: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(default_factory=list)

    # ステータスとメタデータ
    status: ProductStatus = ProductStatus.ACTIVE
    sku: Optional[str] = None  # Stock Keeping Unit
    barcode: Optional[str] = None
    image_url: Optional[str] = None

    # タイムスタンプ
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # ビジネス情報
    profit_margin: Optional[float] = Field(default=None, ge=0, le=1)
    sales_count: int = Field(default=0, ge=0)
    return_rate: float = Field(default=0.0, ge=0, le=1)

    class Config:
        """Pydantic設定"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator("profit_margin", always=True)
    def calculate_profit_margin(cls, v, values):
        """利益率の自動計算"""
        if v is None and "price" in values and "cost" in values:
            price = values["price"]
            cost = values["cost"]
            if price > 0:
                return (price - cost) / price
        return v

    @validator("status", always=True)
    def update_status_based_on_stock(cls, v, values):
        """在庫に基づくステータス更新"""
        if "stock_quantity" in values:
            stock = values["stock_quantity"]
            if stock == 0:
                return ProductStatus.OUT_OF_STOCK
            elif stock > 0:
                return ProductStatus.ACTIVE
        return v

    def is_available(self) -> bool:
        """商品が利用可能かチェック"""
        return (
            self.status == ProductStatus.ACTIVE and
            self.stock_quantity > 0
        )

    def can_restock(self) -> bool:
        """補充可能かチェック"""
        return self.stock_quantity < self.max_stock_quantity

    def get_restock_quantity(self) -> int:
        """推奨補充数量を取得"""
        if not self.can_restock():
            return 0
        return min(
            self.max_stock_quantity - self.stock_quantity,
            self.max_stock_quantity // 2  # 最大在庫の50%を目安に補充
        )

    def update_sales_data(self, quantity_sold: int = 1):
        """販売データを更新"""
        self.sales_count += quantity_sold
        self.updated_at = datetime.now()

    def update_stock(self, quantity: int):
        """在庫数量を更新"""
        old_quantity = self.stock_quantity
        self.stock_quantity = max(0, self.stock_quantity + quantity)
        self.updated_at = datetime.now()

        # 在庫変更ログ（後でログシステムに統合）
        print(f"在庫更新: {self.name} - {old_quantity} -> {self.stock_quantity}")

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "price": self.price,
            "cost": self.cost,
            "stock_quantity": self.stock_quantity,
            "max_stock_quantity": self.max_stock_quantity,
            "min_stock_quantity": self.min_stock_quantity,
            "size": self.size,
            "weight_grams": self.weight_grams,
            "calories": self.calories,
            "ingredients": self.ingredients,
            "allergens": self.allergens,
            "status": self.status,
            "sku": self.sku,
            "barcode": self.barcode,
            "image_url": self.image_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "profit_margin": self.profit_margin,
            "sales_count": self.sales_count,
            "return_rate": self.return_rate,
            "is_available": self.is_available(),
            "can_restock": self.can_restock(),
            "restock_quantity": self.get_restock_quantity()
        }

class ProductCategoryInfo(BaseModel):
    """商品カテゴリ情報"""
    category: ProductCategory
    name: str
    description: str
    icon_url: Optional[str] = None
    display_order: int = 0

    class Config:
        use_enum_values = True

# 商品カテゴリの定義
PRODUCT_CATEGORIES = {
    ProductCategory.DRINK: ProductCategoryInfo(
        category=ProductCategory.DRINK,
        name="飲料",
        description="各種飲料水、ジュース、コーヒー等",
        display_order=1
    ),
    ProductCategory.SNACK: ProductCategoryInfo(
        category=ProductCategory.SNACK,
        name="スナック",
        description="ポテトチップス、チョコレート、キャンディー等",
        display_order=2
    ),
    ProductCategory.FOOD: ProductCategoryInfo(
        category=ProductCategory.FOOD,
        name="食品",
        description="カップ麺、サンドイッチ、弁当等",
        display_order=3
    ),
    ProductCategory.OTHER: ProductCategoryInfo(
        category=ProductCategory.OTHER,
        name="その他",
        description="その他の商品",
        display_order=4
    )
}

# サンプル商品データ（開発・テスト用）
SAMPLE_PRODUCTS = [
    Product(
        name="コカ・コーラ",
        description="爽快な炭酸飲料の定番",
        category=ProductCategory.DRINK,
        price=150.0,
        cost=80.0,
        stock_quantity=20,
        size=ProductSize.MEDIUM,
        weight_grams=500,
        calories=42,
        sku="CC-001",
        barcode="4902102000012"
    ),
    Product(
        name="ポテトチップス うすしお味",
        description="サクサク食感のポテトチップス",
        category=ProductCategory.SNACK,
        price=180.0,
        cost=90.0,
        stock_quantity=15,
        size=ProductSize.MEDIUM,
        weight_grams=75,
        calories=536,
        ingredients=["じゃがいも", "植物油", "食塩"],
        sku="PC-001",
        barcode="4901330500019"
    ),
    Product(
        name="カップヌードル",
        description="熱湯3分で本格的なラーメン",
        category=ProductCategory.FOOD,
        price=200.0,
        cost=120.0,
        stock_quantity=10,
        size=ProductSize.MEDIUM,
        weight_grams=78,
        calories=351,
        ingredients=["小麦粉", "パーム油", "食塩", "醤油"],
        allergens=["小麦", "大豆"],
        sku="CN-001",
        barcode="4902105000016"
    )
]
