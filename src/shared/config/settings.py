import os
from typing import List, Optional

from pydantic import validator
from pydantic_settings import BaseSettings

from src.shared.config.security import secure_config


class VendingMachineSettings(BaseSettings):
    """自動販売機システム設定"""

    # 基本設定
    app_name: str = "AI Vending Machine Simulator"
    machine_id: str = "VM001"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # データベース設定
    database_url: str = "sqlite:///./vending.db"
    mongodb_url: str = "mongodb://localhost:27017/ai_vending_conversations"

    # AI設定
    ai_safety_threshold: float = 0.95
    max_decision_time: float = 5.0
    enable_guardrails: bool = True
    enable_decision_monitoring: bool = True

    # 許可されたアクション
    allowed_actions: List[str] = [
        "select_product",
        "process_payment",
        "dispense_product",
        "check_inventory",
        "generate_report",
        "customer_service",
    ]

    # 禁止パターン
    forbidden_patterns: List[str] = [
        "override_safety",
        "bypass_payment",
        "unlimited_dispense",
        "access_admin",
        "modify_prices",
        "delete_data",
    ]

    # 学習設定
    enable_learning: bool = False  # 本番では慎重に
    learning_rate: float = 0.001

    # データ収集設定
    enable_data_collection: bool = True
    data_validation_level: str = "strict"
    log_ai_decisions: bool = True
    use_nosql_for_conversations: bool = True

    # Agent目的設定 (VendingBench準拠)
    agent_objectives: dict = {
        "primary": [
            "VendingBench Primary Metricsの最適化（利益・在庫切れ率・価格精度・アクション正しさ・顧客満足度）",
            "5つのPrimary Metricsの目標達成：利益¥100,000、在庫切れ率10%以下、価格精度80%以上、アクション正しさ70%以上、顧客満足度3.5以上",
            "長期的一貫性（Secondary Metrics）の確保：75%以上の安定性維持",
        ],
        "optimization_period": {
            "short_term": "Primary Metricsの月次目標達成（今月利益¥100,000、在庫切れ率10%以下）",
            "medium_term": "Secondary Metricsの改善（長期的一貫性75%以上）",
            "long_term": "VendingBench評価基準での総合的な事業成長",
        },
        "constraints": [
            "VendingBench評価基準の厳格遵守",
            "Primary Metricsの目標値維持を最優先",
            "価格設定の正確性確保（精度80%以上）",
            "在庫切れ機会損失の最小化",
            "顧客満足度の継続的な向上",
        ],
        "priority_weight": {"short_term": 0.5, "medium_term": 0.3, "long_term": 0.2},
    }

    # Search Agent設定
    real_web_search: bool = True  # True: Tavily実検索, False: シミュレーション
    search_timeout: int = 30  # seconds
    search_max_retries: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # プロパティフィールドを無視

    @validator("ai_safety_threshold")
    def validate_safety_threshold(cls, v):
        """AI安全性閾値の検証"""
        if v < 0.9:
            raise ValueError("Safety threshold must be at least 0.9")
        if v > 1.0:
            raise ValueError("Safety threshold must be at most 1.0")
        return v

    @validator("data_validation_level")
    def validate_validation_level(cls, v):
        """データ検証レベルの検証"""
        valid_levels = ["strict", "moderate", "lenient"]
        if v not in valid_levels:
            raise ValueError(f"Validation level must be one of: {valid_levels}")
        return v

    def validate_payment_keys(self):
        """決済APIキーの検証"""
        if not any(
            [self.stripe_api_key, self.paypal_client_id, self.square_access_token]
        ):
            raise ValueError("少なくとも1つの決済APIキーが必要です")

    @property
    def stripe_api_key(self) -> Optional[str]:
        """Stripe APIキー"""
        return secure_config.stripe_api_key

    @property
    def paypal_client_id(self) -> Optional[str]:
        """PayPalクライアントID"""
        return os.getenv("PAYPAL_CLIENT_ID")

    @property
    def square_access_token(self) -> Optional[str]:
        """Squareアクセストークン"""
        return os.getenv("SQUARE_ACCESS_TOKEN")

    @property
    def jwt_secret_key(self) -> str:
        """JWTシークレットキー"""
        key = os.getenv("JWT_SECRET_KEY")
        if not key or key == "your_jwt_secret_key_here":
            raise ValueError("有効なJWT_SECRET_KEYが必要です")
        return key

    @property
    def inventory_api_key(self) -> Optional[str]:
        """在庫管理APIキー"""
        return os.getenv("INVENTORY_API_KEY")

    @property
    def maintenance_api_key(self) -> Optional[str]:
        """メンテナンスAPIキー"""
        return os.getenv("MAINTENANCE_API_KEY")

    @property
    def tavily_api_key(self) -> Optional[str]:
        """Tavily検索APIキー"""
        return secure_config.tavily_api_key


# グローバル設定インスタンス
settings = VendingMachineSettings()


# 起動時の設定検証
def validate_startup_settings():
    """起動時の設定検証"""
    try:
        # セキュリティ設定の検証（ダミー値での動作を許可）
        try:
            secure_config.validate_all_keys()
        except ValueError as e:
            print(
                f"警告: 設定検証で一部の問題がありますが、ダミー値で動作を継続します: {e}"
            )

        # 決済設定の検証（オプション）
        try:
            settings.validate_payment_keys()
        except ValueError:
            print("警告: 決済APIキーが設定されていません。決済機能が制限されます。")

        # AI安全性設定の検証
        if not settings.enable_guardrails:
            print("警告: AIガードレールが無効です。本番環境では推奨されません。")

        print("設定検証完了（ダミー値での動作を許可）")
        return True

    except Exception as e:
        print(f"設定検証エラー: {e}")
        return False
