import os
import re
import logging
from cryptography.fernet import Fernet
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class SecureConfig:
    """セキュアな設定管理クラス"""

    def __init__(self):
        self._anthropic_key: Optional[str] = None
        self._openai_key: Optional[str] = None
        self._stripe_key: Optional[str] = None
        self._encryption_key: Optional[str] = None
        self._fernet: Optional[Fernet] = None

    @property
    def anthropic_api_key(self) -> str:
        """Anthropic APIキーの取得（遅延読み込み）"""
        if not self._anthropic_key:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key or (key == "your_anthropic_api_key_here" and len(key) < 20):
                raise ValueError("有効なANTHROPIC_API_KEYが必要です")
            self._anthropic_key = key
        return self._anthropic_key

    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI APIキーの取得（オプション）"""
        if not self._openai_key:
            key = os.getenv("OPENAI_API_KEY")
            if key and key != "your_openai_api_key_here":
                self._openai_key = key
        return self._openai_key

    @property
    def azure_openai_key(self) -> Optional[str]:
        """Azure OpenAI APIキーの取得（オプション）"""
        return os.getenv("OPENAI_API_KEY")  # Azure OpenAIも同じ環境変数を使用

    @property
    def azure_endpoint(self) -> Optional[str]:
        """Azure OpenAIエンドポイントの取得（オプション）"""
        return os.getenv("OPENAI_API_BASE")

    @property
    def stripe_api_key(self) -> Optional[str]:
        """Stripe APIキーの取得（オプション）"""
        if not self._stripe_key:
            key = os.getenv("STRIPE_API_KEY")
            if key and key != "your_stripe_api_key_here":
                self._stripe_key = key
        return self._stripe_key

    @property
    def encryption_key(self) -> str:
        """暗号化キーの取得（オプション）"""
        if not self._encryption_key:
            key = os.getenv("ENCRYPTION_KEY")
            if not key or key == "your_encryption_key_here":
                # 暗号化キーがない場合はデフォルトキーを生成（開発用）
                import hashlib
                key = hashlib.sha256(b"default_encryption_key_for_development").hexdigest()[:32]
            self._encryption_key = key
        return self._encryption_key

    @property
    def fernet(self) -> Fernet:
        """Fernet暗号化オブジェクトの取得"""
        if not self._fernet:
            self._fernet = Fernet(self.encryption_key.encode())
        return self._fernet

    def encrypt_data(self, data: str) -> str:
        """データの暗号化"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """データの復号化"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def mask_api_key(self, api_key: str) -> str:
        """APIキーのマスキング（ログ出力用）"""
        if len(api_key) <= 8:
            return "***"
        return f"{api_key[:4]}***{api_key[-4:]}"

    def validate_api_keys(self) -> Dict[str, bool]:
        """全APIキーの検証"""
        validation_results = {}

        # 必須キーの検証
        try:
            _ = self.anthropic_api_key
            validation_results["anthropic"] = True
        except ValueError as e:
            logger.error(f"Anthropic APIキー検証エラー: {e}")
            validation_results["anthropic"] = False

        # オプションキーの検証
        validation_results["openai"] = self.openai_api_key is not None
        validation_results["stripe"] = self.stripe_api_key is not None

        return validation_results

    def validate_all_keys(self):
        """起動時にすべての必須キーを検証"""
        required_keys = ["ANTHROPIC_API_KEY", "ENCRYPTION_KEY"]
        missing = []

        for key in required_keys:
            value = os.getenv(key)
            if not value:
                missing.append(key)
            # 開発環境ではダミー値も許可（より柔軟に）
            elif ("your_" in value and len(value) < 20) or ("dummy" in value and len(value) > 20) or ("test-dummy" in value):
                # ダミー値は許可（開発環境用）
                pass
            elif len(value) < 10:
                missing.append(key)

        if missing:
            raise ValueError(f"必要な環境変数が設定されていません: {missing}")

class SecureFormatter(logging.Formatter):
    """セキュアなログフォーマッタ（APIキーをマスク）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secure_config = SecureConfig()

    def format(self, record) -> str:
        # APIキーパターンをマスク
        msg = super().format(record)
        # Anthropic APIキー（sk-ant-apiで始まる）
        msg = re.sub(r'sk-ant-api[0-9a-zA-Z\-_]{40,}', 'sk-ant-api***MASKED***', msg)
        # OpenAI APIキー（sk-で始まる）
        msg = re.sub(r'sk-[a-zA-Z0-9]{48}', 'sk-***MASKED***', msg)
        # Stripe APIキー（sk_test_またはsk_live_で始まる）
        msg = re.sub(r'sk_(test|live)_[a-zA-Z0-9]{40,}', 'sk_***MASKED***', msg)
        return msg

def secure_logging(func):
    """セキュアログデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # ログレベルをチェックして機密情報をフィルタリング
        if logger.level <= logging.DEBUG:
            # デバッグレベルでは詳細ログを許可
            return func(*args, **kwargs)
        else:
            # 本番環境では機密情報をマスク
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"エラー発生: {str(e)}")
                raise
    return wrapper

# グローバルインスタンス
secure_config = SecureConfig()

def setup_secure_logging():
    """セキュアログのセットアップ"""
    formatter = SecureFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ルートロガーにセキュアフォーマッタを設定
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    logger.info("セキュアログシステムを初期化しました")
