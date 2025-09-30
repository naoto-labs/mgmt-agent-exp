import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from src.config.security import secure_config
from src.config.settings import settings

logger = logging.getLogger(__name__)

class AIModelType(str, Enum):
    """AIモデルタイプ"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"  # テスト用モックモデル

class AIModelStatus(str, Enum):
    """AIモデルステータス"""
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class AIModelConfig:
    """AIモデル設定"""
    model_type: AIModelType
    model_name: str
    api_key: str
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class AIMessage:
    """AIメッセージ"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class AIResponse:
    """AI応答"""
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    success: bool
    error_message: Optional[str] = None

class BaseAIModel(ABC):
    """AIモデルの基底クラス"""

    def __init__(self, config: AIModelConfig):
        self.config = config
        self.status = AIModelStatus.READY
        self.last_used: Optional[float] = None
        self.total_requests = 0
        self.failed_requests = 0

    @abstractmethod
    async def generate_response(self, messages: List[AIMessage], **kwargs) -> AIResponse:
        """応答を生成"""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """ヘルスチェック"""
        pass

    def get_usage_stats(self) -> Dict[str, Any]:
        """使用統計を取得"""
        return {
            "model_name": self.config.model_name,
            "status": self.status,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            "last_used": self.last_used
        }

class AnthropicModel(BaseAIModel):
    """Anthropic Claudeモデル"""

    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """クライアントを初期化"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
            logger.info(f"Anthropicモデル {self.config.model_name} を初期化しました")
        except ImportError:
            logger.warning("anthropicパッケージがインストールされていません")
            self.status = AIModelStatus.UNAVAILABLE
        except Exception as e:
            logger.error(f"Anthropicクライアント初期化エラー: {e}")
            self.status = AIModelStatus.ERROR

    async def generate_response(self, messages: List[AIMessage], **kwargs) -> AIResponse:
        """Claudeで応答を生成"""
        if self.status != AIModelStatus.READY:
            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=0.0,
                success=False,
                error_message=f"モデルが利用不可: {self.status}"
            )

        self.status = AIModelStatus.BUSY
        start_time = time.time()

        try:
            # メッセージをAnthropic形式に変換
            anthropic_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # リクエストパラメータの設定
            request_params = {
                "model": self.config.model_name,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }

            # 追加パラメータの設定
            if "system" in kwargs:
                request_params["system"] = kwargs["system"]

            # API呼び出し
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.messages.create, **request_params
            )

            response_time = time.time() - start_time

            # 応答の処理
            content = ""
            if response.content:
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])

            # トークン使用量の取得（利用可能な場合）
            tokens_used = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time
            logger.error(f"Anthropic APIエラー: {e}")

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        try:
            if not self.client:
                return False

            # シンプルなメッセージでテスト
            test_messages = [AIMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=10)

            return response.success
        except Exception as e:
            logger.error(f"Anthropicヘルスチェックエラー: {e}")
            return False

class OpenAIModel(BaseAIModel):
    """OpenAI GPTモデル"""

    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """クライアントを初期化"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key)
            logger.info(f"OpenAIモデル {self.config.model_name} を初期化しました")
        except ImportError:
            logger.warning("openaiパッケージがインストールされていません")
            self.status = AIModelStatus.UNAVAILABLE
        except Exception as e:
            logger.error(f"OpenAIクライアント初期化エラー: {e}")
            self.status = AIModelStatus.ERROR

    async def generate_response(self, messages: List[AIMessage], **kwargs) -> AIResponse:
        """GPTで応答を生成"""
        if self.status != AIModelStatus.READY:
            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=0.0,
                success=False,
                error_message=f"モデルが利用不可: {self.status}"
            )

        self.status = AIModelStatus.BUSY
        start_time = time.time()

        try:
            # メッセージをOpenAI形式に変換
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # API呼び出し
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.chat.completions.create, **{
                    "model": self.config.model_name,
                    "messages": openai_messages,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "temperature": kwargs.get("temperature", self.config.temperature)
                }
            )

            response_time = time.time() - start_time

            # 応答の処理
            content = response.choices[0].message.content if response.choices else ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time
            logger.error(f"OpenAI APIエラー: {e}")

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        try:
            if not self.client:
                return False

            # シンプルなメッセージでテスト
            test_messages = [AIMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=10)

            return response.success
        except Exception as e:
            logger.error(f"OpenAIヘルスチェックエラー: {e}")
            return False

class MockModel(BaseAIModel):
    """モックモデル（テスト用）"""

    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        self.responses = [
            "これはモック応答です。",
            "テスト用の応答になります。",
            "開発環境でのテストに使用してください。"
        ]

    async def generate_response(self, messages: List[AIMessage], **kwargs) -> AIResponse:
        """モック応答を生成"""
        self.status = AIModelStatus.BUSY
        start_time = time.time()

        try:
            # 遅延をシミュレート
            await asyncio.sleep(0.1)

            # ユーザーメッセージに基づいて応答を選択
            user_content = messages[-1].content if messages else ""
            response_index = hash(user_content) % len(self.responses)
            content = self.responses[response_index]

            response_time = time.time() - start_time

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=len(content.split()),
                response_time=response_time,
                success=True
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        return self.status == AIModelStatus.READY

class ModelManager:
    """AIモデル管理クラス"""

    def __init__(self):
        self.models: Dict[str, BaseAIModel] = {}
        self.primary_model: Optional[str] = None
        self.fallback_models: List[str] = []
        self._initialize_models()

    def _initialize_models(self):
        """モデルを初期化"""
        try:
            # Anthropicモデル（プライマリ）
            if secure_config.anthropic_api_key:
                anthropic_config = AIModelConfig(
                    model_type=AIModelType.ANTHROPIC,
                    model_name="claude-3-sonnet-20240229",
                    api_key=secure_config.anthropic_api_key,
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30.0
                )
                self.models["anthropic"] = AnthropicModel(anthropic_config)
                self.primary_model = "anthropic"
                logger.info("Anthropicモデルをプライマリとして設定しました")

            # OpenAIモデル（フォールバック）
            if secure_config.openai_api_key:
                openai_config = AIModelConfig(
                    model_type=AIModelType.OPENAI,
                    model_name="gpt-3.5-turbo",
                    api_key=secure_config.openai_api_key,
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30.0
                )
                self.models["openai"] = OpenAIModel(openai_config)
                if not self.primary_model:
                    self.primary_model = "openai"
                else:
                    self.fallback_models.append("openai")
                logger.info("OpenAIモデルをフォールバックとして設定しました")

            # モックモデル（開発用）
            mock_config = AIModelConfig(
                model_type=AIModelType.MOCK,
                model_name="mock-model",
                api_key="mock-key"
            )
            self.models["mock"] = MockModel(mock_config)
            if not self.primary_model:
                self.primary_model = "mock"
            else:
                self.fallback_models.append("mock")

            logger.info(f"モデルマネージャーを初期化しました。プライマリ: {self.primary_model}")

        except Exception as e:
            logger.error(f"モデルマネージャー初期化エラー: {e}")

    async def generate_response(self, messages: List[AIMessage], **kwargs) -> AIResponse:
        """応答を生成（フォールバック機能付き）"""
        # プライマリモデルを試行
        if self.primary_model and self.primary_model in self.models:
            response = await self.models[self.primary_model].generate_response(messages, **kwargs)
            if response.success:
                return response

        # フォールバックモデルを試行
        for model_name in self.fallback_models:
            if model_name in self.models:
                logger.info(f"フォールバックモデル {model_name} を使用します")
                response = await self.models[model_name].generate_response(messages, **kwargs)
                if response.success:
                    return response

        # 全モデルが失敗した場合
        return AIResponse(
            content="申し訳ありませんが、現在AIサービスが利用できません。",
            model_used="none",
            tokens_used=0,
            response_time=0.0,
            success=False,
            error_message="全AIモデルが利用不可です"
        )

    async def check_all_models_health(self) -> Dict[str, bool]:
        """全モデルのヘルスチェック"""
        health_results = {}
        for name, model in self.models.items():
            health_results[name] = await model.check_health()
        return health_results

    def get_model_stats(self) -> Dict[str, Any]:
        """全モデルの統計情報を取得"""
        return {
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "models": {
                name: model.get_usage_stats()
                for name, model in self.models.items()
            }
        }

    def set_primary_model(self, model_name: str):
        """プライマリモデルを設定"""
        if model_name in self.models:
            self.primary_model = model_name
            logger.info(f"プライマリモデルを {model_name} に変更しました")

    def add_fallback_model(self, model_name: str):
        """フォールバックモデルを追加"""
        if model_name in self.models and model_name not in self.fallback_models:
            self.fallback_models.append(model_name)
            logger.info(f"フォールバックモデルに {model_name} を追加しました")

# グローバルインスタンス
model_manager = ModelManager()
