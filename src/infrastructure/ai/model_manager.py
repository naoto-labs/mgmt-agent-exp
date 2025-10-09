import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain.schema import AIMessage as LC_AIMessage
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.shared.config.security import secure_config
from src.shared.config.settings import settings

logger = logging.getLogger(__name__)


def convert_messages(
    msgs: List["AIMessage"],
) -> List[Union[HumanMessage, LC_AIMessage, SystemMessage]]:
    """Convert AIMessage list to LangChain message formats."""
    lc_msgs = []
    for msg in msgs:
        if msg.role == "user":
            lc_msgs.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_msgs.append(LC_AIMessage(content=msg.content))
        elif msg.role == "system":
            lc_msgs.append(SystemMessage(content=msg.content))
        else:
            lc_msgs.append(HumanMessage(content=msg.content))  # fallback
    return lc_msgs


class AIModelType(str, Enum):
    """AIモデルタイプ"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
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

    def __init__(
        self, config: AIModelConfig, tracer: Optional[BaseCallbackHandler] = None
    ):
        self.config = config
        self.tracer = tracer
        self.status = AIModelStatus.READY
        self.last_used: Optional[float] = None
        self.total_requests = 0
        self.failed_requests = 0

    @abstractmethod
    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
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
            "success_rate": (self.total_requests - self.failed_requests)
            / max(self.total_requests, 1),
            "last_used": self.last_used,
        }


class AnthropicModel(BaseAIModel):
    """Anthropic Claudeモデル"""

    def __init__(
        self, config: AIModelConfig, tracer: Optional[BaseCallbackHandler] = None
    ):
        super().__init__(config, tracer)
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """LLMを初期化"""
        try:
            self.llm = ChatAnthropic(
                api_key=self.config.api_key,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                max_retries=self.config.max_retries,
            )
            logger.info(
                f"ChatAnthropicモデル {self.config.model_name} を初期化しました"
            )
        except Exception as e:
            logger.error(f"ChatAnthropic初期化エラー: {e}")
            self.status = AIModelStatus.ERROR

    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
        """Claudeで応答を生成"""
        if self.status != AIModelStatus.READY or not self.llm:
            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=0.0,
                success=False,
                error_message=f"モデルが利用不可: {self.status}",
            )

        self.status = AIModelStatus.BUSY
        start_time = time.time()
        callbacks = [self.tracer] if self.tracer else []

        try:
            lc_messages = convert_messages(messages)
            response = await self.llm.ainvoke(
                lc_messages, config={"callbacks": callbacks}
            )

            response_time = time.time() - start_time
            content = response.content
            usage = response.response_metadata.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time
            logger.error(f"ChatAnthropic APIエラー: {e}")

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        try:
            if not self.llm:
                return False

            # シンプルなメッセージでテスト
            test_messages = [AIMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=10)

            return response.success
        except Exception as e:
            logger.error(f"ChatAnthropicヘルスチェックエラー: {e}")
            return False


class OpenAIModel(BaseAIModel):
    """OpenAI GPTモデル"""

    def __init__(
        self, config: AIModelConfig, tracer: Optional[BaseCallbackHandler] = None
    ):
        super().__init__(config, tracer)
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """LLMを初期化"""
        try:
            self.llm = ChatOpenAI(
                api_key=self.config.api_key,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                max_retries=self.config.max_retries,
            )
            logger.info(f"ChatOpenAIモデル {self.config.model_name} を初期化しました")
        except Exception as e:
            logger.error(f"ChatOpenAI初期化エラー: {e}")
            self.status = AIModelStatus.ERROR

    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
        """GPTで応答を生成"""
        if self.status != AIModelStatus.READY or not self.llm:
            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=0.0,
                success=False,
                error_message=f"モデルが利用不可: {self.status}",
            )

        self.status = AIModelStatus.BUSY
        start_time = time.time()
        callbacks = [self.tracer] if self.tracer else []

        try:
            lc_messages = convert_messages(messages)
            response = await self.llm.ainvoke(
                lc_messages, config={"callbacks": callbacks}
            )

            response_time = time.time() - start_time
            content = response.content
            usage = response.response_metadata.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time
            logger.error(f"ChatOpenAI APIエラー: {e}")

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        try:
            if not self.llm:
                return False

            # シンプルなメッセージでテスト
            test_messages = [AIMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=10)

            return response.success
        except Exception as e:
            logger.error(f"ChatOpenAIヘルスチェックエラー: {e}")
            return False


class AzureOpenAIModel(BaseAIModel):
    """Azure OpenAIモデル"""

    def __init__(
        self, config: AIModelConfig, tracer: Optional[BaseCallbackHandler] = None
    ):
        super().__init__(config, tracer)
        self.llm = None
        self.azure_endpoint = os.getenv("OPENAI_API_BASE")
        self.azure_deployment = os.getenv("OPENAI_API_DEPLOYMENT", config.model_name)
        self._initialize_llm()

    def _initialize_llm(self):
        """LLMを初期化"""
        try:
            if not self.azure_endpoint:
                logger.warning("OPENAI_API_BASEが設定されていません")
                self.status = AIModelStatus.UNAVAILABLE
                return

            self.llm = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.config.api_key,
                azure_deployment=self.azure_deployment,
                api_version="2024-12-01-preview",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                max_retries=self.config.max_retries,
            )
            logger.info(
                f"AzureChatOpenAIモデル {self.azure_deployment} を初期化しました"
            )
        except Exception as e:
            logger.error(f"AzureChatOpenAI初期化エラー: {e}")
            self.status = AIModelStatus.ERROR

    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
        """Azure OpenAIで応答を生成"""
        if self.status != AIModelStatus.READY or not self.llm:
            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=0.0,
                success=False,
                error_message=f"モデルが利用不可: {self.status}",
            )

        self.status = AIModelStatus.BUSY
        start_time = time.time()
        callbacks = [self.tracer] if self.tracer else []

        try:
            lc_messages = convert_messages(messages)
            response = await self.llm.ainvoke(
                lc_messages, config={"callbacks": callbacks}
            )

            response_time = time.time() - start_time
            content = response.content
            usage = response.response_metadata.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            self.total_requests += 1
            self.last_used = time.time()

            return AIResponse(
                content=content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True,
            )

        except Exception as e:
            self.failed_requests += 1
            response_time = time.time() - start_time
            logger.error(f"AzureChatOpenAI APIエラー: {e}")

            return AIResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        try:
            if not self.llm:
                return False

            # シンプルなメッセージでテスト
            test_messages = [AIMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=10)

            return response.success
        except Exception as e:
            logger.error(f"AzureChatOpenAIヘルスチェックエラー: {e}")
            return False


class MockModel(BaseAIModel):
    """モックモデル（テスト用）"""

    def __init__(
        self, config: AIModelConfig, tracer: Optional[BaseCallbackHandler] = None
    ):
        super().__init__(config, tracer)
        self.responses = [
            "これはモック応答です。",
            "テスト用の応答になります。",
            "開発環境でのテストに使用してください。",
        ]

    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
        """モック応答を生成"""
        self.status = AIModelStatus.BUSY
        start_time = time.time()

        callbacks = [self.tracer] if self.tracer else []

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
                success=True,
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
                error_message=str(e),
            )
        finally:
            self.status = AIModelStatus.READY

    async def check_health(self) -> bool:
        """ヘルスチェック"""
        return self.status == AIModelStatus.READY


class ModelManager:
    """AIモデル管理クラス"""

    def __init__(self, tracer: Optional[BaseCallbackHandler] = None):
        self.tracer = tracer
        self.models: Dict[str, BaseAIModel] = {}
        self.primary_model: Optional[str] = None
        self.fallback_models: List[str] = []
        self._initialize_models()

    def _initialize_models(self):
        """モデルを初期化"""
        try:
            logger.info("モデル初期化を開始します...")

            # Azure OpenAIモデル（最優先・デフォルト）
            try:
                azure_key = secure_config.azure_openai_key
                azure_endpoint = secure_config.azure_endpoint
                azure_deployment = secure_config.azure_deployment or "gpt-4o-mini"

                logger.info(
                    f"Azure設定確認: key={bool(azure_key)}, endpoint={bool(azure_endpoint)}, deployment={azure_deployment}"
                )

                if azure_key and azure_endpoint:
                    logger.info(f"AzureChatOpenAIモデルを初期化します...")
                    azure_config = AIModelConfig(
                        model_type=AIModelType.AZURE_OPENAI,
                        model_name=azure_deployment,  # Azureでデプロイされたモデル名
                        api_key=azure_key,
                        max_tokens=1000,
                        temperature=0.7,
                        timeout=30.0,
                    )
                    self.models["azure_openai"] = AzureOpenAIModel(
                        azure_config, tracer=self.tracer
                    )
                    self.primary_model = "azure_openai"
                    logger.info(
                        f"✅ AzureChatOpenAIモデルをプライマリとして設定しました（モデル: {azure_deployment}）"
                    )
                else:
                    logger.warning("❌ Azure OpenAI設定が不完全です")
                    logger.info(f"   - APIキー: {bool(azure_key)}")
                    logger.info(f"   - エンドポイント: {bool(azure_endpoint)}")
                    logger.info(f"   - デプロイメント: {azure_deployment}")

            except Exception as e:
                logger.error(f"❌ AzureChatOpenAIモデル初期化エラー: {e}")
                import traceback

                logger.error(f"詳細なエラー情報: {traceback.format_exc()}")

            # OpenAIモデル（Azure OpenAIがなければ）
            try:
                if not self.primary_model and secure_config.openai_api_key:
                    openai_config = AIModelConfig(
                        model_type=AIModelType.OPENAI,
                        model_name="gpt-3.5-turbo",
                        api_key=secure_config.openai_api_key,
                        max_tokens=1000,
                        temperature=0.7,
                        timeout=30.0,
                    )
                    self.models["openai"] = OpenAIModel(
                        openai_config, tracer=self.tracer
                    )
                    self.primary_model = "openai"
                    logger.info("ChatOpenAIモデルをプライマリとして設定しました")
            except Exception as e:
                logger.error(f"ChatOpenAIモデル初期化エラー: {e}")

            # Anthropicモデル（OpenAIがなければ）
            try:
                if not self.primary_model and secure_config.anthropic_api_key:
                    anthropic_config = AIModelConfig(
                        model_type=AIModelType.ANTHROPIC,
                        model_name="claude-3-sonnet-20240229",
                        api_key=secure_config.anthropic_api_key,
                        max_tokens=1000,
                        temperature=0.7,
                        timeout=30.0,
                    )
                    self.models["anthropic"] = AnthropicModel(
                        anthropic_config, tracer=self.tracer
                    )
                    self.primary_model = "anthropic"
                    logger.info("ChatAnthropicモデルをプライマリとして設定しました")
            except Exception as e:
                logger.error(f"ChatAnthropicモデル初期化エラー: {e}")

            # モックモデル（開発用）
            try:
                mock_config = AIModelConfig(
                    model_type=AIModelType.MOCK,
                    model_name="mock-model",
                    api_key="mock-key",
                )
                self.models["mock"] = MockModel(mock_config, tracer=self.tracer)
                if not self.primary_model:
                    self.primary_model = "mock"
                else:
                    self.fallback_models.append("mock")
            except Exception as e:
                logger.error(f"モックモデル初期化エラー: {e}")

            logger.info(
                f"モデルマネージャーを初期化しました。プライマリ: {self.primary_model}, モデル数: {len(self.models)}"
            )

        except Exception as e:
            logger.error(f"モデルマネージャー初期化エラー: {e}")

    async def generate_response(
        self, messages: List[AIMessage], **kwargs
    ) -> AIResponse:
        """応答を生成（フォールバック機能付き）"""
        # プライマリモデルを試行
        if self.primary_model and self.primary_model in self.models:
            response = await self.models[self.primary_model].generate_response(
                messages, **kwargs
            )
            if response.success:
                return response

        # フォールバックモデルを試行
        for model_name in self.fallback_models:
            if model_name in self.models:
                logger.info(f"フォールバックモデル {model_name} を使用します")
                response = await self.models[model_name].generate_response(
                    messages, **kwargs
                )
                if response.success:
                    return response

        # 全モデルが失敗した場合
        return AIResponse(
            content="申し訳ありませんが、現在AIサービスが利用できません。",
            model_used="none",
            tokens_used=0,
            response_time=0.0,
            success=False,
            error_message="全AIモデルが利用不可です",
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
                name: model.get_usage_stats() for name, model in self.models.items()
            },
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

    @staticmethod
    def create_ai_message(role: str, content: str) -> AIMessage:
        """AIMessageインスタンスを作成する静的メソッド"""
        return AIMessage(role=role, content=content)


# グローバルインスタンス
model_manager = ModelManager()
