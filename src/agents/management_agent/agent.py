"""
セッション型経営管理Agent

LangChainで実装した統合経営管理システム
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

# カスタムログ設定をインポートして初期化
from src.shared.config.logging_config import (
    configure_langsmith_tracing,
    get_logger,
    setup_logging,
)

# ログ設定を初期化
setup_logging()
configure_langsmith_tracing()

# ロガーを取得
logger = get_logger(__name__)

import functools
import time
from typing import Any, List

# メモリー関連import
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_chroma import Chroma
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel, Field

from src.domain.models.product import SAMPLE_PRODUCTS
from src.shared.utils.trace_control import conditional_traceable


class BusinessMetrics(BaseModel):
    """事業メトリクス"""

    sales: float
    profit_margin: float
    inventory_level: Dict[str, int]
    customer_satisfaction: float
    timestamp: datetime


class ManagementState(BaseModel):
    """Management Agentの完全な状態管理クラス (VendingBench準拠・Multi-day運用対応)"""

    # ===== セッション管理 =====
    session_id: str = Field(description="セッション固有ID")
    session_type: str = Field(
        description="セッションタイプ (management_flow, node_based_managementなど)"
    )

    # ===== 日時・期間管理 =====
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="セッション開始日時 (ログ保存・メモリ用)",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="最終更新日時"
    )
    business_date: date = Field(
        default_factory=date.today, description="事業日 (営業日単位)"
    )
    day_sequence: int = Field(default=1, description="連続稼働日数 (1日目、2日目...)")

    # ===== ビジネスデータ入力 =====
    business_metrics: Optional[BusinessMetrics] = Field(
        default=None, description="売上、利益、在庫、顧客満足度の基本指標"
    )

    # 分析フェーズの出力
    inventory_analysis: Optional[Dict] = Field(
        default=None,
        description="在庫状況の詳細分析結果（ステータス、アラート、再発注推奨）",
    )

    sales_analysis: Optional[Dict] = Field(
        default=None, description="売上・財務パフォーマンス分析（トレンド、戦略推奨）"
    )

    financial_analysis: Optional[Dict] = Field(
        default=None, description="詳細財務分析結果"
    )

    sales_processing: Optional[Dict] = Field(
        default=None, description="売上処理・販売効率分析結果"
    )

    profit_calculation: Optional[Dict] = Field(
        default=None, description="利益計算・財務健全性詳細分析結果"
    )

    # 戦略決定フェーズ
    pricing_decision: Optional[Dict] = Field(
        default=None, description="価格戦略決定（価格変更、新価格、理由）"
    )

    restock_decision: Optional[Dict] = Field(
        default=None, description="補充タスク決定（製品リスト、タスクID、緊急度）"
    )

    procurement_decision: Optional[Dict] = Field(
        default=None, description="調達依頼決定（製品、数量、発注情報）"
    )

    # 顧客対応
    customer_interaction: Optional[Dict] = Field(
        default=None, description="顧客対応結果（フィードバック、新規キャンペーン）"
    )

    # 実行履歴
    executed_actions: List[Dict] = Field(
        default_factory=list, description="実行済みアクションの履歴"
    )

    # 状態管理
    current_step: str = Field(
        default="initialization", description="現在の処理ステップ"
    )

    processing_status: str = Field(
        default="pending",
        description="全体処理ステータス (pending, processing, completed, error)",
    )

    # エラーハンドリング
    errors: List[str] = Field(
        default_factory=list, description="発生したエラーメッセージ一覧"
    )

    # ===== メモリ連携フィールド (ConversationBufferWindowMemory + VectorStore連携) ====
    # TODO VectorStore未作成
    memory_snapshot: Optional[Dict] = Field(
        default=None, description="直近の会話履歴スナップショット（短期メモリ）"
    )
    learned_patterns: Optional[Dict] = Field(
        default=None, description="VectorStoreからの学習パターン（長期メモリ）"
    )
    historical_insights: List[Dict] = Field(
        default_factory=list,
        description="過去データからの洞察（売上傾向、在庫パターン等）",
    )

    # ===== Multi-day運用フィールド =====
    previous_day_carry_over: Optional[Dict] = Field(
        default=None, description="前日のfinal_reportデータ引き継ぎ"
    )
    cumulative_kpis: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_profit": 0,
            "average_stockout_rate": 0.0,
            "customer_satisfaction_trend": [],
            "action_accuracy_history": [],
        },
        description="全稼働期間の累積KPI（VendingBench Secondary Metrics用）",
    )

    # ===== イベント駆動対応フィールド (Case C向けしばらく未使用) =====
    external_events: List[Dict] = Field(
        default_factory=list, description="人間による制約、突発イベントの履歴"
    )
    agent_communications: List[Dict] = Field(
        default_factory=list, description="他のAgentとのメッセージ交換（Queueベース）"
    )
    pending_human_tasks: List[Dict] = Field(
        default_factory=list, description="人間従業員待ちのタスク（補充、調達依頼等）"
    )

    # ===== ベンチマーク評価フィールド =====
    primary_metrics_history: List[Dict] = Field(
        default_factory=list, description="各実行回のProfit, StockoutRate等の履歴"
    )
    consistency_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="長期的一貫性評価データ"
    )

    # 最終出力
    feedback: Optional[Dict] = Field(
        default=None, description="最終フィードバックと要約"
    )
    final_report: Optional[Dict] = Field(default=None, description="最終総合レポート")


from src.shared import secure_config, settings

logger = logging.getLogger(__name__)


class LangChainLLMAdapter(BaseLanguageModel):
    """model_managerをLangChain LLMインターフェースに変換するadapter"""

    def __init__(self):
        # 循環インポートを避けるため遅延インポート
        self._model_manager = None

    def _get_model_manager(self):
        """遅延インポートでmodel_managerを取得"""
        if self._model_manager is None:
            # 循環インポートを避けるため相対インポート
            from ..infrastructure.ai.model_manager import model_manager

            self._model_manager = model_manager
        return self._model_manager

    def _generate_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """同期版generate response"""
        import asyncio

        async def async_call():
            # BaseMessage -> AIMessage変換
            ai_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    ai_messages.append(AIMessage(role="user", content=msg.content))
                elif isinstance(msg, AIMessage):
                    ai_messages.append(AIMessage(role="assistant", content=msg.content))
                elif isinstance(msg, SystemMessage):
                    ai_messages.append(AIMessage(role="system", content=msg.content))
                else:
                    ai_messages.append(AIMessage(role="user", content=str(msg.content)))

            response = await self._get_model_manager().generate_response(
                ai_messages, **kwargs
            )
            return response.content if response.success else "Error: LLM not available"

        # 同期コンテキストで非同期実行
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既存ループがある場合、新しいスレッドで実行
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_call())
                    return future.result()
            else:
                return loop.run_until_complete(async_call())
        except Exception as e:
            logger.error(f"LLM adapter error: {e}")
            return "Error: Could not generate response"

    async def _agenerate_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """非同期版generate response"""
        from src.infrastructure.ai.model_manager import AIMessage

        # BaseMessage -> AIMessage変換
        ai_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                ai_messages.append(AIMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                ai_messages.append(AIMessage(role="assistant", content=msg.content))
            elif isinstance(msg, SystemMessage):
                ai_messages.append(AIMessage(role="system", content=msg.content))
            else:
                ai_messages.append(AIMessage(role="user", content=str(msg.content)))

        response = await self._get_model_manager().generate_response(
            ai_messages, **kwargs
        )
        return response.content if response.success else "Error: LLM not available"

    @property
    def _llm_type(self) -> str:
        return "model_manager_adapter"

    def _call(self, prompt: str, **kwargs) -> str:
        """LangChain基底メソッド"""
        messages = [HumanMessage(content=prompt)]
        return self._generate_response(messages, **kwargs)

    # BaseLanguageModelのabstract methodsを実装
    def invoke(self, input, config=None, **kwargs):
        """同期invoke"""
        if isinstance(input, str):
            return self._call(input, **kwargs)
        elif hasattr(input, "content"):  # BaseMessageの場合
            return self._generate_response([input], **kwargs)
        else:
            return self._call(str(input), **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        """非同期invoke"""
        from src.infrastructure.ai.model_manager import AIMessage

        if isinstance(input, str):
            messages = [AIMessage(role="user", content=input)]
        elif hasattr(input, "content"):  # BaseMessageの場合
            messages = [input]
        else:
            messages = [AIMessage(role="user", content=str(input))]

        return await self._agenerate_response(messages, **kwargs)

    def generate_prompt(self, prompts, stop=None, **kwargs):
        """プロンプト生成"""
        responses = []
        for prompt in prompts:
            if hasattr(prompt, "__iter__"):
                # プロンプトがメッセージリストの場合
                response = self._generate_response(list(prompt), **kwargs)
            else:
                response = self._call(str(prompt), **kwargs)
            responses.append(response)
        return responses

    async def agenerate_prompt(self, prompts, stop=None, **kwargs):
        """非同期プロンプト生成"""
        responses = []
        for prompt in prompts:
            if hasattr(prompt, "__iter__"):
                response = await self._agenerate_response(list(prompt), **kwargs)
            else:
                messages = [AIMessage(role="user", content=str(prompt))]
                response = await self._agenerate_response(messages, **kwargs)
            responses.append(response)
        return responses

    def predict(self, text, **kwargs):
        """テキスト予測"""
        return self._call(text, **kwargs)

    async def apredict(self, text, **kwargs):
        """非同期テキスト予測"""
        messages = [AIMessage(role="user", content=text)]
        return await self._agenerate_response(messages, **kwargs)

    def predict_messages(self, messages, **kwargs):
        """メッセージ予測"""
        return self._generate_response(messages, **kwargs)

    async def apredict_messages(self, messages, **kwargs):
        """非同期メッセージ予測"""
        return await self._agenerate_response(messages, **kwargs)


class SessionInfo(BaseModel):
    """セッション情報"""

    session_id: str
    session_type: str  # "morning_routine", "midday_check", "evening_summary"
    start_time: datetime
    end_time: Optional[datetime] = None
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    actions_executed: List[Dict[str, Any]] = Field(default_factory=list)


class BusinessMetrics(BaseModel):
    """事業メトリクス"""

    sales: float
    profit_margin: float
    inventory_level: Dict[str, int]
    customer_satisfaction: float
    timestamp: datetime


class NodeBasedManagementAgent:
    """Node-Based経営管理Agent (RunnableSequence + AgentExecutor)"""

    def __init__(
        self, llm_manager=None, agent_objectives=None, provider: str = "openai"
    ):
        """
        Args:
            llm_manager: LLMマネージャーインスタンス (AgentBuilderから注入、またはNoneで自動取得)
            agent_objectives: Agent設定(目的・制約) (AgentBuilderから注入、またはNoneで設定から取得)
            provider: LLMプロバイダー ("anthropic" or "openai" - 後方互換用)
        """
        # 依存関係注入 (AgentBuilder優先、Noneなら自動取得)
        if llm_manager is not None:
            self.llm_manager = llm_manager
            logger.info("NodeBasedManagementAgent: LLM Manager injected from builder")
        else:
            # 後方互換用: 直接import (循環インポートの問題あり)
            from src.infrastructure.ai import model_manager

            self.llm_manager = model_manager
            logger.info(
                "NodeBasedManagementAgent: LLM Manager auto-loaded (legacy mode)"
            )

        # 設定注入 (AgentBuilder優先、Noneなら設定から取得)
        if agent_objectives is not None:
            self.agent_objectives = agent_objectives
            logger.info(
                "NodeBasedManagementAgent: Agent objectives injected from builder"
            )
        else:
            # 後方互換用: 直接読み込み
            self.agent_objectives = settings.agent_objectives
            logger.info(
                "NodeBasedManagementAgent: Agent objectives loaded from settings"
            )

        # 後方互換用パラメータ
        self.provider = provider
        self.current_session: Optional[SessionInfo] = None
        self._system_prompt_logged = False  # システムプロンプトログ出力フラグ

        logger.info(f"NodeBasedManagementAgent initialized (provider: {provider})")

        # LLM接続確認 (直接参照に変更)
        self._verify_llm_connection()

        # システムプロンプト生成
        self.system_prompt = self._generate_system_prompt()

        # メモリー初期化
        self._initialize_memory()

        # Node定義 (Case A)
        self.nodes = self._create_nodes()

        # LCELパイプライン構築 (Case A - StateGraphではなくRunnableSequenceを使用)
        self.chain = self._build_lcel_pipeline()

        # ツールの実装インポート
        from src.agents.management_agent.management_tools.update_pricing import (
            update_pricing,
        )

        # ツール実装をメソッドとして設定
        self.update_pricing = update_pricing

        # ツールの初期化
        self.tools = self._create_tools()

    async def _verify_llm_connection_async(self):
        """SessionBasedManagementAgent初期化時LLM接続確認（非同期版）"""
        logger.info("SessionBasedManagementAgentのLLM接続を確認しています...")

        try:
            # 遅延インポートして循環インポートを回避
            from src.infrastructure.ai.model_manager import model_manager

            # ヘルスチェックを実行
            health_results = await model_manager.check_all_models_health()

            # 結果のログ出力
            for model_name, is_healthy in health_results.items():
                if is_healthy:
                    logger.info(f"✅ AIモデル {model_name}: 接続確認成功")
                else:
                    logger.warning(f"❌ AIモデル {model_name}: 接続失敗")

            # 少なくとも1つのモデルが利用可能か確認
            available_models = [
                name for name, healthy in health_results.items() if healthy
            ]
            if not available_models:
                logger.warning(
                    "⚠️ 利用可能なAIモデルがありません。システムは制限モードで動作します。"
                )
            else:
                logger.info(
                    f"🚀 AI処理準備完了（利用可能モデル: {', '.join(available_models)}）"
                )

        except Exception as e:
            logger.error(f"LLM接続確認中にエラーが発生しました: {e}", exc_info=True)
            logger.warning("⚠️ AIモデル接続確認をスキップします。")

    def _verify_llm_connection(self):
        """SessionBasedManagementAgent初期化時LLM接続確認 - 同期版フォールバック"""
        logger.info("LLM接続確認を同期的に実行")
        try:
            # シンプルな同期的なヘルスチェック
            logger.info(
                "✅ LLM接続確認: 同期モードで動作 - 詳細なチェックは実行時に行われます"
            )
        except Exception as e:
            logger.error(f"LLM接続確認エラー: {e}")
            logger.warning("⚠️ AIモデル接続確認をスキップします。")

    def _get_memory_context(self, node_name: str) -> str:
        """
        指定ノードの過去ビジネス洞察を取得（LangSmithトレース付き）

        Args:
            node_name: ノード名 (例: "inventory_check", "sales_plan")

        Returns:
            過去洞察の要約文字列
        """
        if not self.short_term_memory:
            logger.debug(f"No memory available for {node_name}")
            return "No previous context available."

        try:
            # メモリーから過去の会話を取得
            # ConversationBufferMemoryは最新の会話を優先的に返す
            memory_variables = self.short_term_memory.load_memory_variables({})

            # 各Nodeに関連する過去洞察をフィルタリング
            relevant_insights = []
            node_prefix = f"Previous {node_name} insight:"

            if "history" in memory_variables:
                history = memory_variables["history"]
                if isinstance(history, str):
                    # 単一の履歴文字列の場合
                    if node_prefix in history:
                        # このNodeの洞察を含むセッションを特定
                        insights = history.split(node_prefix)
                        for insight in insights[1:]:  # 最初の要素はprefix前の部分
                            clean_insight = insight.split("\nAssistant: ")[0].strip()
                            relevant_insights.append(clean_insight[:200])  # 長さ制限

                elif isinstance(history, list):
                    # 会話履歴のリストの場合
                    for msg in history:
                        if hasattr(msg, "content") and node_prefix in msg.content:
                            # このNodeの洞察を抽出
                            content = msg.content
                            insight_start = content.find(node_prefix) + len(node_prefix)
                            insight_end = content.find("\n", insight_start)
                            if insight_end == -1:
                                insight_end = len(content)

                            insight = content[insight_start:insight_end].strip()
                            relevant_insights.append(insight[:200])

            if relevant_insights:
                # 最新3つの洞察を結合
                context = " | ".join(relevant_insights[-3:])
                logger.info(
                    f"Retrieved {len(relevant_insights)} insights for {node_name}: {context[:100]}..."
                )
                return context
            else:
                logger.debug(f"No relevant insights found for {node_name}")
                return f"No previous {node_name} insights available."

        except Exception as e:
            logger.warning(f"Error retrieving memory context for {node_name}: {e}")
            return f"Memory retrieval error for {node_name}."

    def _extract_and_save_business_insight(self, node_name: str, llm_response: str):
        """
        LLMレスポンスからビジネス洞察を抽出し、メモリーに保存（LangSmithトレース付き）

        Args:
            node_name: ノード名
            llm_response: LLMからのレスポンス文字列
        """
        try:
            # LLMレスポンスからビジネス洞察を抽出
            insight = self._extract_business_insight(llm_response, node_name)

            if insight and insight != "No insight extracted":
                # 洞察をメモリーに保存（LangSmithトレース）
                self._save_business_insight(node_name, insight)
                logger.info(
                    f"Saved business insight for {node_name}: {insight[:100]}..."
                )

                # LangSmithトレース用メタデータ
                trace_metadata = {
                    "node_name": node_name,
                    "insight_extracted": insight[:200],
                    "insight_length": len(insight),
                    "memory_saved": True,
                    "timestamp": datetime.now().isoformat(),
                }

            else:
                logger.debug(
                    f"No significant insight extracted from {node_name} response"
                )

        except Exception as e:
            logger.error(f"Error extracting/saving insight for {node_name}: {e}")

    def _extract_business_insight(self, llm_response: str, node_name: str) -> str:
        """
        LLMレスポンスからビジネス洞察を抽出

        Args:
            llm_response: LLMレスポンス
            node_name: ノード名

        Returns:
            抽出されたビジネス洞察の要約
        """
        if not llm_response or len(llm_response.strip()) < 50:
            return "No insight extracted"

        try:
            # Node別の洞察抽出ロジック
            insights = {
                "inventory_check": self._extract_inventory_insight,
                "sales_plan": self._extract_sales_insight,
                "pricing": self._extract_pricing_insight,
                "profit_calculation": self._extract_profit_insight,
                "customer_interaction": self._extract_customer_insight,
            }

            extract_func = insights.get(node_name, self._extract_generic_insight)
            insight = extract_func(llm_response)

            return insight if insight else "No specific insight extracted"

        except Exception as e:
            logger.debug(f"Insight extraction failed for {node_name}: {e}")
            return "Insight extraction error"

    def _extract_inventory_insight(self, response: str) -> str:
        """在庫分析からの洞察抽出"""
        if "critical" in response.lower():
            return "在庫状況が危機的。緊急補充が必要"
        elif "low" in response.lower():
            return "在庫水準が低下傾向。計画的補充を検討"
        elif "normal" in response.lower():
            return "在庫状況は安定。現在の管理方針を継続"
        return f"在庫分析: {response[:100]}..." if len(response) > 100 else response

    def _extract_sales_insight(self, response: str) -> str:
        """売上分析からの洞察抽出"""
        if "concerning" in response.lower():
            return "売上動向が懸念される。戦略見直しが必要"
        elif "positive" in response.lower():
            return "売上トレンド良好。既存戦略を強化"
        elif "stable" in response.lower():
            return "売上安定。リスク分散戦略を検討"
        return f"売上分析: {response[:100]}..." if len(response) > 100 else response

    def _extract_pricing_insight(self, response: str) -> str:
        """価格戦略からの洞察抽出"""
        if "increase" in response.lower():
            return "価格引き上げ戦略採用。利益率改善を優先"
        elif "maintain" in response.lower():
            return "価格安定戦略。競争力維持を重視"
        elif "decrease" in response.lower():
            return "価格引き下げ戦略。市場シェア拡大を狙う"
        return f"価格戦略: {response[:100]}..." if len(response) > 100 else response

    def _extract_profit_insight(self, response: str) -> str:
        """利益計算からの洞察抽出"""
        if "excellent" in response.lower():
            return "財務状況極めて良好。事業拡大の機会"
        elif "good" in response.lower():
            return "財務状況良好。安定経営を継続"
        elif "critical" in response.lower():
            return "財務状況が危機的。抜本的な改善が必要"
        return f"財務分析: {response[:100]}..." if len(response) > 100 else response

    def _extract_customer_insight(self, response: str) -> str:
        """顧客対応からの洞察抽出"""
        if "improve" in response.lower():
            return "顧客満足度向上施策が必要"
        elif "campaign" in response.lower():
            return "エンゲージメントキャンペーン実施"
        elif "monitor" in response.lower():
            return "顧客フィードバックの継続監視を継続"
        return f"顧客対応: {response[:100]}..." if len(response) > 100 else response

    def _extract_generic_insight(self, response: str) -> str:
        """汎用洞察抽出"""
        # 最初の意味のある文を抽出（JSONレスポンスの場合はスキップ）
        if response.startswith("{") or response.startswith("["):
            return "Structured response received"

        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            if len(line) > 20 and not line.startswith("```"):
                return line[:150] + "..." if len(line) > 150 else line

        return response[:150] + "..." if len(response) > 150 else response

    def _save_business_insight(self, node_name: str, insight: str):
        """
        ビジネス洞察を短期メモリーに保存（LangSmithトレース付き）

        Args:
            node_name: ノード名
            insight: 保存する洞察内容
        """
        if not self.short_term_memory:
            logger.debug("Short-term memory not available")
            return

        try:
            # 会話形式で洞察を保存
            # System: 過去の洞察 -> Assistant: [洞察内容]
            self.short_term_memory.save_context(
                inputs={"input": f"Previous {node_name} insight:"},  # 人間の入力
                outputs={"output": insight},  # AIの出力
            )

            logger.debug(f"Saved insight to memory: {node_name} -> {insight[:50]}...")

        except Exception as e:
            logger.error(f"Failed to save business insight for {node_name}: {e}")

    def _generate_system_prompt(self) -> str:
        """Agent目的設定に基づいてシステムプロンプトを生成"""
        objectives = self.agent_objectives

        prompt = f"""
あなたは自動販売機事業の経営者です。以下の設定に基づいて意思決定を行ってください。

【主要目的】
{chr(10).join(f"- {obj}" for obj in objectives["primary"])}

【最適化期間枠設定】(戦略的優先度: {objectives["priority_weight"]})
"""

        for period_key, descriptions in objectives["optimization_period"].items():
            weight = objectives["priority_weight"].get(period_key, 0.0)
            prompt += f"- {period_key}: {descriptions} (重み: {weight})\n"

        prompt += f"""
【制約条件】
{chr(10).join(f"- {constraint}" for constraint in objectives["constraints"])}

【業務統括】
- 売上・財務データの分析と戦略立案
- 在庫状況の監視と補充計画
- 価格戦略の決定と実行指示
- 従業員への作業指示（補充、調達、メンテナンス）
- 顧客からの問い合わせ対応と苦情処理

【意思決定原則】
- 短期・中期・長期目標のバランスを考慮して収益性を最優先
- 顧客満足度を維持しつつ長期的な成長を図る
- リスクを適切に管理し、安定的な事業運営を行う
- データに基づいた戦略的判断を行う
"""

        return prompt

    def _initialize_memory(self):
        """メモリー初期化"""
        # 短期メモリー: ConversationBufferWindowMemoryの警告を修正 (ConversationBufferMemoryを使用)
        try:
            # LangChain v0.1.x以降ではConversationBufferWindowMemoryが非推奨
            # ConversationBufferMemoryに変更し、max_token_limitで制限を設ける
            from langchain.memory import ConversationBufferMemory

            self.short_term_memory = ConversationBufferMemory(
                max_token_limit=1000,  # トークン50個/対話 × 20対話程度の制限
                return_messages=True,
            )
        except ImportError:
            # 旧バージョンの場合のみConversationBufferWindowMemoryを使用
            try:
                self.short_term_memory = ConversationBufferWindowMemory(k=5)
            except Exception as e:
                logger.warning(f"メモリー初期化失敗、簡易Fallbackを使用: {e}")
                self.short_term_memory = None

        # 長期メモリー: VectorStoreRetrieverMemory (ベクターストアによる検索)
        try:
            # Azure検知を回避するために明示的にOpenAIパラメータを設定
            import os

            if "AZURE_OPENAI" in os.environ or "OPENAI_API_KEY" not in os.environ:
                logger.info(
                    "Using simple fallback for embeddings (Azure/conf key issue)"
                )
                raise Exception("Azure OpenAI detected or no OpenAI key")

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # より高速で安価なモデルを使用
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
            )

            vectorstore = Chroma(
                collection_name="agent_memory", embedding_function=embeddings
            )
            self.long_term_memory = VectorStoreRetrieverMemory(
                retriever=vectorstore.as_retriever()
            )
        except Exception as e:
            logger.warning(f"長期メモリー初期化に失敗、簡易Fallbackを使用: {e}")
            self.long_term_memory = None

    def _create_nodes(self):
        """Case Aのノード群を定義"""
        return {
            "inventory_check": self.inventory_check_node,
            "sales_plan": self.sales_plan_node,
            "pricing": self.pricing_node,
            "restock": self.restock_node,
            "procurement": self.procurement_request_generation_node,
            "sales_processing": self.sales_processing_node,
            "customer_interaction": self.customer_interaction_node,
            "profit_calculation": self.profit_calculation_node,
            "feedback": self.feedback_node,
        }

    def _build_state_graph(self):
        """StateGraphによるグラフ構築 (Case A) - 直線的ノード接続"""
        try:
            # LangGraph importとバージョン確認
            import langgraph

            try:
                version = getattr(langgraph, "__version__", "unknown")
                logger.info(f"Using LangGraph version: {version}")
            except:
                logger.info("Using LangGraph (version unknown)")
            from langgraph.graph import StateGraph

            # StateGraph初期化 (ManagementStateを使用)
            graph = StateGraph(ManagementState)
            logger.info("StateGraph initialized with ManagementState")

            print("DEBUG: Importing langgraph modules...")
            try:
                from langgraph.graph import StateGraph

                print("DEBUG: StateGraph imported successfully")
            except Exception as e:
                print(f"DEBUG: StateGraph import failed: {e}")
                self.state_graph = None
                return

            # 各ノードを追加 (Case A: 9つのノード全て)
            graph.add_node("inventory_check", self.inventory_check_node)
            graph.add_node("sales_plan", self.sales_plan_node)
            graph.add_node("pricing", self.pricing_node)
            graph.add_node("restock", self.restock_node)
            graph.add_node("procurement", self.procurement_request_generation_node)
            graph.add_node("sales_processing", self.sales_processing_node)
            graph.add_node("customer_interaction", self.customer_interaction_node)
            graph.add_node("profit_calculation", self.profit_calculation_node)
            graph.add_node("feedback", self.feedback_node)
            logger.info("All nodes added to StateGraph")

            # 直線的なノード接続 (Case A: 各ノードを順番に遷移)
            graph.add_edge("inventory_check", "sales_plan")
            graph.add_edge("sales_plan", "pricing")
            graph.add_edge("pricing", "restock")
            graph.add_edge("restock", "procurement")
            graph.add_edge("procurement", "sales_processing")
            graph.add_edge("sales_processing", "customer_interaction")
            graph.add_edge("customer_interaction", "profit_calculation")
            graph.add_edge("profit_calculation", "feedback")
            logger.info("All edges added to StateGraph")

            # エントリーポイントを設定 (最初のノード)
            graph.set_entry_point("inventory_check")
            logger.info("Entry point set to inventory_check")

            # グラフをコンパイル
            print("DEBUG: Attempting to compile StateGraph...")
            try:
                compiled_graph = graph.compile()
                self.state_graph = compiled_graph
                print(
                    f"DEBUG: StateGraph compiled successfully, type: {type(self.state_graph)}"
                )
                logger.info("✅ StateGraph for Case A compiled successfully")
            except Exception as compile_e:
                print(f"DEBUG: StateGraph compile failed: {compile_e}")
                self.state_graph = None
                logger.error(f"StateGraph compile failed: {compile_e}")
                import traceback

                logger.error(f"StateGraph compile traceback: {traceback.format_exc()}")

        except ImportError as e:
            logger.error(f"LangGraph import error: {e}. Please install langgraph")
            self.state_graph = None
        except Exception as e:
            logger.error(f"StateGraph build failed: {e}")
            logger.error(f"StateGraph build error type: {type(e).__name__}")
            import traceback

            logger.error(f"StateGraph build traceback: {traceback.format_exc()}")
            self.state_graph = None

    def _build_lcel_pipeline(self):
        """LCEL RunnableSequenceによるパイプライン構築 (Case A)"""
        try:
            from langchain_core.runnables import RunnableLambda, RunnableSequence

            # 各ノードをRunnableLambdaでラップ
            inventory_runnable = RunnableLambda(self.inventory_check_node)
            sales_plan_runnable = RunnableLambda(self.sales_plan_node)
            pricing_runnable = RunnableLambda(self.pricing_node)
            restock_runnable = RunnableLambda(self.restock_node)
            procurement_runnable = RunnableLambda(
                self.procurement_request_generation_node
            )
            sales_processing_runnable = RunnableLambda(self.sales_processing_node)
            customer_runnable = RunnableLambda(self.customer_interaction_node)
            profit_runnable = RunnableLambda(self.profit_calculation_node)
            feedback_runnable = RunnableLambda(self.feedback_node)

            # ノードを直線的に接続したRunnableSequenceを作成
            # input -> inventory_check -> sales_plan -> pricing -> restock -> procurement -> sales_processing -> customer_interaction -> profit_calculation -> feedback -> output
            self.chain = RunnableSequence(
                inventory_runnable,
                sales_plan_runnable,
                pricing_runnable,
                restock_runnable,
                procurement_runnable,
                sales_processing_runnable,
                customer_runnable,
                profit_runnable,
                feedback_runnable,
            )

            logger.info("✅ LCEL RunnableSequence pipeline built successfully")
            return self.chain

        except Exception as e:
            logger.error(f"LCEL Pipeline build failed: {e}")
            import traceback

            logger.error(f"Pipeline build traceback: {traceback.format_exc()}")
            self.chain = None
            return None

    def _build_chain(self):
        """後方互換用chain構築 (使用推奨せず)"""
        return self._build_lcel_pipeline()

    def _test_llm_connection_sync(self):
        """LLM接続確認（同期版） - 循環インポートを回避するため削除"""
        # このメソッドは循環インポートを避けるためコメントアウト
        # 実際の接続確認は_asyncバージョンのみを使用
        pass

    def _create_tools(self) -> List[StructuredTool]:
        """LangChainツールの作成 - Tool Registry使用"""
        from src.agents.management_agent.tools.tool_registry import create_tool_registry

        # Tool Registryから全ツールを取得
        return create_tool_registry()

    # Note: Tool definition moved to tool_registry.py
    # Old tool creation methods removed to eliminate duplication

    # ツール実装メソッド

    def get_business_metrics(self) -> Dict[str, Any]:
        """ビジネスメトリクスを取得（実際のシステムと連携）"""
        logger.info("Getting business metrics from actual systems")

        try:
            # 各種サービスをインポート
            from datetime import date, timedelta

            from src.application.services.inventory_service import inventory_service
            from src.domain.accounting.management_accounting import management_analyzer

            # 在庫情報を取得
            inventory_summary = inventory_service.get_inventory_summary()
            inventory_level = {}

            # 商品別在庫を集計
            for slot in inventory_service.vending_machine_slots.values():
                product_name = slot.product_name.lower()
                if product_name not in inventory_level:
                    inventory_level[product_name] = 0
                inventory_level[product_name] += slot.current_quantity

            # 財務情報を取得（管理会計から）
            end_date = date.today()
            start_date = end_date - timedelta(days=30)

            # 売上情報を取得（会計システムから）
            sales = abs(
                management_analyzer.journal_processor.get_account_balance(
                    "4001", start_date, end_date
                )
            )

            period_profitability = management_analyzer.analyze_period_profitability(
                start_date, end_date
            )
            profit_margin = period_profitability.get("gross_margin", 0.35)

            # 顧客満足度の計算
            # 在庫充足率と売上実績から推定
            total_inventory = sum(inventory_level.values())
            max_inventory = (
                len(inventory_service.vending_machine_slots) * 50
            )  # 想定最大在庫
            inventory_score = (
                min(total_inventory / max_inventory, 1.0) if max_inventory > 0 else 0.5
            )

            # 売上目標との比較（月間目標: 100万円）
            monthly_target = 1000000
            sales_score = min(sales / monthly_target, 1.0)

            # 総合満足度（3.0-5.0のスケール）
            customer_satisfaction = 3.0 + (inventory_score * 1.0 + sales_score * 1.0)

            metrics_result = {
                "sales": round(sales, 2),
                "profit_margin": round(profit_margin, 3),
                "inventory_level": inventory_level,
                "customer_satisfaction": round(customer_satisfaction, 2),
                "timestamp": datetime.now().isoformat(),
                "inventory_status": {
                    "total_slots": len(inventory_service.vending_machine_slots),
                    "low_stock_count": len(inventory_service.get_low_stock_slots()),
                    "out_of_stock_count": len(
                        inventory_service.get_out_of_stock_slots()
                    ),
                },
                "sales_stats": {
                    "total_revenue": sales,  # 会計システムから取得
                },
            }

            # デバッグログ: 取得したビジネスデータをログ出力
            logger.debug("=== BUSINESS METRICS RETRIEVED ===")
            logger.debug(f"Sales (accounting_system): ¥{sales:.2f}")
            logger.debug(f"Profit Margin: {profit_margin:.3f}")
            logger.debug(f"Inventory Level: {inventory_level}")
            logger.debug(f"Inventory Status: {metrics_result['inventory_status']}")
            logger.debug(f"Customer Satisfaction: {customer_satisfaction:.2f}")
            logger.debug("=== END BUSINESS METRICS ===")

            return metrics_result

        except Exception as e:
            logger.error(f"ビジネスメトリクス取得エラー: {e}", exc_info=True)
            # エラー時はフォールバック値を返す
            return {
                "sales": 0.0,
                "profit_margin": 0.0,
                "inventory_level": {},
                "customer_satisfaction": 3.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    @conditional_traceable(name="financial_performance_analysis")
    async def analyze_financial_performance(self) -> Dict[str, Any]:
        """財務パフォーマンスを分析（注入されたllm_manager経由）"""
        logger.info("Analyzing financial performance using LLM")
        try:
            metrics = self.get_business_metrics()

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user",
                    content=f"""
以下の財務データを分析し、パフォーマンス評価と改善提案を行ってください。

【財務データ】
- 売上: ¥{metrics["sales"]:,}
- 利益率: {metrics["profit_margin"]:.1%}
- 在庫状況: {metrics["inventory_level"]}
- 顧客満足度: {metrics["customer_satisfaction"]}/5.0

【出力形式】
JSON形式で回答してください：
```json
{{
    "analysis": "財務状況の全体的な評価と分析（100文字以上）",
    "recommendations": ["改善提案1", "改善提案2", "改善提案3"]
}}
```
""",
                ),
            ]

            response = await self.llm_manager.generate_response(
                messages, max_tokens=1000
            )

            if response.success:
                try:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    llm_response = json.loads(content)
                    return {
                        "analysis": llm_response.get(
                            "analysis", "分析できませんでした"
                        ),
                        "recommendations": llm_response.get("recommendations", []),
                        "metrics": metrics,
                    }
                except json.JSONDecodeError:
                    logger.warning(
                        f"財務分析LLMレスポンスパース失敗: {response.content}"
                    )

            # LLM失敗時はハードコードされたレスポンス
            logger.warning("LLM財務分析失敗、ハードコードレスポンスを使用")
            return {
                "analysis": "売上は予算比95%で推移。利益率は良好。",
                "recommendations": ["在庫回転率の改善", "高利益商品の強化"],
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"財務パフォーマンス分析エラー: {e}")
            metrics = self.get_business_metrics()
            return {
                "analysis": f"分析エラー: {str(e)}",
                "recommendations": ["管理者へ連絡してください"],
                "metrics": metrics,
            }

    @conditional_traceable(name="inventory_status_analysis")
    async def check_inventory_status(self) -> Dict[str, Any]:
        """在庫状況を確認（注入されたllm_manager経由）"""
        logger.info("Checking inventory status using LLM")
        try:
            metrics = self.get_business_metrics()
            inventory_level = metrics["inventory_level"]

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user",
                    content=f"""
以下の在庫状況を分析し、在庫管理の推奨事項を提案してください。

【現在の在庫状況】
{inventory_level}

【出力形式】
JSON形式で回答してください：
```json
{{
    "status": "在庫状況の全体評価 (normal/critical/low)",
    "low_stock_items": ["在庫の少ない商品名リスト"],
    "reorder_needed": ["発注が必要な商品名リスト"],
    "estimated_stockout": {{"商品名": "在庫切れ予測日"}}
}}
```
""",
                ),
            ]

            response = await self.llm_manager.generate_response(
                messages, max_tokens=1000
            )

            if response.success:
                try:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    llm_response = json.loads(content)
                    return llm_response
                except json.JSONDecodeError:
                    logger.warning(
                        f"在庫状況LLMレスポンスパース失敗: {response.content}"
                    )

            # LLM失敗時はハードコードされたレスポンス
            logger.warning("LLM在庫分析失敗、ハードコードレスポンスを使用")
            return {
                "status": "normal",
                "low_stock_items": ["water"],
                "reorder_needed": ["water"],
                "estimated_stockout": {"water": "2日後"},
            }

        except Exception as e:
            logger.error(f"在庫状況確認エラー: {e}")
            return {
                "status": "error",
                "low_stock_items": [],
                "reorder_needed": [],
                "estimated_stockout": {},
            }

    def assign_restocking_task(
        self, products: List[str], urgency: str = "normal"
    ) -> Dict[str, Any]:
        """補充タスクを割り当て"""
        logger.info("Tool assign_restocking called")
        logger.info(f"Assigning restocking task for {products} with urgency {urgency}")
        task_id = str(uuid4())
        return {
            "task_id": task_id,
            "task_type": "restocking",
            "products": products,
            "urgency": urgency,
            "assigned": True,
            "deadline": (
                datetime.now() + timedelta(hours=2 if urgency == "urgent" else 24)
            ).isoformat(),
        }

    def request_procurement(
        self, products: List[str], quantity: Dict[str, int]
    ) -> Dict[str, Any]:
        """調達を依頼"""
        logger.info(f"Requesting procurement for {products}")
        order_id = str(uuid4())
        return {
            "order_id": order_id,
            "products": products,
            "quantity": quantity,
            "status": "pending",
            "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
        }

    def schedule_maintenance(self, task: str, date: str) -> Dict[str, Any]:
        """メンテナンスをスケジュール"""
        logger.info(f"Scheduling maintenance: {task} on {date}")
        return {"success": True, "task": task, "scheduled_date": date}

    async def coordinate_employee_tasks(self) -> Dict[str, Any]:
        """発注/補充が必要な場合に従業員1人にメール通知 + 新商品発注処理"""
        logger.info("Coordinating employee tasks")

        notifications = []
        employees_status = {}

        # === 1. 在庫補充タスク ===
        inventory_status = await self.check_inventory_status()
        low_stock_items = inventory_status.get("low_stock_items", [])

        if low_stock_items:
            notification = {
                "recipient": "employee@vending-company.com",
                "subject": "在庫補充依頼",
                "body": f"以下の商品が在庫不足です。補充をお願いします: {', '.join(low_stock_items)}",
                "priority": "normal",
                "timestamp": datetime.now().isoformat(),
                "task_type": "restock",
            }
            notifications.append(notification)
            employees_status["restock"] = low_stock_items
            logger.info(f"在庫補充通知送信: {low_stock_items}")

        # === 2. 新商品発注タスク ===
        # 在庫データを基に新商品検索クエリを生成
        try:
            # 在庫状況から商品カテゴリを把握
            metrics = self.get_business_metrics()
            inventory_level = metrics.get("inventory_level", {})
            sales = metrics.get("sales", 0)

            # カテゴリ別在庫を確認
            drink_categories = [
                item
                for item in inventory_level.keys()
                if "コーラ" in item or "飲料" in item or "ジュース" in item
            ]
            food_categories = [
                item
                for item in inventory_level.keys()
                if "チップス" in item or "ヌードル" in item or "お菓子" in item
            ]

            # 売上実績に基づいて検索クエリを決定
            if sales > 1000:  # 売上が良い場合
                search_query = "人気飲料 新商品"
                logger.info("売上好調のため、新商品飲料を検索")
            elif (
                drink_categories
                and min([inventory_level.get(cat, 0) for cat in drink_categories]) < 5
            ):  # 飲料在庫が少ない場合
                search_query = "人気清涼飲料 ボトル飲料"
                logger.info("飲料在庫不足のため、供給安定した飲料を検索")
            elif food_categories:
                search_query = "人気スナック 健康志向"
                logger.info("既存食品を補完する人気スナックを検索")
            else:
                search_query = "人気飲料"
                logger.info("デフォルトで人気飲料を検索")

            logger.info(f"生成された検索クエリ: {search_query}")

            # Shared Toolsから商品検索機能を使用
            from src.agents.shared_tools import shared_registry

            search_tool = shared_registry.get_tool("market_search")
            if search_tool:
                search_results = await search_tool.asearch(query=search_query)
                logger.info(
                    f"検索結果取得: {len(search_results) if search_results else 0}件 (クエリ: {search_query})"
                )
                recommended_products = (
                    search_results[:2] if search_results else []
                )  # 上位2つ
            else:
                recommended_products = []
                logger.warning("検索ツールが利用できません")

            if recommended_products:
                procurement_tasks = []
                for product in recommended_products[:2]:  # dict形式を想定
                    # Procurement AgentからShared Toolsに変更
                    procurement_tool = shared_registry.get_tool("procurement_order")
                    if procurement_tool:
                        procurement_result = await procurement_tool.aexecute(
                            product_info={
                                "product_name": product.get("name", "") or product,
                                "recommended_quantity": 10,
                            },
                            supplier_info={
                                "name": "Search Supplier",
                                "url": product.get("url", ""),
                                "price": product.get("price", 150),
                            },
                        )

                        if procurement_result.get("success"):
                            order = procurement_result.get("order", {})
                            procurement_tasks.append(
                                {
                                    "product": product.get("name", "") or product,
                                    "order_id": order.get("order_id", "unknown"),
                                }
                            )

                if procurement_tasks:
                    procurement_notification = {
                        "recipient": "employee@vending-company.com",
                        "subject": "新商品発注完了通知",
                        "body": f"以下の新商品を発注しました。入荷管理をお願いします:\n"
                        + "\n".join(
                            [
                                f"- {t['product']} (注文ID: {t['order_id']})"
                                for t in procurement_tasks
                            ]
                        ),
                        "priority": "high",
                        "timestamp": datetime.now().isoformat(),
                        "task_type": "new_procurement",
                        "orders": procurement_tasks,
                    }
                    notifications.append(procurement_notification)
                    employees_status["new_procurement"] = [
                        t["product"] for t in procurement_tasks
                    ]
                    logger.info(f"新商品発注通知送信: {len(procurement_tasks)}件")

        except Exception as e:
            logger.error(f"新商品発注プロセスエラー: {e}")

        # === 結果返却 ===
        if notifications:
            return {
                "active_tasks": len(notifications),
                "completed_today": 0,
                "pending": len(notifications),
                "notifications_sent": notifications,
                "employees": {"employee@vending-company.com": employees_status},
            }
        else:
            return {
                "active_tasks": 0,
                "completed_today": 0,
                "pending": 0,
                "notifications_sent": [],
                "employees": {"employee@vending-company.com": "特記事項なし"},
            }

    def respond_to_customer_inquiry(
        self, customer_id: str, inquiry: str
    ) -> Dict[str, Any]:
        """顧客問い合わせに対応"""
        logger.info(f"Responding to customer {customer_id} inquiry")
        return {
            "customer_id": customer_id,
            "inquiry": inquiry,
            "response": "お問い合わせありがとうございます。担当者が確認して折り返しご連絡いたします。",
            "status": "responded",
        }

    def handle_customer_complaint(
        self, customer_id: str, complaint: str
    ) -> Dict[str, Any]:
        """顧客苦情を処理"""
        logger.info(f"Handling complaint from customer {customer_id}")
        return {
            "customer_id": customer_id,
            "complaint": complaint,
            "resolution": "商品の返金処理を行い、次回使用可能なクーポンを発行しました。",
            "status": "resolved",
            "compensation": "500円クーポン",
        }

    def collect_customer_feedback(self) -> Dict[str, Any]:
        """顧客フィードバックを収集"""
        logger.info("Collecting customer feedback")
        return {
            "feedback_count": 15,
            "average_rating": 4.2,
            "top_requests": ["新しいフレーバー", "温かい飲み物", "健康志向商品"],
            "trends": "健康志向商品への関心が高まっている",
        }

    def create_customer_engagement_campaign(self, campaign_type: str) -> Dict[str, Any]:
        """エンゲージメントキャンペーンを作成"""
        logger.info(f"Creating {campaign_type} campaign")
        return {
            "campaign_type": campaign_type,
            "target": "全顧客",
            "duration": "2週間",
            "expected_impact": "売上10%増",
            "status": "planned",
        }

    async def start_management_session(self, session_type: str) -> str:
        """管理セッションを開始"""
        session_id = str(uuid4())
        self.current_session = SessionInfo(
            session_id=session_id, session_type=session_type, start_time=datetime.now()
        )

        logger.info(f"Started {session_type} session: {session_id}")
        return session_id

    async def end_management_session(self) -> Dict[str, Any]:
        """管理セッションを終了"""
        if not self.current_session:
            raise ValueError("No active session")

        self.current_session.end_time = datetime.now()
        duration = self.current_session.end_time - self.current_session.start_time

        session_summary = {
            "session_id": self.current_session.session_id,
            "session_type": self.current_session.session_type,
            "duration": str(duration),
            "decisions_count": len(self.current_session.decisions_made),
            "actions_count": len(self.current_session.actions_executed),
        }

        logger.info(f"Ended session {self.current_session.session_id}")
        self.current_session = None

        return session_summary

    @conditional_traceable(name="strategic_decision")
    async def make_strategic_decision(self, context: str) -> Dict[str, Any]:
        """戦略的意思決定を行う（model_manager経由）"""
        if not self.current_session:
            raise ValueError("No active session. Start a session first.")

        logger.info("Making strategic decision using model_manager")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": self.current_session.session_id
            if self.current_session
            else "unknown",
            "session_type": self.current_session.session_type
            if self.current_session
            else "unknown",
            "context_length": len(context),
            "context_preview": context[:200] + "..." if len(context) > 200 else context,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "decision_phase": "strategic_planning",
        }

        try:
            # LLMに渡すメッセージを作成
            user_content = f"""
以下のビジネス状況を分析し、戦略的意思決定を行ってください。

【状況】
{context}

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "decision": "決定事項の簡潔な要約",
    "rationale": "決定の根拠と理由",
    "actions": ["具体的なアクション1", "具体的なアクション2"]
}}
```

注意: JSON形式のみで回答し、他のテキストは含めないでください。
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(role="user", content=user_content),
            ]

            # デバッグログ: LLMプロンプト内容を出力（初回のみ）
            if not self._system_prompt_logged:
                logger.debug("=== LLM PROMPT ===")
                logger.debug(
                    f"System Prompt: {self.system_prompt[:500]}..."
                )  # システムプロンプトは長すぎるので一部のみ
                logger.debug(f"User Content: {user_content}")
                logger.debug("=== END PROMPT ===")
                self._system_prompt_logged = True
            else:
                logger.debug("LLM called with established system prompt")

            # model_manager経由でLLM呼び出し (注入されたllm_managerを使用)
            response = await self.llm_manager.generate_response(
                messages, max_tokens=1000
            )

            if not response.success:
                # フォールバックとしてハードコードされた決定を使用
                logger.warning(
                    f"LLM呼び出し失敗 ({response.error_message})、ハードコードされた決定を使用"
                )
                decision = {
                    "context": context,
                    "decision": "在庫水準を維持しつつ、売れ筋商品の価格を最適化する",
                    "rationale": "データ分析の結果、価格調整により利益率5%改善が見込める",
                    "actions": ["価格更新", "在庫補充依頼"],
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # LLMレスポンスをパース
                try:
                    import json

                    # JSON部分を抽出（コードブロックがある場合）
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]  # ```jsonを削除
                    if content.endswith("```"):
                        content = content[:-3]  # ```を削除
                    content = content.strip()

                    # デバッグログ: LLMレスポンス内容を出力
                    logger.debug("=== LLM RESPONSE ===")
                    logger.debug(f"Raw Response: {content}")
                    logger.debug("=== END RESPONSE ===")

                    llm_response = json.loads(content)

                    decision = {
                        "context": context,
                        "decision": llm_response.get(
                            "decision", "決定できませんでした"
                        ),
                        "rationale": llm_response.get("rationale", "理由不明"),
                        "actions": llm_response.get("actions", []),
                        "timestamp": datetime.now().isoformat(),
                        "llm_used": response.model_used,
                    }

                    logger.info(f"LLM意思決定完了: {decision['decision']}")
                    logger.debug(
                        f"LLM意思決定詳細: rationale='{decision['rationale']}', actions={decision.get('actions', [])}"
                    )

                except json.JSONDecodeError as e:
                    logger.error(f"LLMレスポンスのパース失敗: {e}")
                    logger.error(f"LLMレスポンス内容: {response.content}")
                    # フォールバック
                    decision = {
                        "context": context,
                        "decision": "在庫水準を維持しつつ、売れ筋商品の価格を最適化する",
                        "rationale": f"LLMレスポンスのパースに失敗したため、デフォルト決定を使用: {response.content[:200]}",
                        "actions": ["価格更新", "在庫補充依頼"],
                        "timestamp": datetime.now().isoformat(),
                    }

        except Exception as e:
            logger.error(f"戦略的意思決定中にエラー発生: {e}", exc_info=True)
            decision = {
                "context": context,
                "decision": "エラーが発生したため、デフォルト意思決定",
                "rationale": f"システムエラー: {str(e)}",
                "actions": ["管理者への連絡"],
                "timestamp": datetime.now().isoformat(),
            }

        # Execute actions using system data
        executed_actions = []
        for action in decision.get("actions", []):
            if "在庫補充" in action or "補充" in action:
                try:
                    inventory_status = await self.check_inventory_status()
                    low_stock_items = inventory_status.get("low_stock_items", [])
                    if low_stock_items:
                        result = self.assign_restocking_task(
                            low_stock_items, urgency="normal"
                        )
                        executed_actions.append(
                            f"Executed restocking for {low_stock_items}, task_id: {result.get('task_id')}"
                        )
                except Exception as e:
                    logger.error(f"Error executing restocking: {e}")
            elif "価格更新" in action or "価格" in action:
                try:
                    if SAMPLE_PRODUCTS:
                        product = SAMPLE_PRODUCTS[0]  # Using first registered product
                        new_price = round(product.price * 1.05, 0)  # Example adjustment
                        result = self.update_pricing(product.product_id, new_price)
                        executed_actions.append(
                            f"Executed pricing update for {product.product_id} to ¥{new_price}"
                        )
                except Exception as e:
                    logger.error(f"Error executing pricing update: {e}")
            else:
                executed_actions.append(f"Action '{action}' not executable")

        # Log executed actions
        for ea in executed_actions:
            logger.info(f"Executed action: {ea}")

        decision["executed_actions"] = executed_actions

        self.current_session.decisions_made.append(decision)
        return decision

    # Case A node functions (LangGraph Stateful Functions - agent_design.md準拠)

    @conditional_traceable(name="memory_enhanced_inventory_analysis")
    async def inventory_check_node(self, state: ManagementState) -> ManagementState:
        """在庫確認nodeのLangGraph Stateful関数 - LLMベースの在庫分析を実行"""
        logger.info(f"✅ 在庫確認開始: step={state.current_step}")

        # ノード開始時の入力状態を記録（状態変更前に記録）
        input_state_snapshot = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "processing_status": state.processing_status,
            "business_metrics": state.business_metrics.dict()
            if state.business_metrics
            else None,
            "inventory_analysis": state.inventory_analysis,
            "sales_analysis": state.sales_analysis,
            "financial_analysis": state.financial_analysis,
            "executed_actions_count": len(state.executed_actions),
            "errors_count": len(state.errors),
        }

        # トレース用メタデータの準備（入力状態を含む）
        trace_metadata = {
            "input_state": input_state_snapshot,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "inventory_analysis",
            "expected_next_step": "inventory_check",
        }

        try:
            # ステップ更新
            state.current_step = "inventory_check"
            state.processing_status = "processing"

            # ビジネスデータを取得（事前投入されたテストデータを優先）
            if state.business_metrics:
                # テストデータが事前投入されている場合はそれを使用
                metrics = {
                    "sales": state.business_metrics.sales,
                    "profit_margin": state.business_metrics.profit_margin,
                    "inventory_level": state.business_metrics.inventory_level,
                    "customer_satisfaction": state.business_metrics.customer_satisfaction,
                    "timestamp": state.business_metrics.timestamp,
                }
                logger.info("Using pre-loaded test business metrics")
            else:
                # 本番時は実際のシステムから取得
                metrics = self.get_business_metrics()
                state.business_metrics = BusinessMetrics(**metrics)

            # メモリー活用: 過去の在庫分析洞察を取得
            memory_context = self._get_memory_context("inventory_check")

            # LLMベースの在庫分析を実施 (メモリー強化プロンプト)
            inventory_data = metrics.get("inventory_level", {})

            enhanced_prompt = f"""
以下の現在の在庫状況を分析し、在庫管理の総合評価と改善提案を行ってください。

【現在の在庫状況】 (商品名: 数量)
{inventory_data}

【過去の分析洞察】 (参考情報)
{memory_context}

【分析項目】
- 在庫全体の健全性評価 (normal/critical/low)
- 在庫切れリスクのある商品と予想タイミング
- 補充が必要な商品リスト
- 過剰在庫がある可能性のある商品
- 過去の分析結果との整合性確認
- 推奨されるアクション

【分析の考慮点】
- 過去の売上傾向との関連性
- 季節的な需要変動の考慮
- 在庫回転率の改善機会

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "inventory_status": "全体評価",
    "critical_items": ["危機的な商品リスト"],
    "low_stock_items": ["補充優先商品リスト"],
    "reorder_needed": ["発注が必要な商品リスト"],
    "stockout_risks": {{"商品名": "在庫切れ予想タイミング"}},
    "recommended_actions": ["推奨アクションリスト"],
    "analysis": "在庫状況の詳細分析と解説（過去の洞察を踏まえた評価を含む）"
}}
```
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=enhanced_prompt
                ),
            ]

            # LangSmithトレース:
            logger.info("LangSmithトレース: 在庫分析 - memory_contextを利用")

            try:
                response = await self.llm_manager.generate_response(
                    messages, max_tokens=1500
                )

                if response.success:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    analysis_result = json.loads(content)

                    # デフォルト値を設定
                    analysis_result.setdefault("inventory_status", "normal")
                    analysis_result.setdefault("critical_items", [])
                    analysis_result.setdefault("low_stock_items", [])
                    analysis_result.setdefault("reorder_needed", [])
                    analysis_result.setdefault("stockout_risks", {})
                    analysis_result.setdefault("recommended_actions", ["在庫状況確認"])
                    analysis_result.setdefault("analysis", "LLMによる在庫分析実施")

                    logger.info(
                        f"LLM在庫分析成功: {analysis_result['inventory_status']}, メモリー統合準備完了"
                    )

                    # メモリー保存: 在庫分析結果から洞察を抽出し保存
                    llm_response_raw = json.dumps(
                        analysis_result
                    )  # レスポンス全体を渡す
                    self._extract_and_save_business_insight(
                        "inventory_check", llm_response_raw
                    )
                else:
                    # LLM失敗時のフォールバック
                    logger.warning(f"LLM在庫分析失敗: {response.error_message}")
                    analysis_result = {
                        "inventory_status": "normal",
                        "critical_items": [],
                        "low_stock_items": ["water"] if len(inventory_data) > 0 else [],
                        "reorder_needed": ["water"] if len(inventory_data) > 0 else [],
                        "stockout_risks": {"water": "数日後"}
                        if len(inventory_data) > 0
                        else {},
                        "recommended_actions": ["在庫状況確認"],
                        "analysis": "LLM分析不可、本番データ待機",
                    }
            except Exception as e:
                logger.error(f"在庫分析エラー: {e}")
                # 完全なフォールバック
                total_products = len(inventory_data)
                analysis_result = {
                    "inventory_status": "normal",
                    "critical_items": [],
                    "low_stock_items": list(inventory_data.keys())[
                        : min(3, total_products)
                    ]
                    if total_products > 0
                    else [],
                    "reorder_needed": list(inventory_data.keys())[
                        : min(3, total_products)
                    ]
                    if total_products > 0
                    else [],
                    "stockout_risks": {list(inventory_data.keys())[0]: "1週間後"}
                    if total_products > 0
                    else {},
                    "recommended_actions": ["在庫状況確認"],
                    "analysis": f"分析エラー: {str(e)}",
                }

            # State更新
            state.inventory_analysis = {
                "status": analysis_result.get("inventory_status", "unknown"),
                "low_stock_items": analysis_result.get("low_stock_items", []),
                "reorder_needed": analysis_result.get("reorder_needed", []),
                "estimated_stockout": analysis_result.get("stockout_risks", {}),
                "critical_items": analysis_result.get("critical_items", []),
                "recommended_actions": analysis_result.get("recommended_actions", []),
                "llm_analysis": analysis_result.get("analysis", ""),
                "analysis_timestamp": datetime.now().isoformat(),
            }

            # ログ出力
            total_low = len(state.inventory_analysis.get("low_stock_items", [])) + len(
                state.inventory_analysis.get("critical_items", [])
            )
            logger.info(
                f"✅ 在庫確認完了: 分析項目={total_low}, ステータス={analysis_result['inventory_status']}"
            )

        except Exception as e:
            logger.error(f"Stateful在庫確認エラー: {e}")
            state.errors.append(f"inventory_check: {str(e)}")
            state.processing_status = "error"

        return state

    @conditional_traceable(name="memory_enhanced_sales_plan_analysis")
    async def sales_plan_node(self, state: ManagementState) -> ManagementState:
        """売上計画nodeのLangGraph Stateful関数 - 財務・売上分析を実行"""
        logger.info(f"✅ 売上計画開始: step={state.current_step}")

        # ノード開始時の入力状態を記録（状態変更前に記録）
        input_state_snapshot = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "processing_status": state.processing_status,
            "business_metrics": state.business_metrics.dict()
            if state.business_metrics
            else None,
            "inventory_analysis": state.inventory_analysis,
            "executed_actions_count": len(state.executed_actions),
            "errors_count": len(state.errors),
        }

        # トレース用メタデータの準備（入力状態を含む）
        trace_metadata = {
            "input_state": input_state_snapshot,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "sales_plan_analysis",
            "expected_next_step": "sales_plan",
        }

        try:
            # ステップ更新
            state.current_step = "sales_plan"
            state.processing_status = "processing"

            # ビジネスデータを取得
            metrics = self.get_business_metrics()
            if not state.business_metrics:
                state.business_metrics = BusinessMetrics(**metrics)

            # LLMベースの売上・財務分析を実施
            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user",
                    content=f"""
以下のビジネスメトリクスを分析し、売上パフォーマンス評価と戦略的推奨を行ってください。

【現在のビジネスメトリクス】
- 売上: ¥{metrics.get("sales", 0):,}
- 利益率: {metrics.get("profit_margin", 0):.1%}
- 顧客満足度: {metrics.get("customer_satisfaction", 3.0)}/5.0

【分析項目】
- 売上トレンドの評価 (positive/stable/concerning)
- パフォーマンスの詳細評価
- 推奨される戦略的アクション
- 期待される改善効果とタイムライン

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "sales_trend": "売上トレンド評価",
    "sales_performance": "パフォーマンスの詳細説明",
    "financial_analysis": {{
        "sales": "売上数値",
        "profit_margin": "利益率数値",
        "customer_satisfaction": "顧客満足度数値",
        "analysis_timestamp": "分析時刻"
    }},
    "strategies": ["戦略アクション1", "戦略アクション2", "戦略アクション3"],
    "expected_impact": "改善効果の全体評価",
    "timeline": "実施タイムライン",
    "analysis": "総合的な分析と解説（100文字以上）"
}}
```
""",
                ),
            ]

            try:
                response = await self.llm_manager.generate_response(
                    messages, max_tokens=1500
                )

                if response.success:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    analysis_result = json.loads(content)

                    # デフォルト値の設定
                    analysis_result.setdefault("sales_trend", "unknown")
                    analysis_result.setdefault("sales_performance", "分析未完了")
                    analysis_result.setdefault(
                        "financial_analysis",
                        {
                            "sales": metrics.get("sales", 0),
                            "profit_margin": metrics.get("profit_margin", 0),
                            "customer_satisfaction": metrics.get(
                                "customer_satisfaction", 3.0
                            ),
                            "analysis_timestamp": datetime.now().isoformat(),
                        },
                    )
                    analysis_result.setdefault("strategies", [])
                    analysis_result.setdefault("expected_impact", "分析未完了")
                    analysis_result.setdefault("timeline", "未設定")
                    analysis_result.setdefault(
                        "analysis", "LLMによる売上・財務分析実施"
                    )

                    strategies = analysis_result["strategies"]
                    sales_trend = analysis_result["sales_trend"]
                    financial_analysis_result = analysis_result["financial_analysis"]

                    logger.info(f"LLM売上計画分析成功: trend={sales_trend}")
                else:
                    # LLM失敗時のフォールバック
                    logger.warning(f"LLM売上計画分析失敗: {response.error_message}")
                    analysis_result = {
                        "sales_trend": "unknown",
                        "sales_performance": "LLM分析不可、本番データ待機",
                        "financial_analysis": {
                            "sales": metrics.get("sales", 0),
                            "profit_margin": metrics.get("profit_margin", 0),
                            "customer_satisfaction": metrics.get(
                                "customer_satisfaction", 3.0
                            ),
                            "analysis_timestamp": datetime.now().isoformat(),
                        },
                        "strategies": ["基本戦略検討"],
                        "expected_impact": "分析未完了",
                        "timeline": "未設定",
                        "analysis": f"LLM分析エラー: {response.error_message}",
                    }
                    strategies = analysis_result["strategies"]
                    sales_trend = analysis_result["sales_trend"]
                    financial_analysis_result = analysis_result["financial_analysis"]
            except Exception as e:
                logger.error(f"売上計画分析エラー: {e}")
                # 完全なフォールバック
                sales_trend = "unknown"
                strategies = ["基本戦略検討"]
                financial_analysis_result = {
                    "sales": metrics.get("sales", 0),
                    "profit_margin": metrics.get("profit_margin", 0),
                    "customer_satisfaction": metrics.get("customer_satisfaction", 3.0),
                    "analysis_timestamp": datetime.now().isoformat(),
                }
                analysis_result = {
                    "sales_performance": f"分析エラー: {str(e)}",
                    "expected_impact": "分析失敗",
                    "timeline": "未設定",
                    "analysis": f"エラー分析: {str(e)}",
                }

            # State更新
            state.sales_analysis = {
                "financial_overview": f"{metrics.get('profit_margin', 0):.1%}利益率・売上{metrics.get('sales', 0):,.0f}",
                "sales_trend": sales_trend,
                "profit_analysis": financial_analysis_result,
                "strategies": strategies,
                "action_plan": [f"戦略: {s}" for s in strategies],
                "expected_impact": f"{len(strategies)}個の改善施策を実施",
                "timeline": "次回の経営会議で実施",
                "analysis_timestamp": datetime.now().isoformat(),
            }

            state.financial_analysis = financial_analysis_result

            # ログ出力
            logger.info(
                f"✅ 売上計画完了: trend={sales_trend}, strategies={len(strategies)}"
            )

        except Exception as e:
            logger.error(f"Stateful売上計画エラー: {e}")
            state.errors.append(f"sales_plan: {str(e)}")
            state.processing_status = "error"

        return state

    async def pricing_node(self, state: ManagementState) -> ManagementState:
        """価格戦略決定nodeのLangGraph Stateful関数 - LLMベースの価格決定を実行"""
        logger.info(f"✅ 価格戦略開始: step={state.current_step}")

        # ノード開始時の入力状態を記録（状態変更前に記録）
        input_state_snapshot = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "processing_status": state.processing_status,
            "business_metrics": state.business_metrics.dict()
            if state.business_metrics
            else None,
            "sales_analysis": state.sales_analysis,
            "financial_analysis": state.financial_analysis,
            "inventory_analysis": state.inventory_analysis,
            "executed_actions_count": len(state.executed_actions),
            "errors_count": len(state.errors),
        }

        # トレース用メタデータの準備（入力状態を含む）
        trace_metadata = {
            "input_state": input_state_snapshot,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "pricing_decision",
            "expected_next_step": "pricing",
        }

        try:
            # ステップ更新
            state.current_step = "pricing"
            state.processing_status = "processing"

            # 前提分析データを取得
            sales_analysis = state.sales_analysis
            financial_analysis = state.financial_analysis
            inventory_analysis = state.inventory_analysis

            if not sales_analysis or not financial_analysis:
                logger.warning("前提分析データがありません")
                state.errors.append("pricing: 前提分析データなし")
                state.processing_status = "error"
                return state

            # LLMベースの価格戦略決定を実施
            pricing_context = f"""
以下のビジネス状況を分析し、価格戦略を決定してください。

【売上・財務分析結果】
- 売上トレンド: {sales_analysis.get("sales_trend", "unknown")}
- 財務分析: {financial_analysis.get("analysis", "なし")}
- 戦略提案: {sales_analysis.get("strategies", [])}

【現在の財務状況】
- 売上: ¥{financial_analysis.get("sales", 0):,}
- 利益率: {financial_analysis.get("profit_margin", 0):.1%}
- 顧客満足度: {financial_analysis.get("customer_satisfaction", 3.0)}/5.0

【在庫状況（参考）】
- 在庫ステータス: {inventory_analysis.get("status", "unknown") if inventory_analysis else "なし"}
- 危機的商品: {inventory_analysis.get("critical_items", []) if inventory_analysis else []}
- 補充優先商品: {inventory_analysis.get("low_stock_items", []) if inventory_analysis else []}

【価格決定の考慮点】
1. 売上トレンドと財務状況に基づく価格戦略
2. 在庫状況と商品の需要バランス
3. 顧客満足度への影響
4. 競争力の維持
5. 利益率の最適化

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "pricing_strategy": "価格戦略の種類 (increase/decrease/maintain/mixed)",
    "reasoning": "価格決定の詳細な理由",
    "product_updates": [
        {{
            "product_name": "商品名",
            "current_price": 基準価格,
            "new_price": 新価格,
            "price_change_percent": 価格変更率,
            "reason": "当該商品の価格変更理由"
        }}
    ],
    "expected_impact": "戦略実行による期待効果",
    "risk_assessment": "リスク評価と対策",
    "analysis": "総合的な分析と解説（100文字以上）"
}}
```
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=pricing_context
                ),
            ]

            logger.info("LLM価格戦略分析開始 - 前工程データ統合")

            try:
                # 非同期関数なので直接awaitを使用
                response = await self.llm_manager.generate_response(
                    messages, max_tokens=1500
                )

                if response.success:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    pricing_result = json.loads(content)

                    # デフォルト値の設定
                    pricing_result.setdefault("pricing_strategy", "maintain")
                    pricing_result.setdefault("reasoning", "分析結果に基づく価格戦略")
                    pricing_result.setdefault("product_updates", [])
                    pricing_result.setdefault(
                        "expected_impact", "価格戦略による影響評価"
                    )
                    pricing_result.setdefault("risk_assessment", "リスク評価なし")
                    pricing_result.setdefault("analysis", "LLMによる価格戦略分析実施")

                    logger.info(
                        f"LLM価格戦略分析成功: strategy={pricing_result['pricing_strategy']}"
                    )

                    # LLM分析結果をログ出力
                    logger.info("=== LLM Pricing Strategy Analysis ===")
                    logger.info(f"Strategy: {pricing_result['pricing_strategy']}")
                    logger.info(f"Reasoning: {pricing_result['reasoning']}")
                    logger.info(
                        f"Product Updates: {len(pricing_result['product_updates'])}"
                    )
                    logger.info(f"Expected Impact: {pricing_result['expected_impact']}")

                else:
                    # LLM失敗時のフォールバック
                    logger.warning(f"LLM価格戦略分析失敗: {response.error_message}")
                    pricing_result = {
                        "pricing_strategy": "maintain",
                        "reasoning": "LLM分析不可のため安定維持を選択",
                        "product_updates": [],
                        "expected_impact": "安定重視",
                        "risk_assessment": "リスク回避優先",
                        "analysis": f"LLM分析エラー: {response.error_message}",
                    }

            except Exception as e:
                logger.error(f"価格戦略分析エラー: {e}")
                # 完全なフォールバック
                pricing_result = {
                    "pricing_strategy": "maintain",
                    "reasoning": f"分析エラー: {str(e)}",
                    "product_updates": [],
                    "expected_impact": "安定重視",
                    "risk_assessment": "リスク回避優先",
                    "analysis": f"価格戦略分析エラー: {str(e)}",
                }

            # 価格更新の実行（LLM結果に基づく）
            executed_updates = []

            if pricing_result["product_updates"]:
                for update in pricing_result["product_updates"]:
                    try:
                        product_name = update.get("product_name", "unknown")
                        new_price = update.get("new_price", 150)

                        # ツール呼び出しでシステム反映
                        update_result = self.update_pricing(product_name, new_price)
                        logger.info(
                            f"ツール update_pricing 呼び出し成功: {product_name} -> ¥{new_price}"
                        )

                        # 実行アクション記録
                        action = {
                            "type": "pricing_update",
                            "product_name": product_name,
                            "new_price": new_price,
                            "price_change_percent": update.get(
                                "price_change_percent", 0
                            ),
                            "reason": update.get("reason", pricing_result["reasoning"]),
                            "tool_called": "update_pricing",
                            "tool_result": update_result,
                            "timestamp": datetime.now().isoformat(),
                        }
                        state.executed_actions.append(action)
                        executed_updates.append(update)

                    except Exception as e:
                        logger.error(f"価格更新ツール呼び出し失敗 {product_name}: {e}")
                        # エラー時も記録
                        action = {
                            "type": "pricing_update_error",
                            "product_name": product_name,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                        state.executed_actions.append(action)

            # State更新
            state.pricing_decision = {
                "strategy": pricing_result["pricing_strategy"],
                "reasoning": pricing_result["reasoning"],
                "product_updates": executed_updates,
                "expected_impact": pricing_result["expected_impact"],
                "risk_assessment": pricing_result["risk_assessment"],
                "llm_analysis": pricing_result["analysis"],
                "analysis_timestamp": datetime.now().isoformat(),
            }

            # ログ出力
            logger.info(
                f"✅ Stateful価格戦略完了: strategy={pricing_result['pricing_strategy']}, updates={len(executed_updates)}"
            )

        except Exception as e:
            logger.error(f"Stateful価格戦略エラー: {e}")
            state.errors.append(f"pricing: {str(e)}")
            state.processing_status = "error"

        return state

    @traceable(name="restock_tasks_llm")
    async def restock_node(self, state: ManagementState) -> ManagementState:
        """在庫補充タスク割り当てnodeのLangGraph Stateful関数 - LLM常時使用：補充戦略分析＆実現可能アクション決定"""
        logger.info(f"✅ Stateful補充タスク開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "restock_tasks_llm",
            "input_state": {
                "has_inventory_analysis": state.inventory_analysis is not None,
                "low_stock_items_count": len(
                    state.inventory_analysis.get("low_stock_items", [])
                )
                if state.inventory_analysis
                else 0,
                "critical_items_count": len(
                    state.inventory_analysis.get("critical_items", [])
                )
                if state.inventory_analysis
                else 0,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "restock"
            state.processing_status = "processing"

            # 前提分析を取得
            inventory_analysis = state.inventory_analysis
            if not inventory_analysis:
                logger.warning("在庫分析データがありません")
                state.errors.append("restock: 在庫分析データなし")
                state.processing_status = "error"
                return state

            # LLM常時使用：補充戦略の詳細分析＆実現可能アクション決定
            restock_context = f"""
以下の在庫状況を分析し、自動販売機経営における実現可能な補充戦略を決定してください。

【在庫分析結果】 (参照情報)
- 在庫ステータス: {inventory_analysis.get("status", "unknown")}
- 危機的商品: {inventory_analysis.get("critical_items", [])}
- 在庫不足商品: {inventory_analysis.get("low_stock_items", [])}
- 在庫切れリスク: {inventory_analysis.get("stockout_risks", {})}
- 再発注推奨商品: {inventory_analysis.get("reorder_needed", [])}
- 在庫分析LLM結果: {inventory_analysis.get("llm_analysis", "なし")}

【現在の事業状況】 (自動販売機運営制約考慮)
- 営業時間: 24時間対応の制約 (従業員訪問は制限される可能性)
- 補充リソース: 従業員による手動補充作業
- 緊急対応: 在庫切れ時の営業停止回避を最優先
- コスト制約: 過剰補充による運用コスト増加の回避

【補充戦略の考慮点】
1. 危機的商品に対する緊急補充アクション
2. 通常補充のリソース効率性
3. 自動販売機固有の運営効率 (訪問頻度最小化)
4. 季節性需要変動の考慮
5. 在庫回転率の最適化

【出力形式】
JSON形式で以下の構造で回答してください：
{{
    "restock_strategy": "全体補充戦略 (emergency_response/regular_maintenance/optimized_rotating)",
    "action_plan": {{
        "immediate_actions": ["即時実行アクション（危機的商品対応）"],
        "scheduled_actions": ["計画的アクション（通常補充）"],
        "preventive_actions": ["予防的アクション（将来リスク回避）"]
    }},
    "resource_allocation": {{
        "urgent_tasks": ["緊急度高の補充タスク（担当者即時割り当て）"],
        "normal_tasks": ["通常補充タスク（通常業務と並行）"],
        "long_term_tasks": ["長期検討タスク（次の戦略会議まで）"]
    }},
    "efficiency_considerations": {{
        "visit_optimization": "自動販売機配置別補充効率化策",
        "cost_benefit": "補充コストと営業機会損失のバランス",
        "contingency_plans": ["緊急時対応策（災害・大量消費時）"]
    }},
    "expected_outcomes": ["補充実行による期待効果とKPI改善"],
    "analysis": "総合的な補充戦略分析と自動販売機経営への影響評価（100文字以上）"
}}
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=restock_context
                ),
            ]

            logger.info("LLM補充戦略分析開始 - 自動販売機運営制約統合")

            try:
                # 非同期関数なので直接awaitを使用
                llm_response = await self.llm_manager.generate_response(
                    messages, max_tokens=1500
                )

                if llm_response.success:
                    import json

                    content = llm_response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    restock_strategy = json.loads(content)

                    # デフォルト値の設定
                    restock_strategy.setdefault(
                        "restock_strategy", "regular_maintenance"
                    )
                    restock_strategy.setdefault(
                        "action_plan",
                        {
                            "immediate_actions": [],
                            "scheduled_actions": [],
                            "preventive_actions": [],
                        },
                    )
                    restock_strategy.setdefault(
                        "resource_allocation",
                        {"urgent_tasks": [], "normal_tasks": [], "long_term_tasks": []},
                    )
                    restock_strategy.setdefault(
                        "efficiency_considerations",
                        {
                            "visit_optimization": "未考慮",
                            "cost_benefit": "バランス検討",
                            "contingency_plans": [],
                        },
                    )
                    restock_strategy.setdefault("expected_outcomes", ["在庫安定化"])
                    restock_strategy.setdefault("analysis", "LLMによる補充戦略分析実施")

                    logger.info(
                        f"LLM補充戦略分析成功: strategy={restock_strategy['restock_strategy']}, llm_used=True"
                    )

                    # LLM分析結果をログ出力
                    logger.info("=== LLM Restock Strategy Analysis ===")
                    logger.info(f"Strategy: {restock_strategy['restock_strategy']}")
                    logger.info(
                        f"Immediate Actions: {len(restock_strategy['action_plan']['immediate_actions'])}"
                    )
                    logger.info(
                        f"Urgent Tasks: {len(restock_strategy['resource_allocation']['urgent_tasks'])}"
                    )
                    logger.info(f"Analysis: {restock_strategy['analysis'][:100]}...")

                else:
                    # LLM失敗時のフォールバック
                    logger.warning(f"LLM補充戦略分析失敗: {llm_response.error_message}")
                    restock_strategy = {
                        "restock_strategy": "regular_maintenance",
                        "action_plan": {
                            "immediate_actions": [],
                            "scheduled_actions": ["通常補充作業実施"],
                            "preventive_actions": ["在庫監視強化"],
                        },
                        "resource_allocation": {
                            "urgent_tasks": [],
                            "normal_tasks": ["通常補充タスク"],
                            "long_term_tasks": ["補充スケジュール最適化"],
                        },
                        "efficiency_considerations": {
                            "visit_optimization": "標準訪問頻度維持",
                            "cost_benefit": "コスト削減優先",
                            "contingency_plans": ["緊急補充計画策定"],
                        },
                        "expected_outcomes": ["在庫安定化"],
                        "analysis": f"LLM分析エラー: {llm_response.error_message}",
                    }

            except Exception as e:
                logger.error(f"補充戦略分析エラー: {e}")
                # 完全なフォールバック
                restock_strategy = {
                    "restock_strategy": "regular_maintenance",
                    "action_plan": {
                        "immediate_actions": [],
                        "scheduled_actions": ["通常補充作業"],
                        "preventive_actions": [],
                    },
                    "resource_allocation": {
                        "urgent_tasks": [],
                        "normal_tasks": ["補充タスク実行"],
                        "long_term_tasks": [],
                    },
                    "efficiency_considerations": {
                        "visit_optimization": "標準業務フロー",
                        "cost_benefit": "バランス考慮",
                        "contingency_plans": [],
                    },
                    "expected_outcomes": ["在庫維持"],
                    "analysis": f"LLM分析エラー: {str(e)}",
                }

            # 補充タスク決定 (LLM戦略に基づく)
            restock_decision = {
                "action": "tasks_assigned"
                if restock_strategy["resource_allocation"]["urgent_tasks"]
                else "strategic_planning",
                "reasoning": f"LLM補充戦略分析に基づくタスク割り当て: {restock_strategy['restock_strategy']}",
                "strategy": restock_strategy["restock_strategy"],
                "action_plan": restock_strategy["action_plan"],
                "resource_allocation": restock_strategy["resource_allocation"],
                "efficiency_considerations": restock_strategy[
                    "efficiency_considerations"
                ],
                "expected_outcomes": restock_strategy["expected_outcomes"],
                "llm_analysis": restock_strategy["analysis"],
                "analysis_timestamp": datetime.now().isoformat(),
                "tasks_assigned": [],
                "total_items": 0,
            }

            # 在庫情報から具体的な補充タスクを組み立て
            inventory_analysis = state.inventory_analysis
            low_stock_items = inventory_analysis.get("low_stock_items", [])
            critical_items = inventory_analysis.get("critical_items", [])

            # LLM戦略に基づき補充タスクを実行
            all_tasks = []
            urgent_products = []
            normal_products = []

            # 緊急タスク割り当て
            urgent_tasks = restock_strategy["resource_allocation"]["urgent_tasks"]
            if urgent_tasks:
                for urgent_task in urgent_tasks:
                    # タスクから商品名を抽出
                    if critical_items:
                        urgent_products.extend(critical_items)
                    elif low_stock_items:
                        urgent_products.extend(low_stock_items[: len(urgent_tasks)])

            # 通常タスク割り当て
            normal_tasks = restock_strategy["resource_allocation"]["normal_tasks"]
            if normal_tasks:
                for normal_task in normal_tasks:
                    remaining_low_stock = [
                        item for item in low_stock_items if item not in urgent_products
                    ]
                    normal_products.extend(remaining_low_stock)

            # 重複除去
            urgent_products = list(set(urgent_products))
            normal_products = list(set(normal_products) - set(urgent_products))

            # 具体的なタスク実行
            for product in urgent_products + normal_products:
                urgency = "urgent" if product in urgent_products else "normal"
                task = self.assign_restocking_task([product], urgency)
                task_info = {
                    "product": product,
                    "task_id": task.get("task_id"),
                    "urgency": urgency,
                    "deadline": task.get("deadline"),
                    "strategy_driven": True,  # LLM戦略によるタスク
                }
                all_tasks.append(task_info)

            restock_decision["tasks_assigned"] = all_tasks
            restock_decision["total_items"] = len(all_tasks)

            # 実行アクション記録
            if all_tasks:
                for task in all_tasks:
                    action = {
                        "type": "restock_task_llm",
                        "product": task["product"],
                        "task_id": task["task_id"],
                        "urgency": task["urgency"],
                        "strategy_driven": task["strategy_driven"],
                        "llm_strategy": restock_strategy["restock_strategy"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    state.executed_actions.append(action)

            # State更新
            state.restock_decision = restock_decision

            # ログ出力
            tasks_count = len(all_tasks)
            strategy = restock_strategy["restock_strategy"]
            logger.info(
                f"✅ Stateful補充タスク完了: tasks={tasks_count}, strategy={strategy}, llm_used=True"
            )

        except Exception as e:
            logger.error(f"Stateful補充タスクエラー: {e}")
            state.errors.append(f"restock: {str(e)}")
            state.processing_status = "error"

        return state

    @traceable(name="procurement_requests_llm")
    async def procurement_request_generation_node(
        self, state: ManagementState
    ) -> ManagementState:
        """発注依頼nodeのLangGraph Stateful関数 - LLM常時使用：発注最適化戦略分析＆実現可能発注決定"""
        logger.info(f"✅ Stateful発注依頼開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "procurement_requests_llm",
            "input_state": {
                "has_inventory_analysis": state.inventory_analysis is not None,
                "has_restock_decision": state.restock_decision is not None,
                "reorder_items_count": len(
                    state.inventory_analysis.get("reorder_needed", [])
                )
                if state.inventory_analysis
                else 0,
                "assigned_tasks_count": len(
                    state.restock_decision.get("tasks_assigned", [])
                )
                if state.restock_decision
                else 0,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "procurement"
            state.processing_status = "processing"

            # 前提分析を取得
            inventory_analysis = state.inventory_analysis
            restock_decision = state.restock_decision

            if not inventory_analysis or not restock_decision:
                logger.warning("前提データがありません")
                state.errors.append("procurement: 前提データなし")
                state.processing_status = "error"
                return state

            # LLM常時使用：発注最適化戦略の詳細分析＆実現可能発注決定
            procurement_context = f"""
以下の補充タスクと在庫状況を分析し、自動販売機経営における実現可能な発注最適化戦略を決定してください。

【補充タスク状況】 (参照情報)
- 補充戦略: {restock_decision.get("strategy", "unknown")}
- 補充LLM分析: {restock_decision.get("llm_analysis", "なし")}
- 割り当てタスク数: {len(restock_decision.get("tasks_assigned", []))}
- 緊急タスク: {len([t for t in restock_decision.get("tasks_assigned", []) if t.get("urgency") == "urgent"])}
- 通常タスク: {len([t for t in restock_decision.get("tasks_assigned", []) if t.get("urgency") != "urgent"])}

【在庫分析状況】 (参照情報)
- 再発注推奨商品: {inventory_analysis.get("reorder_needed", [])}
- 危機的商品: {inventory_analysis.get("critical_items", [])}
- 在庫不足商品: {inventory_analysis.get("low_stock_items", [])}
- 在庫ステータス: {inventory_analysis.get("status", "unknown")}
- 在庫分析LLM結果: {inventory_analysis.get("llm_analysis", "なし")}

【現在の事業状況】 (自動販売機運営制約考慮)
- 仕入先選定: 信頼性・価格・納期のバランスを考慮
- 資金繰り制約: 過剰発注による資金流動性悪化を回避
- 在庫保管: 自動販売機容量の制限 (約50スロット×商品)
- 納期管理: 緊急時対応 vs 定期発注の棲み分け
- コスト最適化: 調達コスト vs 欠品機会損失のトレードオフ

【発注戦略の考慮点】
1. 補充タスクの優先順位付けと発注タイミング
2. 仕入先ポートフォリオの多様化リスク分散
3. 発注ロット最適化 (経済発注量 vs 即時性)
4. 納期シナリオのリアルな想定（通常3営業日）
5. 季節変動・需要予測の取り込み
6. 競争力確保のための予備在庫戦略

【出力形式】
JSON形式で以下の構造で回答してください：
{{
    "procurement_strategy": "全体発注戦略 (emergency_procurement/standard_procurement/optimized_batching/supplier_diversification)",
    "supplier_allocation": {{
        "primary_supplier": ["信頼性重視商品（安定供給優先）"],
        "alternative_suppliers": ["価格競争力重視商品（コスト削減優先）"],
        "emergency_suppliers": ["即日対応可能商品（危機的発注専用）"]
    }},
    "order_optimization": {{
        "consolidated_orders": ["発注統合商品（ロット効率化）"],
        "urgent_orders": ["緊急発注商品（即時納入優先）"],
        "scheduled_orders": ["計画発注商品（安価ルート利用）"]
    }},
    "cost_benefit_analysis": {{
        "immediate_costs": "発注実行コストの見積もり",
        "expected_savings": "最適化による削減効果",
        "risk_mitigation": "欠品・過剰在庫リスク評価と対策",
        "roi_expectations": "投資回収期間とROI予測"
    }},
    "delivery_timeline": {{
        "emergency_delivery": ["24-48時間以内の商品"],
        "standard_delivery": ["3-5営業日以内の商品"],
        "bulk_delivery": ["1-2週間程度の計画発注商品"]
    }},
    "contingency_plans": ["緊急時対応策と代替調達ルート"],
    "expected_outcomes": ["発注実行による期待効果と事業KPI改善"],
    "analysis": "総合的な発注戦略分析と自動販売機経営への影響評価（100文字以上）"
}}
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system", content=self.system_prompt
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=procurement_context
                ),
            ]

            logger.info("LLM発注戦略分析開始 - 自動販売機調達制約統合")

            try:
                # 非同期関数なので直接awaitを使用
                llm_response = await self.llm_manager.generate_response(
                    messages, max_tokens=1600
                )

                if llm_response.success:
                    import json

                    content = llm_response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    procurement_strategy = json.loads(content)

                    # デフォルト値の設定
                    procurement_strategy.setdefault(
                        "procurement_strategy", "standard_procurement"
                    )
                    procurement_strategy.setdefault(
                        "supplier_allocation",
                        {
                            "primary_supplier": [],
                            "alternative_suppliers": [],
                            "emergency_suppliers": [],
                        },
                    )
                    procurement_strategy.setdefault(
                        "order_optimization",
                        {
                            "consolidated_orders": [],
                            "urgent_orders": [],
                            "scheduled_orders": [],
                        },
                    )
                    procurement_strategy.setdefault(
                        "cost_benefit_analysis",
                        {
                            "immediate_costs": "計算中",
                            "expected_savings": "分析中",
                            "risk_mitigation": "評価中",
                            "roi_expectations": "計算中",
                        },
                    )
                    procurement_strategy.setdefault(
                        "delivery_timeline",
                        {
                            "emergency_delivery": [],
                            "standard_delivery": [],
                            "bulk_delivery": [],
                        },
                    )
                    procurement_strategy.setdefault("contingency_plans", [])
                    procurement_strategy.setdefault("expected_outcomes", ["発注安定化"])
                    procurement_strategy.setdefault(
                        "analysis", "LLMによる発注戦略分析実施"
                    )

                    logger.info(
                        f"LLM発注戦略分析成功: strategy={procurement_strategy['procurement_strategy']}, llm_used=True"
                    )

                    # LLM分析結果をログ出力
                    logger.info("=== LLM Procurement Strategy Analysis ===")
                    logger.info(
                        f"Strategy: {procurement_strategy['procurement_strategy']}"
                    )
                    logger.info(
                        f"Urgent Orders: {len(procurement_strategy['order_optimization']['urgent_orders'])}"
                    )
                    logger.info(
                        f"Consolidated Orders: {len(procurement_strategy['order_optimization']['consolidated_orders'])}"
                    )
                    logger.info(
                        f"Analysis: {procurement_strategy['analysis'][:100]}..."
                    )

                else:
                    # LLM失敗時のフォールバック
                    logger.warning(f"LLM発注戦略分析失敗: {llm_response.error_message}")
                    procurement_strategy = {
                        "procurement_strategy": "standard_procurement",
                        "supplier_allocation": {
                            "primary_supplier": [],
                            "alternative_suppliers": [],
                            "emergency_suppliers": [],
                        },
                        "order_optimization": {
                            "consolidated_orders": [],
                            "urgent_orders": [],
                            "scheduled_orders": ["標準発注商品"],
                        },
                        "cost_benefit_analysis": {
                            "immediate_costs": "標準配送料",
                            "expected_savings": "ロット効果",
                            "risk_mitigation": "分散発注",
                            "roi_expectations": "3ヶ月以内",
                        },
                        "delivery_timeline": {
                            "emergency_delivery": [],
                            "standard_delivery": ["通常商品"],
                            "bulk_delivery": [],
                        },
                        "contingency_plans": ["代替発注ルート確保"],
                        "expected_outcomes": ["発注安定化"],
                        "analysis": f"LLM分析エラー: {llm_response.error_message}",
                    }

            except Exception as e:
                logger.error(f"発注戦略分析エラー: {e}")
                # 完全なフォールバック
                procurement_strategy = {
                    "procurement_strategy": "standard_procurement",
                    "supplier_allocation": {
                        "primary_supplier": [],
                        "alternative_suppliers": [],
                        "emergency_suppliers": [],
                    },
                    "order_optimization": {
                        "consolidated_orders": [],
                        "urgent_orders": [],
                        "scheduled_orders": ["全部商品"],
                    },
                    "cost_benefit_analysis": {
                        "immediate_costs": "標準コスト",
                        "expected_savings": "ロット割引",
                        "risk_mitigation": "通常レベル",
                        "roi_expectations": "標準期間",
                    },
                    "delivery_timeline": {
                        "emergency_delivery": [],
                        "standard_delivery": ["全部商品"],
                        "bulk_delivery": [],
                    },
                    "contingency_plans": ["標準対応"],
                    "expected_outcomes": ["発注実行"],
                    "analysis": f"LLM分析エラー: {str(e)}",
                }

            # 発注判定と実行 (LLM戦略に基づく)
            procurement_decision = {
                "action": "strategic_procurement"
                if procurement_strategy["order_optimization"]["urgent_orders"]
                else "optimized_procurement",
                "reasoning": f"LLM発注戦略分析に基づく調達実行: {procurement_strategy['procurement_strategy']}",
                "strategy": procurement_strategy["procurement_strategy"],
                "supplier_allocation": procurement_strategy["supplier_allocation"],
                "order_optimization": procurement_strategy["order_optimization"],
                "cost_benefit_analysis": procurement_strategy["cost_benefit_analysis"],
                "delivery_timeline": procurement_strategy["delivery_timeline"],
                "contingency_plans": procurement_strategy["contingency_plans"],
                "expected_outcomes": procurement_strategy["expected_outcomes"],
                "llm_analysis": procurement_strategy["analysis"],
                "analysis_timestamp": datetime.now().isoformat(),
                "orders_placed": [],
                "total_orders": 0,
            }

            # 在庫分析と補充決定から具体的な発注商品を決定
            reorder_needed = inventory_analysis.get("reorder_needed", [])
            tasks_assigned = restock_decision.get("tasks_assigned", [])

            # LLM戦略に基づき発注対象を分類・最適化
            all_orders = []
            urgent_products = procurement_strategy["order_optimization"][
                "urgent_orders"
            ]
            consolidated_products = procurement_strategy["order_optimization"][
                "consolidated_orders"
            ]
            scheduled_products = procurement_strategy["order_optimization"][
                "scheduled_orders"
            ]

            # 発注対象の優先順位付け
            for task in tasks_assigned:
                product = task.get("product")
                if product in reorder_needed:
                    # LLM戦略による発注最適化
                    if task.get("urgency") == "urgent" or product in urgent_products:
                        order_quantity = 15  # 緊急発注:少量・高頻度
                        delivery_priority = "emergency"
                    elif product in consolidated_products:
                        order_quantity = 30  # 統合発注:大量・割安
                        delivery_priority = "bulk"
                    elif product in scheduled_products:
                        order_quantity = 25  # 計画発注:標準量
                        delivery_priority = "standard"
                    else:
                        # デフォルト戦略
                        order_quantity = 20
                        delivery_priority = "standard"

                    # 発注実行
                    procurement_result = self.request_procurement(
                        [product],
                        {product: order_quantity},
                    )

                    order_info = {
                        "product": product,
                        "quantity": order_quantity,
                        "order_id": procurement_result.get("order_id"),
                        "estimated_delivery": procurement_result.get(
                            "estimated_delivery"
                        ),
                        "urgency": task.get("urgency", "normal"),
                        "delivery_priority": delivery_priority,
                        "strategy_driven": True,  # LLM戦略による発注
                        "procurement_strategy": procurement_strategy[
                            "procurement_strategy"
                        ],
                    }
                    all_orders.append(order_info)

            procurement_decision["orders_placed"] = all_orders
            procurement_decision["total_orders"] = len(all_orders)

            # 実行アクション記録
            if all_orders:
                for order in all_orders:
                    action = {
                        "type": "procurement_order_llm",
                        "product": order["product"],
                        "quantity": order["quantity"],
                        "order_id": order["order_id"],
                        "urgency": order["urgency"],
                        "delivery_priority": order["delivery_priority"],
                        "strategy_driven": order["strategy_driven"],
                        "llm_strategy": order["procurement_strategy"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    state.executed_actions.append(action)

            # State更新
            state.procurement_decision = procurement_decision

            # ログ出力
            orders_count = len(all_orders)
            strategy = procurement_strategy["procurement_strategy"]
            logger.info(
                f"✅ Stateful発注依頼完了: orders={orders_count}, strategy={strategy}, llm_used=True"
            )

        except Exception as e:
            logger.error(f"Stateful発注依頼エラー: {e}")
            state.errors.append(f"procurement: {str(e)}")
            state.processing_status = "error"

        return state

    @traceable(name="sales_processing_analysis")
    async def sales_processing_node(self, state: ManagementState) -> ManagementState:
        """売上処理nodeのLangGraph Stateful関数 - LLMベースの査定分析を実行"""
        logger.info(f"✅ Stateful売上処理開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "sales_processing_analysis",
            "input_state": {
                "has_business_metrics": state.business_metrics is not None,
                "current_sales": state.business_metrics.sales
                if state.business_metrics
                else 0,
                "current_profit_margin": state.business_metrics.profit_margin
                if state.business_metrics
                else 0,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "sales_processing"
            state.processing_status = "processing"

            # LLMベース売上処理分析 (常にLLM使用)
            try:
                from src.simulations.sales_simulation import simulate_purchase_events

                # 販売シミュレーションを実行 (短時間バージョン)
                sales_lambda = 5.0
                simulation_result = await simulate_purchase_events(
                    sales_lambda=sales_lambda,
                    verbose=False,
                    period_name="営業時間",
                )

                # シミュレーション結果を取得
                conversion_rate = simulation_result.get("conversion_rate", 0)
                total_revenue = simulation_result.get("total_revenue", 0)
                transactions = simulation_result.get("successful_sales", 0)
                total_events = simulation_result.get("total_events", 0)

                # **LLMを常に呼び出し** - シミュレーション結果をプロンプトに含めて分析
                llm_prompt = f"""
以下の売上シミュレーション結果を詳細に分析し、パフォーマンス評価と改善戦略を提案してください。

【シミュレーション結果】
- 総イベント数: {total_events}
- 成功トランザクション数: {transactions}
- コンバージョン率: {conversion_rate:.3f} ({conversion_rate:.1%})
- 総売上: ¥{total_revenue:.0f}

【分析要求】
1. パフォーマンスレベルの評価 (excellent/good/acceptable/needs_improvement)
2. 売上効率の詳細分析
3. 改善提案 (3-5個の具体的な戦略)
4. 予測される改善効果
5. 実施の優先順位付け

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "performance_rating": "パフォーマンス評価レベル",
    "efficiency_analysis": "売上効率の詳細分析文",
    "recommendations": ["改善提案1", "改善提案2", "改善提案3"],
    "expected_impact": "改善効果の全体評価",
    "priority_actions": ["優先度高: アクション1", "優先度中: アクション2", "優先度低: アクション3"],
    "analysis_summary": "全体的な分析まとめ（100文字以上）"
}}
```
"""
                messages = [
                    self.llm_manager.create_ai_message(
                        role="system", content=self.system_prompt
                    ),
                    self.llm_manager.create_ai_message(role="user", content=llm_prompt),
                ]

                logger.info("LLM売上処理分析開始 - シミュレーション結果統合")
                response = await self.llm_manager.generate_response(
                    messages, max_tokens=1200
                )

                if response.success:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    llm_analysis_result = json.loads(content)

                    # LLMレスポンスからデータを抽出
                    performance_rating = llm_analysis_result.get(
                        "performance_rating", "unknown"
                    )
                    efficiency_analysis = llm_analysis_result.get(
                        "efficiency_analysis", ""
                    )
                    recommendations = llm_analysis_result.get("recommendations", [])
                    expected_impact = llm_analysis_result.get("expected_impact", "")
                    priority_actions = llm_analysis_result.get("priority_actions", [])
                    analysis_summary = llm_analysis_result.get("analysis_summary", "")

                    logger.info(
                        f"LLM売上処理分析成功: rating={performance_rating}, recommendations={len(recommendations)}"
                    )

                    # LLM分析結果をログ出力
                    logger.info("=== LLM Sales Processing Analysis Details ===")
                    logger.info(f"Performance Rating: {performance_rating}")
                    logger.info(f"Efficiency Analysis: {efficiency_analysis[:100]}...")
                    logger.info(f"Recommendations Count: {len(recommendations)}")
                    logger.info(f"Expected Impact: {expected_impact}")
                    logger.info(f"Analysis Summary: {analysis_summary[:100]}...")

                else:
                    # LLM失敗時のフォールバック - ハードコード評価
                    logger.warning(
                        f"LLM売上処理分析失敗: {response.error_message}, フォールバック使用"
                    )
                    performance_rating = "acceptable"
                    efficiency_analysis = f"コンバージョン率{conversion_rate:.1%}の標準的な売上効率。さらなる分析が必要。"
                    recommendations = [
                        "売上データの傾向分析",
                        "顧客行動の調査",
                        "プロモーション効果の検証",
                    ]
                    expected_impact = "基本的な売上改善効果"
                    priority_actions = [
                        "優先度高: データ分析実施",
                        "優先度中: 顧客調査",
                        "優先度低: 効果検証",
                    ]
                    analysis_summary = f"売上データに基づく基本分析を実施。コンバージョン率{conversion_rate:.1%}での営業活動を評価。"

                # 実行アクション項目 (LLM結果に基づく)
                action_items = (
                    priority_actions
                    if priority_actions
                    else [
                        "売上データの詳細分析",
                        "顧客フィードバック収集",
                        "競合店の動向調査",
                        "スタッフ研修実施",
                        "マーケティング予算の見直し",
                    ]
                )

                # 実行アクション記録 (LLM結果に基づくアクションのみ)
                for item in action_items[:3]:  # 上位3つを記録
                    action = {
                        "type": "sales_improvement",
                        "content": item,
                        "performance_rating": performance_rating,
                        "llm_based": True,
                        "timestamp": datetime.now().isoformat(),
                    }
                    state.executed_actions.append(action)

                # State更新
                state.sales_processing = {
                    "transactions": transactions,
                    "total_events": total_events,
                    "total_revenue": total_revenue,
                    "conversion_rate": f"{conversion_rate:.1%}",
                    "performance_rating": performance_rating,
                    "efficiency_analysis": efficiency_analysis,
                    "analysis": analysis_summary,
                    "recommendations": recommendations,
                    "expected_impact": expected_impact,
                    "priority_actions": priority_actions,
                    "action_items": action_items,
                    "simulation_result": simulation_result,
                    "llm_analysis_performed": bool(response.success),
                    "execution_timestamp": datetime.now().isoformat(),
                }

                logger.info(
                    f"✅ Stateful売上処理完了: rating={performance_rating}, revenue=¥{total_revenue}, llm_used={bool(response.success)}"
                )

            except Exception as e:
                logger.warning(f"売上処理LLM分析失敗: {e}")
                # 完全フォールバック
                state.sales_processing = {
                    "performance_rating": "error",
                    "analysis": f"LLM分析エラー: {str(e)}",
                    "recommendations": ["管理者へ連絡"],
                    "action_items": [],
                    "llm_analysis_performed": False,
                    "execution_timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Stateful売上処理エラー: {e}")
            state.errors.append(f"sales_processing: {str(e)}")
            state.processing_status = "error"

        return state

    @traceable(name="customer_service_interactions")
    async def customer_interaction_node(
        self, state: ManagementState
    ) -> ManagementState:
        """顧客対応nodeのLangGraph Stateful関数 - LLMで顧客フィードバックを分析し現実的な対応戦略を決定"""
        logger.info(f"✅ Stateful顧客対応開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "customer_service_interactions",
            "input_state": {
                "has_business_metrics": state.business_metrics is not None,
                "current_customer_satisfaction": state.business_metrics.customer_satisfaction
                if state.business_metrics
                else 0,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "customer_interaction"
            state.processing_status = "processing"

            # 顧客フィードバック収集
            feedback = self.collect_customer_feedback()

            # 現在のビジネス状況取得
            customer_score = (
                state.business_metrics.customer_satisfaction
                if state.business_metrics
                else 3.0
            )
            current_sales = (
                state.business_metrics.sales if state.business_metrics else 0
            )

            # LLMによる顧客対応戦略分析
            customer_strategy_prompt = f"""
あなたは自動販売機事業の顧客成功マネージャーです。以下の顧客フィードバックと事業状況を分析し、最適な顧客対応戦略を決定してください。

【顧客フィードバック状況】
- 収集フィードバック数: {feedback.get("feedback_count", 0)}件
- 平均顧客満足度: {feedback.get("average_rating", 0)}/5.0
- 人気リクエスト: {feedback.get("top_requests", [])}
- 全体トレンド: {feedback.get("trends", "")}

【事業状況】
- 全体顧客満足度: {customer_score}/5.0
- 月間売上: ¥{current_sales:,}
- サービス特性: 自動販売機 (24時間・セルフサービス)

【分析要件】
1. フィードバック内容の感情分析（満足/不満/提案）
2. 対応優先度の判断（即時/計画的/モニタリング）
3. 事業インパクトの評価（売上影響/口コミ影響/ブランド影響）
4. 現実的な対応策の立案（自動販売機運営制約を考慮）

【自動販売機事業の現実的対応オプション】
- 即時対応: 商品補充、機械メンテナンス、クレーム処理
- 計画的施策: キャンペーン企画、メニュー改善、スタッフ研修
- モニタリング: 継続調査、アンケート実施、トレンド分析

【出力形式】
JSON形式で以下の構造で回答してください：
{{
    "feedback_analysis": {{
        "sentiment_summary": "全体的な感情傾向",
        "priority_level": "対応優先度 (urgent/high/medium/low)",
        "key_insights": ["重要な洞察1", "重要な洞察2"],
        "business_impact": "事業への影響度評価"
    }},
    "recommended_strategy": {{
        "primary_approach": "主要対応方針",
        "immediate_actions": ["即時実行アクション"],
        "long_term_initiatives": ["長期施策"],
        "resource_allocation": "必要リソースの見積もり",
        "expected_timeline": "期待効果発現期間"
    }},
    "specific_recommendations": {{
        "customer_service": ["カスタマーサービス改善策"],
        "product_offerings": ["商品・サービス改善提案"],
        "marketing_communications": ["コミュニケーション施策"],
        "operational_improvements": ["運営改善策"]
    }},
    "success_measurement": {{
        "kpi_tracking": ["追跡指標"],
        "target_improvements": "目標改善値",
        "monitoring_period": "モニタリング期間",
        "success_criteria": ["成功判定基準"]
    }},
    "implementation_considerations": {{
        "feasibility": "実行可能性評価",
        "cost_benefit": "費用対効果分析",
        "risk_assessment": "リスク評価",
        "contingency_plans": ["代替案"]
    }}
}}
"""

            messages = [
                self.llm_manager.create_ai_message(
                    role="system",
                    content="あなたは自動販売機事業の顧客マネージャーです。現実的で実行可能な顧客対応戦略を立案してください。",
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=customer_strategy_prompt
                ),
            ]

            logger.info("LLM顧客対応戦略分析開始 - フィードバック統合")

            try:
                response = await self.llm_manager.generate_response(
                    messages, max_tokens=1200
                )

                if response.success:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    strategy_analysis = json.loads(content)

                    # デフォルト値の設定
                    feedback_analysis = strategy_analysis.get("feedback_analysis", {})
                    feedback_analysis.setdefault("sentiment_summary", "neutral")
                    feedback_analysis.setdefault("priority_level", "medium")
                    feedback_analysis.setdefault("key_insights", [])
                    feedback_analysis.setdefault("business_impact", "minimal")

                    recommended_strategy = strategy_analysis.get(
                        "recommended_strategy", {}
                    )
                    recommended_strategy.setdefault(
                        "primary_approach", "monitor_and_maintain"
                    )
                    recommended_strategy.setdefault("immediate_actions", [])
                    recommended_strategy.setdefault("long_term_initiatives", [])
                    recommended_strategy.setdefault("resource_allocation", "minimal")
                    recommended_strategy.setdefault("expected_timeline", "ongoing")

                    specific_recommendations = strategy_analysis.get(
                        "specific_recommendations", {}
                    )
                    specific_recommendations.setdefault("customer_service", [])
                    specific_recommendations.setdefault("product_offerings", [])
                    specific_recommendations.setdefault("marketing_communications", [])
                    specific_recommendations.setdefault("operational_improvements", [])

                    success_measurement = strategy_analysis.get(
                        "success_measurement", {}
                    )
                    success_measurement.setdefault(
                        "kpi_tracking", ["customer_satisfaction"]
                    )
                    success_measurement.setdefault(
                        "target_improvements", "5% improvement"
                    )
                    success_measurement.setdefault("monitoring_period", "monthly")
                    success_measurement.setdefault(
                        "success_criteria", ["feedback_improvement"]
                    )

                    implementation_considerations = strategy_analysis.get(
                        "implementation_considerations", {}
                    )
                    implementation_considerations.setdefault("feasibility", "high")
                    implementation_considerations.setdefault("cost_benefit", "positive")
                    implementation_considerations.setdefault("risk_assessment", "low")
                    implementation_considerations.setdefault("contingency_plans", [])

                    logger.info(
                        f"LLM顧客対応戦略分析成功: 優先度={feedback_analysis['priority_level']}, アプローチ={recommended_strategy['primary_approach']}"
                    )

                    # LLM分析結果に基づく実際のアクション実行
                    executed_actions = []

                    if feedback_analysis["priority_level"] == "urgent":
                        # エンゲージメントキャンペーン作成
                        from src.agents.management_agent.customer_tools.create_customer_engagement_campaign import (
                            create_customer_engagement_campaign,
                        )

                        campaign_result = await create_customer_engagement_campaign(
                            "retention"
                        )

                        executed_actions.append(
                            {
                                "type": "emergency_campaign",
                                "campaign_type": campaign_result.get(
                                    "campaign_type", "emergency"
                                ),
                                "reason": f"Urgent customer feedback: {feedback_analysis['sentiment_summary']}",
                                "llm_driven": True,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        customer_interaction = {
                            "action": "urgent_customer_engagement",
                            "reasoning": f"LLM分析に基づく緊急顧客対応: {feedback_analysis['business_impact']}",
                            "feedback_analysis": feedback_analysis,
                            "strategy": recommended_strategy,
                            "recommendations": specific_recommendations,
                            "success_measurement": success_measurement,
                            "campaign_implemented": campaign_result,
                            "feedback_collected": feedback,
                            "llm_analysis_performed": True,
                            "execution_timestamp": datetime.now().isoformat(),
                        }

                    elif feedback_analysis["priority_level"] in ["high", "medium"]:
                        # サービス改善アクション
                        executed_actions.append(
                            {
                                "type": "service_improvement_program",
                                "improvements": specific_recommendations[
                                    "customer_service"
                                ],
                                "reason": f"Customer satisfaction initiative: {feedback_analysis['key_insights']}",
                                "llm_driven": True,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        customer_interaction = {
                            "action": "planned_service_improvement",
                            "reasoning": f"LLM分析に基づく計画的顧客対応: {recommended_strategy['primary_approach']}",
                            "feedback_analysis": feedback_analysis,
                            "strategy": recommended_strategy,
                            "recommendations": specific_recommendations,
                            "success_measurement": success_measurement,
                            "feedback_collected": feedback,
                            "llm_analysis_performed": True,
                            "execution_timestamp": datetime.now().isoformat(),
                        }
                    else:
                        # モニタリング継続
                        customer_interaction = {
                            "action": "feedback_monitoring",
                            "reasoning": f"安定した顧客状況継続モニタリング: {feedback_analysis['sentiment_summary']}",
                            "feedback_analysis": feedback_analysis,
                            "strategy": recommended_strategy,
                            "recommendations": specific_recommendations,
                            "success_measurement": success_measurement,
                            "feedback_collected": feedback,
                            "llm_analysis_performed": True,
                            "execution_timestamp": datetime.now().isoformat(),
                        }

                else:
                    logger.warning(f"LLM顧客対応戦略分析失敗: {response.error_message}")
                    # LLM失敗時のフォールバック
                    feedback_count = feedback.get("feedback_count", 0)

                    if feedback_count > 15:
                        from src.agents.management_agent.customer_tools.create_customer_engagement_campaign import (
                            create_customer_engagement_campaign,
                        )

                        campaign_result = await create_customer_engagement_campaign(
                            "loyalty"
                        )

                        customer_interaction = {
                            "action": "engagement_campaign_created",
                            "reasoning": f"多くのフィードバック({feedback_count}件)に基づく顧客エンゲージメント強化",
                            "actions_planned": ["ロイヤリティキャンペーン実施"],
                            "campaign_details": campaign_result,
                            "feedback_collected": feedback,
                            "llm_analysis_performed": False,
                        }

                        executed_actions = [
                            {
                                "type": "customer_campaign",
                                "campaign_type": campaign_result.get("campaign_type"),
                                "target_audience": campaign_result.get("target"),
                                "expected_impact": campaign_result.get(
                                    "expected_impact"
                                ),
                                "llm_driven": False,
                                "timestamp": datetime.now().isoformat(),
                            }
                        ]

                    elif customer_score < 3.5:
                        customer_interaction = {
                            "action": "customer_service_improvement",
                            "reasoning": f"顧客満足度({customer_score}/5.0)が低いためサービス改善",
                            "actions_planned": ["サービス品質調査", "スタッフ研修実施"],
                            "feedback_collected": feedback,
                            "llm_analysis_performed": False,
                        }

                        executed_actions = [
                            {
                                "type": "service_improvement",
                                "trigger_reason": f"Low satisfaction score: {customer_score}",
                                "planned_actions": customer_interaction[
                                    "actions_planned"
                                ],
                                "llm_driven": False,
                                "timestamp": datetime.now().isoformat(),
                            }
                        ]
                    else:
                        customer_interaction = {
                            "action": "monitor_feedback",
                            "reasoning": f"現在の満足度({customer_score}/5.0)は安定",
                            "actions_planned": ["継続フィードバック収集"],
                            "feedback_collected": feedback,
                            "llm_analysis_performed": False,
                        }
                        executed_actions = []

            except Exception as e:
                logger.error(f"顧客対応戦略LLM分析エラー: {e}")
                # 完全フォールバック
                feedback_count = feedback.get("feedback_count", 0)

                if feedback_count > 15:
                    from src.agents.management_agent.customer_tools.create_customer_engagement_campaign import (
                        create_customer_engagement_campaign,
                    )

                    try:
                        campaign_result = await create_customer_engagement_campaign(
                            "loyalty"
                        )
                        action_type = "engagement_campaign_created"
                    except:
                        campaign_result = {"status": "fallback"}
                        action_type = "campaign_planned_manually"

                    customer_interaction = {
                        "action": action_type,
                        "reasoning": f"フィードバック多数({feedback_count}件)で対応策実行",
                        "feedback_collected": feedback,
                        "llm_analysis_performed": False,
                    }
                    executed_actions = [
                        {
                            "type": "customer_engagement",
                            "reason": "high feedback volume",
                            "llm_driven": False,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ]
                else:
                    customer_interaction = {
                        "action": "basic_feedback_collection",
                        "reasoning": "通常顧客対応継続",
                        "feedback_collected": feedback,
                        "llm_analysis_performed": False,
                    }
                    executed_actions = []

            # 実行アクション記録
            for action in executed_actions:
                state.executed_actions.append(action)

            # State更新
            state.customer_interaction = customer_interaction

            feedback_count = feedback.get("feedback_count", 0)
            action_taken = customer_interaction.get("action", "no_action")
            llm_used = customer_interaction.get("llm_analysis_performed", False)

            logger.info(
                f"✅ Stateful顧客対応完了: action={action_taken}, feedback={feedback_count}, llm_used={llm_used}"
            )

        except Exception as e:
            logger.error(f"Stateful顧客対応エラー: {e}")
            state.errors.append(f"customer_interaction: {str(e)}")
            state.processing_status = "error"

        return state

    @traceable(name="financial_calculations")
    async def profit_calculation_node(self, state: ManagementState) -> ManagementState:
        """利益計算nodeのLangGraph Stateful関数 - ツールベースの財務分析を実行"""
        logger.info(f"✅ Stateful利益計算開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "financial_calculations",
            "input_state": {
                "has_financial_analysis": state.financial_analysis is not None,
                "has_sales_analysis": state.sales_analysis is not None,
                "current_profit_margin": state.financial_analysis.get(
                    "profit_margin", 0
                )
                if state.financial_analysis
                else 0,
                "current_sales": state.financial_analysis.get("sales", 0)
                if state.financial_analysis
                else 0,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "profit_calculation"
            state.processing_status = "processing"

            # ツールレジストリから必要なツールを取得
            tools = {tool.name: tool for tool in self.tools}

            if "get_business_data" not in tools:
                logger.error("get_business_dataツールが利用できません")
                state.errors.append("profit_calculation: get_business_dataツール未取得")
                state.processing_status = "error"
                return state

            if "analyze_financials" not in tools:
                logger.error("analyze_financialsツールが利用できません")
                state.errors.append(
                    "profit_calculation: analyze_financialsツール未取得"
                )
                state.processing_status = "error"
                return state

            get_business_data_tool = tools["get_business_data"]
            analyze_financials_tool = tools["analyze_financials"]

            # ツール使用: 最新ビジネス指標取得
            logger.info("ツール経由でビジネス指標を取得")
            try:
                business_data_result = await get_business_data_tool.ainvoke({})
                logger.info(
                    f"ツール get_business_data 呼び出し成功: {type(business_data_result)}"
                )
                latest_metrics = (
                    business_data_result
                    if isinstance(business_data_result, dict)
                    else {}
                )

                # ツール使用: 詳細財務分析実行
                logger.info("ツール経由で財務パフォーマンス分析を実行")
                try:
                    raw_result = await analyze_financials_tool.ainvoke({})
                    logger.info(
                        f"ツール analyze_financials 生結果タイプ: {type(raw_result)}"
                    )

                    # 結果が辞書の場合そのまま使用、辞書でない場合はデフォルトを使用
                    if isinstance(raw_result, dict):
                        financial_analysis_result = raw_result
                    elif isinstance(raw_result, str):
                        # 文字列の場合はJSONとしてパースを試行
                        try:
                            import json

                            financial_analysis_result = json.loads(raw_result)
                        except json.JSONDecodeError:
                            financial_analysis_result = {
                                "analysis": raw_result,
                                "recommendations": ["ツール出力パース失敗"],
                            }
                    else:
                        # その他の型の場合は基本構造を作成
                        financial_analysis_result = {
                            "analysis": str(raw_result),
                            "recommendations": ["ツール出力処理済み"],
                        }

                    logger.info(
                        f"ツール analyze_financials 処理成功: 推奨事項={len(financial_analysis_result.get('recommendations', []))}件"
                    )
                except Exception as tool_error:
                    logger.error(
                        f"analyze_financialsツール実行詳細エラー: {tool_error}"
                    )
                    import traceback

                    logger.error(f"ツール実行トレース: {traceback.format_exc()}")
                    financial_analysis_result = {
                        "analysis": f"ツール実行エラー: {str(tool_error)}",
                        "recommendations": ["ツール使用失敗、フォールバック分析"],
                    }

                # 実行アクション記録 (ツール使用)
                action = {
                    "type": "profit_calculation_with_tools",
                    "tools_used": ["get_business_data", "analyze_financials"],
                    "latest_data_integrated": latest_metrics,
                    "extended_analysis": financial_analysis_result,
                    "timestamp": datetime.now().isoformat(),
                }
                state.executed_actions.append(action)

            except Exception as e:
                logger.error(f"ツール経由財務データ取得失敗: {e}")
                import traceback

                logger.error(f"詳細トレース: {traceback.format_exc()}")
                # フォールバック: 既存データを使用
                financial_analysis = state.financial_analysis or {}
                latest_metrics = financial_analysis
                financial_analysis_result = {
                    "recommendations": ["ツール使用失敗、フォールバック分析"]
                }

                # エラーアクション記録
                action = {
                    "type": "profit_calculation_fallback",
                    "error_details": f"ツール使用失敗: {str(e)}",
                    "fallback_used": True,
                    "timestamp": datetime.now().isoformat(),
                }
                state.executed_actions.append(action)

            # 利益計算: ツールから取得したデータを使用
            current_revenue = latest_metrics.get("sales", 0)
            current_profit_margin = latest_metrics.get("profit_margin", 0)
            current_customer_satisfaction = latest_metrics.get(
                "customer_satisfaction", 3.0
            )

            # 精密な利益計算 (ツールデータベース)
            profit_margin_val = (
                float(current_profit_margin)
                if isinstance(current_profit_margin, (int, float))
                else 0.0
            )
            profit_amount = current_revenue * profit_margin_val

            # 財務健全性評価 (ツール推奨と組み合わせ)
            margin_level = "unknown"
            if profit_margin_val > 0.3:
                margin_level = "excellent"
            elif profit_margin_val > 0.2:
                margin_level = "good"
            elif profit_margin_val > 0.1:
                margin_level = "acceptable"
            else:
                margin_level = "critical"

            # ツールによる推奨事項と内部推奨を統合
            tool_recommendations = financial_analysis_result.get("recommendations", [])
            internal_recommendations = []
            if margin_level == "excellent":
                internal_recommendations.append("規模拡大検討")
            elif margin_level == "good":
                internal_recommendations.append("安定維持")
            elif margin_level == "acceptable":
                internal_recommendations.append("効率改善")
            else:
                internal_recommendations.append("抜本的見直し")

            all_recommendations = tool_recommendations + internal_recommendations

            profit_calculation_result = {
                "total_revenue": current_revenue,
                "profit_margin": profit_margin_val,
                "profit_amount": profit_amount,
                "customer_satisfaction_score": current_customer_satisfaction,
                "margin_level": margin_level,
                "tool_based_analysis": financial_analysis_result.get("analysis", ""),
                "recommendations": all_recommendations,
                "calculation_method": "tool_integrated",
                "data_source": "get_business_data_tool",
                "analysis_source": "analyze_financials_tool",
                "calculation_timestamp": datetime.now().isoformat(),
            }

            # 危機的状況の場合、追加アクション記録
            if margin_level == "critical":
                action = {
                    "type": "financial_alert",
                    "alert_level": "critical",
                    "margin": profit_margin_val,
                    "recommendations": all_recommendations,
                    "tool_based": True,
                    "timestamp": datetime.now().isoformat(),
                }
                state.executed_actions.append(action)

            # State更新
            state.profit_calculation = profit_calculation_result

            # ログ出力 (ツール使用状況含む)
            logger.info(
                f"✅ Stateful利益計算完了（ツール統合）: margin={profit_margin_val:.1%}, level={margin_level}, tools_used=2"
            )

        except Exception as e:
            logger.error(f"Stateful利益計算エラー: {e}")
            state.errors.append(f"profit_calculation: {str(e)}")
            state.processing_status = "error"

        return state

    @conditional_traceable(name="strategic_management_feedback")
    async def feedback_node(self, state: ManagementState) -> ManagementState:
        """フィードバックnodeのLangGraph Stateful関数 - LLMベースの戦略的フィードバック分析を実行"""
        logger.info(f"✅ Stateful戦略的フィードバック開始: step={state.current_step}")

        # トレース用メタデータの準備
        trace_metadata = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "current_step": state.current_step,
            "business_date": state.business_date.isoformat()
            if state.business_date
            else None,
            "start_time": datetime.now().isoformat(),
            "agent_type": "management_agent",
            "node_type": "strategic_feedback",
            "input_state": {
                "total_actions_count": len(state.executed_actions),
                "errors_count": len(state.errors),
                "has_business_metrics": state.business_metrics is not None,
                "processing_status": state.processing_status,
            },
        }

        try:
            # ステップ更新
            state.current_step = "feedback"
            state.processing_status = "processing"

            # 戦略的フィードバック分析のための全データ集約
            comprehensive_context = self._prepare_strategic_context(state)

            # LLMによる戦略的分析実行 (async版を使用)
            strategic_analysis = await self._perform_strategic_feedback_analysis(
                comprehensive_context
            )

            # 分析結果から構造化されたフィードバック生成
            feedback_data = self._structure_strategic_feedback(
                state, strategic_analysis
            )

            # 最終報告書の生成 (戦略的視点を含む)
            final_report = self._generate_strategic_final_report(state, feedback_data)

            # State更新
            state.feedback = feedback_data
            state.final_report = final_report
            state.processing_status = "completed"

            # 戦略的洞察のログ出力
            logger.info(
                f"✅ Strategic feedback completed - Priorities: {len(feedback_data.get('tomorrow_priorities', []))}"
            )

        except Exception as e:
            logger.error(f"Strategic feedback node error: {e}")
            state.errors.append(f"feedback: {str(e)}")
            state.processing_status = "completed_with_errors"

            # エラー時も最小限のフィードバック生成
            feedback_data = self._create_fallback_feedback(state)
            final_report = self._generate_minimal_final_report(state, feedback_data)
            state.feedback = feedback_data
            state.final_report = final_report

        return state

    def _prepare_strategic_context(self, state: ManagementState) -> str:
        """
        戦略的フィードバック分析のための全Stateデータを構造化された文脈に集約

        Args:
            state: 現在のManagementState

        Returns:
            LLMプロンプト用の戦略的文脈文字列
        """
        context_parts = []

        # 基本ビジネスメトリクス
        if state.business_metrics:
            context_parts.append(
                f"""
【基本事業指標】
- 売上: ¥{state.business_metrics.sales:,}
- 利益率: {state.business_metrics.profit_margin:.1%}
- 顧客満足度: {state.business_metrics.customer_satisfaction}/5.0
- 在庫状態: {state.business_metrics.inventory_level}
            """.strip()
            )

        # 在庫分析
        if state.inventory_analysis:
            context_parts.append(
                f"""
【在庫管理分析】
- ステータス: {state.inventory_analysis.get("status", "unknown")}
- 在庫不足商品: {", ".join(state.inventory_analysis.get("low_stock_items", []))}
- 危機的商品: {", ".join(state.inventory_analysis.get("critical_items", []))}
- 在庫切れリスク: {state.inventory_analysis.get("estimated_stockout", {})}
- LLM分析: {state.inventory_analysis.get("llm_analysis", "なし")[:200]}
            """.strip()
            )

        # 売上・財務分析
        if state.sales_analysis:
            context_parts.append(
                f"""
【売上・財務分析】
- 売上トレンド: {state.sales_analysis.get("sales_trend", "unknown")}
- 戦略提案数: {len(state.sales_analysis.get("strategies", []))}件
- 財務概要: {state.sales_analysis.get("financial_overview", "なし")}
- LLM分析: {state.sales_analysis.get("analysis", "なし")[:200]}
            """.strip()
            )

        # 価格戦略決定
        if state.pricing_decision:
            context_parts.append(
                f"""
【価格戦略】
- 戦略: {state.pricing_decision.get("strategy", "unknown")}
- 商品価格更新: {len(state.pricing_decision.get("product_updates", []))}件
- LLM分析: {state.pricing_decision.get("llm_analysis", "なし")[:200]}
            """.strip()
            )

        # 在庫補充決定
        if state.restock_decision:
            tasks = state.restock_decision.get("tasks_assigned", [])
            context_parts.append(
                f"""
【補充タスク】
- 補充戦略: {state.restock_decision.get("strategy", "unknown")}
- 割り当てタスク数: {len(tasks)}件
- 緊急タスク: {len([t for t in tasks if t.get("urgency") == "urgent"])}件
- LLM分析: {state.restock_decision.get("llm_analysis", "なし")[:200]}
            """.strip()
            )

        # 発注決定
        if state.procurement_decision:
            orders = state.procurement_decision.get("orders_placed", [])
            context_parts.append(
                f"""
【調達発注】
- 発注戦略: {state.procurement_decision.get("strategy", "unknown")}
- 発注数: {len(orders)}件
- LLM分析: {state.procurement_decision.get("llm_analysis", "なし")[:200]}
            """.strip()
            )

        # 売上処理・顧客満足度
        if state.sales_processing:
            context_parts.append(
                f"""
【売上処理パフォーマンス】
- パフォーマンス評価: {state.sales_processing.get("performance_rating", "unknown")}
- 取引数: {state.sales_processing.get("transactions", 0)}件
- コンバージョン率: {state.sales_processing.get("conversion_rate", "unknown")}
- LLM分析: {state.sales_processing.get("analysis", "なし")[:200]}
            """.strip()
            )

        # 顧客対応分析
        if state.customer_interaction:
            context_parts.append(
                f"""
【顧客対応分析】
- 対応アクション: {state.customer_interaction.get("action", "unknown")}
- LLM分析実施: {state.customer_interaction.get("llm_analysis_performed", False)}
- LLM分析: {state.customer_interaction.get("reasoning", "なし")[:200]}
            """.strip()
            )

        # 財務計算
        if state.profit_calculation:
            context_parts.append(
                f"""
【財務健全性】
- 総売上: ¥{state.profit_calculation.get("total_revenue", 0):,}
- 利益率: {state.profit_calculation.get("profit_margin", 0):.1%}
- 利益額: ¥{state.profit_calculation.get("profit_amount", 0):,}
- 健全性レベル: {state.profit_calculation.get("margin_level", "unknown")}
            """.strip()
            )

        # 実行されたアクションの概要
        actions = state.executed_actions
        context_parts.append(
            f"""
【実行アクション概要】
- 総実行アクション数: {len(actions)}件
- LLM駆動アクション: {len([a for a in actions if a.get("llm_based") or a.get("llm_driven") or a.get("strategy_driven")])}件
- エラー発生数: {len(state.errors)}件
- セッションID: {state.session_id}

アクション詳細:
{chr(10).join([f"- {a.get('type', 'unknown')}: {a.get('content', a.get('product', '詳細なし'))}" for a in actions[-5:]])}  # 最新5件
        """.strip()
        )

        return "\n\n".join(context_parts)

    async def _perform_strategic_feedback_analysis(
        self, comprehensive_context: str
    ) -> Dict[str, Any]:
        """
        LLMによる包括的な戦略的フィードバック分析を実行

        Args:
            comprehensive_context: 集約されたビジネス文脈

        Returns:
            戦略的分析結果の辞書
        """
        logger.info("LLM戦略的フィードバック分析開始")

        strategic_prompt = f"""
あなたは自動販売機事業の経営者です。本日の全てのビジネスデータを分析し、明日以降の事業運営に対する戦略的な洞察と優先事項を決定してください。

【本日の業務実行結果】
{comprehensive_context}

【分析要件】
あなたは経営者の立場から以下の観点で分析してください：

1. **事業健全性評価**: 現在の売上・利益・顧客満足度の総合評価
2. **リスクアセスメント**: 明日以降に影響を及ぼす潜在リスクの特定
3. **戦略的優先順位**: 明日から重点的に取り組むべき事項のランキング
4. **長期戦略視点**: 中長期的な事業成長に向けた示唆
5. **アクションプラン**: 具体的な実行計画と担当割り当て

【自動販売機事業特有の考慮点】
- 24時間稼働だが人的リソースは有限
- 商品補充・メンテナンスは定期的に必要
- 顧客満足度が売上に直結しやすい
- 在庫切れは機会損失が大きい
- 競争環境の変化に迅速に対応する必要

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "executive_summary": "経営者向け事業全体の概要と結論（200文字以内）",
    "business_health_assessment": {{
        "overall_rating": "excellent/good/acceptable/poor/critical",
        "key_strengths": ["強み1", "強み2"],
        "key_concerns": ["懸念事項1", "懸念事項2"],
        "trend_direction": "improving/stable/declining"
    }},
    "tomorrow_priorities": [
        {{
            "rank": 1,
            "priority": "最優先事項の簡潔な記述",
            "reason": "この優先度とする理由",
            "expected_impact": "実行による期待効果",
            "assignee": "担当者（employee/manager/automated）",
            "timeline": "完了目標期間"
        }},
        {{
            "rank": 2,
            "priority": "2番目の優先事項...",
            "reason": "...",
            "expected_impact": "...",
            "assignee": "...",
            "timeline": "..."
        }},
        {{
            "rank": 3,
            "priority": "3番目の優先事項...",
            "reason": "...",
            "expected_impact": "...",
            "assignee": "...",
            "timeline": "..."
        }}
    ],
    "risk_assessment": {{
        "immediate_risks": ["緊急対応が必要なリスク"],
        "short_term_risks": ["1-7日以内に影響のリスク"],
        "mitigation_actions": ["リスク軽減のためのアクション"],
        "monitoring_points": ["重点監視事項"]
    }},
    "strategic_insights": {{
        "short_term_focus": "次週~1ヶ月以内の戦略的重点",
        "medium_term_opportunities": "1-3ヶ月程度の中期機会",
        "long_term_considerations": "3ヶ月以上の中長期視点",
        "competitive_positioning": "競争環境での自社位置づけ"
    }},
    "action_plan": {{
        "immediate_next_steps": ["直ちに実行すべきアクション"],
        "resource_allocation": {{
            "high_priority_employees": "緊急業務担当者数",
            "monitoring_responsibility": "監視担当者",
            "backup_plans": ["予備計画"]
        }},
        "success_metrics": ["成功判定指標"],
        "communication_plan": ["関係者への報告・連絡事項"]
    }},
    "comprehensive_analysis": "全体分析の詳細説明と経営判断の根拠（300文字以内）"
}}
```

戦略的洞察は具体的な行動指針となるように記述してください。
"""

        try:
            messages = [
                self.llm_manager.create_ai_message(
                    role="system",
                    content="あなたは自動販売機事業の戦略的経営コンサルタントです。データに基づいた実行可能な戦略的アドバイスを提供してください。",
                ),
                self.llm_manager.create_ai_message(
                    role="user", content=strategic_prompt
                ),
            ]

            response = await self.llm_manager.generate_response(
                messages, max_tokens=2000
            )

            if response.success:
                import json

                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                strategic_analysis = json.loads(content)

                # デフォルト値の設定
                strategic_analysis.setdefault(
                    "executive_summary", "戦略的分析を実行しました"
                )
                strategic_analysis.setdefault(
                    "business_health_assessment",
                    {
                        "overall_rating": "acceptable",
                        "key_strengths": [],
                        "key_concerns": [],
                        "trend_direction": "stable",
                    },
                )
                strategic_analysis.setdefault("tomorrow_priorities", [])
                strategic_analysis.setdefault(
                    "risk_assessment",
                    {
                        "immediate_risks": [],
                        "short_term_risks": [],
                        "mitigation_actions": [],
                        "monitoring_points": [],
                    },
                )
                strategic_analysis.setdefault(
                    "strategic_insights",
                    {
                        "short_term_focus": "安定運営業継続",
                        "medium_term_opportunities": "業務効率化検討",
                        "long_term_considerations": "顧客基盤拡大",
                        "competitive_positioning": "市場標準レベル",
                    },
                )
                strategic_analysis.setdefault(
                    "action_plan",
                    {
                        "immediate_next_steps": [],
                        "resource_allocation": {
                            "high_priority_employees": "1名",
                            "monitoring_responsibility": "manager",
                            "backup_plans": [],
                        },
                        "success_metrics": [],
                        "communication_plan": [],
                    },
                )
                strategic_analysis.setdefault(
                    "comprehensive_analysis",
                    f"LLM戦略分析実行: {len(comprehensive_context)}文字のデータを分析",
                )

                logger.info(
                    f"LLM戦略的フィードバック分析成功 - 優先事項: {len(strategic_analysis.get('tomorrow_priorities', []))}件"
                )
                return strategic_analysis

            else:
                logger.warning(f"LLM戦略的分析失敗: {response.error_message}")
                return self._create_fallback_strategic_analysis(comprehensive_context)

        except Exception as e:
            logger.error(f"戦略的フィードバック分析エラー: {e}")
            return self._create_fallback_strategic_analysis(comprehensive_context)

    def _structure_strategic_feedback(
        self, state: ManagementState, strategic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLMの戦略的分析結果を構造化されたフィードバックデータに変換

        Args:
            state: 現在のManagementState
            strategic_analysis: LLMからの戦略的分析結果

        Returns:
            構造化されたフィードバック辞書
        """
        feedback_data = {
            "executive_summary": strategic_analysis.get(
                "executive_summary", "分析実行"
            ),
            "business_health": strategic_analysis.get("business_health_assessment", {}),
            "tomorrow_priorities": strategic_analysis.get("tomorrow_priorities", []),
            "risk_assessment": strategic_analysis.get("risk_assessment", {}),
            "strategic_insights": strategic_analysis.get("strategic_insights", {}),
            "action_plan": strategic_analysis.get("action_plan", {}),
            "comprehensive_analysis": strategic_analysis.get(
                "comprehensive_analysis", "詳細分析実行"
            ),
            "performance_indicators": {},
            "actions_taken": state.executed_actions.copy(),
            "recommendations": [],
            "execution_timestamp": datetime.now().isoformat(),
        }

        # パフォーマンス指標の集約（戦略的視点から再評価）
        if state.business_metrics:
            feedback_data["performance_indicators"] = {
                "sales": state.business_metrics.sales,
                "profit_margin": state.business_metrics.profit_margin,
                "customer_satisfaction": state.business_metrics.customer_satisfaction,
                "inventory_efficiency": len(state.executed_actions),
                "strategic_rating": strategic_analysis.get(
                    "business_health_assessment", {}
                ).get("overall_rating", "unknown"),
            }

        # 推奨事項の生成（戦略的分析に基づく）
        recommendations = []

        # 明日の優先事項を推奨事項として追加
        for priority in feedback_data["tomorrow_priorities"][:3]:  # トップ3
            recommendations.append(
                f"優先度{priority.get('rank', '?')}: {priority.get('priority', '')}"
            )

        # 戦略的洞察からの推奨
        insights = feedback_data["strategic_insights"]
        if insights.get("short_term_focus"):
            recommendations.append(f"短期戦略: {insights['short_term_focus']}")

        # リスク軽減アクションを推奨事項として追加
        for mitigation in feedback_data["risk_assessment"].get(
            "mitigation_actions", []
        )[:2]:
            recommendations.append(f"リスク対応: {mitigation}")

        feedback_data["recommendations"] = recommendations

        return feedback_data

    def _generate_strategic_final_report(
        self, state: ManagementState, feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        戦略的視点を含む最終報告書を生成

        Args:
            state: ManagementState
            feedback_data: 構造化されたフィードバックデータ

        Returns:
            戦略的視点を含む最終報告書
        """
        final_report = {
            "session_id": state.session_id,
            "session_type": state.session_type,
            "business_metrics": state.business_metrics.dict()
            if state.business_metrics
            else None,
            "analyses_completed": {
                "inventory_analysis": bool(state.inventory_analysis),
                "sales_analysis": bool(state.sales_analysis),
                "pricing_decision": bool(state.pricing_decision),
                "restock_decision": bool(state.restock_decision),
                "procurement_decision": bool(state.procurement_decision),
                "sales_processing": bool(state.sales_processing),
                "customer_interaction": bool(state.customer_interaction),
                "profit_calculation": bool(state.profit_calculation),
                "strategic_feedback": True,
            },
            "actions_executed": state.executed_actions,
            "errors": state.errors,
            "executive_summary": feedback_data.get("executive_summary"),
            "strategic_priorities": feedback_data.get("tomorrow_priorities"),
            "business_health_rating": feedback_data.get("business_health", {}).get(
                "overall_rating", "unknown"
            ),
            "risk_assessment": feedback_data.get("risk_assessment"),
            "strategic_insights": feedback_data.get("strategic_insights"),
            "actionable_plan": feedback_data.get("action_plan"),
            "recommendations": feedback_data.get("recommendations", []),
            "final_status": "completed"
            if state.processing_status == "completed"
            else "completed_with_errors",
            "completion_timestamp": datetime.now().isoformat(),
        }

        return final_report

    def _create_fallback_feedback(self, state: ManagementState) -> Dict[str, Any]:
        """
        エラー時のフォールバックフィードバック生成

        Args:
            state: ManagementState

        Returns:
            基本的なフィードバックデータ
        """
        return {
            "executive_summary": f"業務実行完了 - アクション{len(state.executed_actions)}件、エラー{len(state.errors)}件",
            "business_health": {
                "overall_rating": "acceptable",
                "trend_direction": "stable",
            },
            "tomorrow_priorities": [
                {
                    "rank": 1,
                    "priority": "システム監視継続",
                    "reason": "安定運用維持のため",
                    "assignee": "automated",
                    "timeline": "ongoing",
                }
            ],
            "risk_assessment": {
                "immediate_risks": [],
                "monitoring_points": ["システム安定性"],
            },
            "strategic_insights": {"short_term_focus": "安定運用継続"},
            "action_plan": {"immediate_next_steps": ["通常業務継続"]},
            "comprehensive_analysis": f"基本業務遂行完了 - エラー時フォールバック分析",
            "performance_indicators": {},
            "actions_taken": state.executed_actions,
            "recommendations": ["システム安定性確認", "エラーログの確認"],
            "execution_timestamp": datetime.now().isoformat(),
        }

    def _generate_minimal_final_report(
        self, state: ManagementState, feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        最小限の最終報告書生成（エラー時用）

        Args:
            state: ManagementState
            feedback_data: フィードバックデータ

        Returns:
            最小限の最終報告書
        """
        return {
            "session_id": state.session_id,
            "completion_timestamp": datetime.now().isoformat(),
            "final_status": state.processing_status,
            "executive_summary": feedback_data.get("executive_summary", "エラー時完了"),
            "error_count": len(state.errors),
            "action_count": len(state.executed_actions),
        }

    def _perform_sync_strategic_feedback_analysis(
        self, comprehensive_context: str
    ) -> Dict[str, Any]:
        """
        同期版LLM戦略的フィードバック分析を実行（フォールバック使用）

        Args:
            comprehensive_context: 集約されたビジネス文脈

        Returns:
            戦略的分析結果の辞書
        """
        logger.info("同期LLM戦略的フィードバック分析開始（フォールバック使用）")

        # 直接フォールバック戦略分析を使用（LLMとの同期処理を避ける）
        return self._create_fallback_strategic_analysis(comprehensive_context)

    def _create_fallback_strategic_analysis(self, context: str) -> Dict[str, Any]:
        """
        LLM失敗時のフォールバック戦略的分析

        Args:
            context: ビジネス文脈

        Returns:
            基本的な戦略的分析結果
        """
        # 文脈から基本情報を抽出して分析
        sales_match = None
        profit_match = None
        inventory_match = None

        lines = context.split("\n")
        for line in lines:
            if "売上:" in line and "¥" in line:
                sales_match = line.strip()
            elif "利益率:" in line and "%" in line:
                profit_match = line.strip()
            elif "在庫不足商品:" in line:
                inventory_match = line.strip()

        # 基本分析判断
        ratings = ["excellent", "good", "acceptable", "poor", "critical"]
        rating_index = 2  # default acceptable

        if profit_match and "%" in profit_match:
            try:
                profit_val = float(profit_match.split("%")[0].split()[-1])
                if profit_val > 25:
                    rating_index = 0  # excellent
                elif profit_val > 15:
                    rating_index = 1  # good
                elif profit_val < 5:
                    rating_index = 4  # critical
            except:
                pass

        return {
            "executive_summary": f"事業分析完了。売上・在庫・顧客動向を把握し、明日への戦略的示唆を整理しました。",
            "business_health_assessment": {
                "overall_rating": ratings[rating_index],
                "key_strengths": ["業務実行完了", "戦略的アプローチ実施"],
                "key_concerns": ["システム依存度向上", "継続監視必要"],
                "trend_direction": "stable",
            },
            "tomorrow_priorities": [
                {
                    "rank": 1,
                    "priority": "業務成果の定量評価実施",
                    "reason": "戦略的意思決定の精度向上のため",
                    "expected_impact": "意思決定品質向上",
                    "assignee": "manager",
                    "timeline": "明日中に",
                },
                {
                    "rank": 2,
                    "priority": "システム安定性の確認",
                    "reason": "自動化処理の信頼性確保のため",
                    "expected_impact": "運用安定化",
                    "assignee": "employee",
                    "timeline": "今日中に",
                },
                {
                    "rank": 3,
                    "priority": "戦略的洞察の定期収集",
                    "reason": "中長期視点での改善機会把握のため",
                    "expected_impact": "戦略的柔軟性向上",
                    "assignee": "automated",
                    "timeline": "継続的に",
                },
            ],
            "risk_assessment": {
                "immediate_risks": ["システム不安定性", "分析精度不足"],
                "short_term_risks": ["業務効率化遅延", "戦略的判断の遅れ"],
                "mitigation_actions": [
                    "手動モニタリング強化",
                    "フォールバック運用準備",
                ],
                "monitoring_points": ["システムパフォーマンス", "業務実行結果"],
            },
            "strategic_insights": {
                "short_term_focus": "自動化システムの安定化と業務効率化",
                "medium_term_opportunities": "戦略的意思決定フレームワークの高度化",
                "long_term_considerations": "AI支援経営システムの完全導入",
                "competitive_positioning": "業界標準以上の戦略的洞察力",
            },
            "action_plan": {
                "immediate_next_steps": [
                    "システムパフォーマンス確認",
                    "エラーログ解析",
                    "業務継続性の確認",
                ],
                "resource_allocation": {
                    "high_priority_employees": "0名",
                    "monitoring_responsibility": "automated_system",
                    "backup_plans": ["手動業務切り替え準備"],
                },
                "success_metrics": ["システム安定稼働率95%以上", "業務実行率98%以上"],
                "communication_plan": ["経営陣への完了報告", "チームへの実施結果共有"],
            },
            "comprehensive_analysis": f"LLM分析失敗時のフォールバック戦略分析を実行。{len(lines)}行のデータを基に基本戦略的方向性を整理しました。",
        }

    async def morning_routine(self) -> Dict[str, Any]:
        """朝の業務ルーチン"""
        session_id = await self.start_management_session("morning_routine")

        try:
            # 夜間データ確認
            overnight_data = self.get_business_metrics()

            # 朝の分析
            morning_analysis = f"""
            昨夜の事業データを確認し、今日の業務優先順位を決定してください。
            
            【夜間データ】
            - 売上実績: {overnight_data["sales"]}
            - 在庫状況: {overnight_data["inventory_level"]}
            - 顧客満足度: {overnight_data["customer_satisfaction"]}
            
            【判断項目】
            1. 緊急対応が必要な事項
            2. 今日の重点業務
            3. 従業員への指示事項
            """

            decisions = await self.make_strategic_decision(morning_analysis)

            return {
                "session_id": session_id,
                "session_type": "morning_routine",
                "overnight_data": overnight_data,
                "decisions": decisions,
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def midday_check(self) -> Dict[str, Any]:
        """昼の業務チェック"""
        session_id = await self.start_management_session("midday_check")

        try:
            metrics = self.get_business_metrics()
            financial_analysis = await self.analyze_financial_performance()

            midday_analysis = f"""
            午前中の業績を確認し、午後の調整を行ってください。
            
            【午前実績】
            - 売上: {metrics["sales"]}
            - 利益率: {metrics["profit_margin"]}
            """

            decisions = await self.make_strategic_decision(midday_analysis)

            return {
                "session_id": session_id,
                "session_type": "midday_check",
                "metrics": metrics,
                "analysis": financial_analysis,
                "decisions": decisions,
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def evening_summary(self) -> Dict[str, Any]:
        """夕方の業務総括"""
        session_id = await self.start_management_session("evening_summary")

        try:
            daily_performance = self.get_business_metrics()
            inventory_status = await self.check_inventory_status()

            evening_analysis = f"""
            今日一日の業績を総括し、明日への改善点を特定してください。
            
            【今日の実績】
            - 売上: {daily_performance["sales"]}
            - 利益率: {daily_performance["profit_margin"]}
            - 在庫状況: {inventory_status["status"]}
            
            【分析項目】
            1. 今日の成功要因
            2. 改善が必要な領域
            3. 明日の重点課題
            """

            decisions = await self.make_strategic_decision(evening_analysis)

            return {
                "session_id": session_id,
                "session_type": "evening_summary",
                "daily_performance": daily_performance,
                "inventory_status": inventory_status,
                "decisions": decisions,
                "lessons_learned": [
                    "在庫管理の改善が必要",
                    "顧客満足度を維持できた",
                ],
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def feedback_engine(self) -> Dict[str, Any]:
        """夕方の業務総括"""
        session_id = await self.start_management_session("evening_summary")

        try:
            daily_performance = self.get_business_metrics()
            inventory_status = await self.check_inventory_status()

            evening_analysis = f"""
            今日一日の業績を総括し、明日への改善点を特定してください。
            
            【今日の実績】
            - 売上: {daily_performance["sales"]}
            - 利益率: {daily_performance["profit_margin"]}
            - 在庫状況: {inventory_status["status"]}
            
            【分析項目】
            1. 今日の成功要因
            2. 改善が必要な領域
            3. 明日の重点課題
            """

            decisions = await self.make_strategic_decision(evening_analysis)

            return {
                "session_id": session_id,
                "session_type": "evening_summary",
                "daily_performance": daily_performance,
                "inventory_status": inventory_status,
                "decisions": decisions,
                "lessons_learned": [
                    "在庫管理の改善が必要",
                    "顧客満足度を維持できた",
                ],
                "status": "completed",
            }

        finally:
            await self.end_management_session()


# グローバルインスタンス
management_agent = NodeBasedManagementAgent(provider="openai")
