"""
記録専用Agent

行動記録、パターン分析、学習データ蓄積を担当
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ManagementActionRecord(BaseModel):
    """管理行動記録"""

    record_id: str
    session_id: str
    timestamp: datetime
    action_type: str  # "decision", "instruction", "customer_response"
    context: Dict[str, Any]
    decision_process: str
    executed_action: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success_score: Optional[float] = None


class BusinessOutcomeRecord(BaseModel):
    """事業結果記録"""

    record_id: str
    session_id: str
    related_action_id: Optional[str]
    timestamp: datetime
    outcome_type: str  # "sales", "customer_satisfaction", "efficiency"
    metrics: Dict[str, float]
    success_level: str  # "excellent", "good", "average", "poor"
    lessons_learned: List[str]


class RecorderAgent:
    """行動記録・分析専用Agent"""

    def __init__(
        self,
        persist_directory: str = "./data/vector_store",
        enable_vector_store: bool = True,
    ):
        """
        Args:
            persist_directory: ベクトルストアの永続化ディレクトリ
            enable_vector_store: ベクトルストアを有効にするか（OpenAI APIキーが必要）
        """
        self.persist_directory = persist_directory
        self.enable_vector_store = enable_vector_store

        # シンプルな記録用ストレージ
        self.simple_action_store: List[Dict[str, Any]] = []
        self.simple_outcome_store: List[Dict[str, Any]] = []

        # Embeddings初期化（オプション）
        if enable_vector_store:
            try:
                self.embeddings = OpenAIEmbeddings()
                logger.info("OpenAI Embeddings initialized successfully")
            except Exception as e:
                logger.warning(
                    f"OpenAI Embeddings initialization failed: {e}. Using simple storage instead."
                )
                self.embeddings = None
                self.enable_vector_store = False
        else:
            self.embeddings = None
            logger.info("Vector store disabled, using simple storage")

        # ベクトルストアの初期化
        self._init_vector_stores()

        # 分析用LLM（オプション）- .envからAPIキーを明示的に読み取り
        if enable_vector_store:
            try:
                from src.config.security import secure_config

                if secure_config.openai_api_key:
                    self.analysis_llm = ChatOpenAI(
                        model="gpt-4",
                        temperature=0.1,
                        api_key=secure_config.openai_api_key,  # .envから明示的に読み取り
                    )
                    logger.info(
                        ".envからOpenAI APIキーを読み取りました（RecorderAgent分析用）"
                    )
                else:
                    logger.warning(
                        "OpenAI APIキーが設定されていないため、分析機能を無効化"
                    )
                    self.analysis_llm = None
            except Exception as e:
                logger.warning(f"Analysis LLM initialization failed: {e}")
                self.analysis_llm = None
        else:
            self.analysis_llm = None

        logger.info("RecorderAgent initialized")

    def _init_vector_stores(self):
        """ベクトルストアの初期化"""
        if not self.embeddings:
            logger.warning("Embeddings not available, vector stores disabled")
            self.action_store = None
            self.decision_store = None
            self.outcome_store = None
            return

        try:
            # 行動記録用ストア
            self.action_store = Chroma(
                collection_name="management_actions",
                embedding_function=self.embeddings,
                persist_directory=f"{self.persist_directory}/actions",
            )

            # 意思決定記録用ストア
            self.decision_store = Chroma(
                collection_name="management_decisions",
                embedding_function=self.embeddings,
                persist_directory=f"{self.persist_directory}/decisions",
            )

            # 結果記録用ストア
            self.outcome_store = Chroma(
                collection_name="business_outcomes",
                embedding_function=self.embeddings,
                persist_directory=f"{self.persist_directory}/outcomes",
            )

            logger.info("Vector stores initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector stores: {e}")
            self.action_store = None
            self.decision_store = None
            self.outcome_store = None

    async def record_action(
        self, action_record: ManagementActionRecord
    ) -> Dict[str, Any]:
        """行動を記録"""
        logger.info(f"Recording action: {action_record.record_id}")

        # ベクトルストアが利用可能な場合
        if self.action_store:
            try:
                # ベクトルストアに保存
                self.action_store.add_texts(
                    texts=[
                        f"""
                    Session: {action_record.session_id}
                    Type: {action_record.action_type}
                    Context: {action_record.context}
                    Decision: {action_record.decision_process}
                    Action: {action_record.executed_action}
                    Expected: {action_record.expected_outcome}
                    """
                    ],
                    metadatas=[
                        {
                            "record_id": action_record.record_id,
                            "session_id": action_record.session_id,
                            "action_type": action_record.action_type,
                            "timestamp": action_record.timestamp.isoformat(),
                        }
                    ],
                )

                return {
                    "success": True,
                    "record_id": action_record.record_id,
                    "timestamp": action_record.timestamp.isoformat(),
                    "storage": "vector_store",
                }

            except Exception as e:
                logger.error(f"Failed to record action to vector store: {e}")
                # フォールバックしてシンプルストレージに保存

        # シンプルストレージに保存（フォールバック）
        try:
            self.simple_action_store.append(action_record.dict())
            logger.info(f"Action recorded to simple storage: {action_record.record_id}")

            return {
                "success": True,
                "record_id": action_record.record_id,
                "timestamp": action_record.timestamp.isoformat(),
                "storage": "simple_storage",
            }
        except Exception as e:
            logger.error(f"Failed to record action: {e}")
            return {"success": False, "error": str(e)}

    async def record_outcome(
        self, outcome_record: BusinessOutcomeRecord
    ) -> Dict[str, Any]:
        """結果を記録"""
        logger.info(f"Recording outcome: {outcome_record.record_id}")

        # ベクトルストアが利用可能な場合
        if self.outcome_store:
            try:
                # ベクトルストアに保存
                self.outcome_store.add_texts(
                    texts=[
                        f"""
                    Session: {outcome_record.session_id}
                    Type: {outcome_record.outcome_type}
                    Metrics: {outcome_record.metrics}
                    Success Level: {outcome_record.success_level}
                    Lessons: {", ".join(outcome_record.lessons_learned)}
                    """
                    ],
                    metadatas=[
                        {
                            "record_id": outcome_record.record_id,
                            "session_id": outcome_record.session_id,
                            "outcome_type": outcome_record.outcome_type,
                            "success_level": outcome_record.success_level,
                            "timestamp": outcome_record.timestamp.isoformat(),
                        }
                    ],
                )

                return {
                    "success": True,
                    "record_id": outcome_record.record_id,
                    "timestamp": outcome_record.timestamp.isoformat(),
                    "storage": "vector_store",
                }

            except Exception as e:
                logger.error(f"Failed to record outcome to vector store: {e}")
                # フォールバックしてシンプルストレージに保存

        # シンプルストレージに保存（フォールバック）
        try:
            self.simple_outcome_store.append(outcome_record.dict())
            logger.info(
                f"Outcome recorded to simple storage: {outcome_record.record_id}"
            )

            return {
                "success": True,
                "record_id": outcome_record.record_id,
                "timestamp": outcome_record.timestamp.isoformat(),
                "storage": "simple_storage",
            }
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return {"success": False, "error": str(e)}

    async def search_similar_actions(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """類似の行動を検索"""
        if not self.action_store:
            logger.warning("Action store not available")
            return []

        try:
            results = self.action_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Failed to search similar actions: {e}")
            return []

    async def search_similar_outcomes(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """類似の結果を検索"""
        if not self.outcome_store:
            logger.warning("Outcome store not available")
            return []

        try:
            results = self.outcome_store.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Failed to search similar outcomes: {e}")
            return []

    async def extract_successful_patterns(
        self, session_type: str
    ) -> List[Dict[str, Any]]:
        """成功パターンを抽出"""
        logger.info(f"Extracting successful patterns for {session_type}")

        # 成功事例を検索
        query = f"session_type:{session_type} success_level:excellent"
        similar_outcomes = await self.search_similar_outcomes(query, k=10)

        patterns = []
        for outcome in similar_outcomes:
            if outcome["metadata"].get("success_level") in ["excellent", "good"]:
                patterns.append(
                    {
                        "session_id": outcome["metadata"].get("session_id"),
                        "success_level": outcome["metadata"].get("success_level"),
                        "content": outcome["content"],
                    }
                )

        logger.info(f"Found {len(patterns)} successful patterns")
        return patterns

    async def extract_failure_lessons(self, session_type: str) -> List[Dict[str, Any]]:
        """失敗事例から教訓を抽出"""
        logger.info(f"Extracting failure lessons for {session_type}")

        # 失敗事例を検索
        query = f"session_type:{session_type} success_level:poor"
        similar_outcomes = await self.search_similar_outcomes(query, k=10)

        lessons = []
        for outcome in similar_outcomes:
            if outcome["metadata"].get("success_level") in ["poor", "average"]:
                lessons.append(
                    {
                        "session_id": outcome["metadata"].get("session_id"),
                        "success_level": outcome["metadata"].get("success_level"),
                        "content": outcome["content"],
                    }
                )

        logger.info(f"Found {len(lessons)} failure lessons")
        return lessons

    async def analyze_session_patterns(self, session_type: str) -> Dict[str, Any]:
        """セッションパターンを分析"""
        logger.info(f"Analyzing session patterns for {session_type}")

        successful_patterns = await self.extract_successful_patterns(session_type)
        failure_lessons = await self.extract_failure_lessons(session_type)

        analysis = {
            "session_type": session_type,
            "total_patterns": len(successful_patterns) + len(failure_lessons),
            "successful_count": len(successful_patterns),
            "failure_count": len(failure_lessons),
            "success_rate": (
                len(successful_patterns)
                / (len(successful_patterns) + len(failure_lessons))
                if (len(successful_patterns) + len(failure_lessons)) > 0
                else 0
            ),
            "key_insights": [],
        }

        # LLMを使用して洞察を抽出
        if successful_patterns or failure_lessons:
            try:
                insights_prompt = f"""
                以下のセッションデータから主要な洞察を抽出してください:
                
                成功パターン: {len(successful_patterns)}件
                {successful_patterns[:3] if len(successful_patterns) > 0 else "なし"}
                
                失敗事例: {len(failure_lessons)}件
                {failure_lessons[:3] if len(failure_lessons) > 0 else "なし"}
                
                3つの主要な洞察を箇条書きで提供してください。
                """

                response = await self.analysis_llm.ainvoke(insights_prompt)
                analysis["key_insights"] = response.content.split("\n")

            except Exception as e:
                logger.error(f"Failed to extract insights: {e}")
                analysis["key_insights"] = ["分析に失敗しました"]

        return analysis

    async def generate_session_recommendations(self, session_type: str) -> List[str]:
        """セッションの推奨事項を生成"""
        logger.info(f"Generating recommendations for {session_type}")

        patterns_analysis = await self.analyze_session_patterns(session_type)

        try:
            recommendations_prompt = f"""
            以下の分析結果に基づいて、{session_type}セッションの推奨事項を5つ提供してください:
            
            成功率: {patterns_analysis["success_rate"]:.2%}
            主要な洞察: {patterns_analysis["key_insights"]}
            
            具体的で実行可能な推奨事項を提供してください。
            """

            response = await self.analysis_llm.ainvoke(recommendations_prompt)
            recommendations = [
                line.strip() for line in response.content.split("\n") if line.strip()
            ]

            return recommendations[:5]

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["推奨事項の生成に失敗しました"]

    async def update_daily_experience(self, daily_results: Dict[str, Any]):
        """日次経験データを更新"""
        logger.info("Updating daily experience data")

        # 各セッションの結果を記録
        for session_type, session_data in daily_results.items():
            if isinstance(session_data, dict):
                outcome_record = BusinessOutcomeRecord(
                    record_id=f"{session_data.get('session_id', 'unknown')}",
                    session_id=session_data.get("session_id", "unknown"),
                    related_action_id=None,
                    timestamp=datetime.now(),
                    outcome_type=session_type,
                    metrics=session_data.get("metrics", {}),
                    success_level="good",  # TODO: 実際の評価ロジック
                    lessons_learned=session_data.get("lessons_learned", []),
                )

                await self.record_outcome(outcome_record)

    def get_recent_outcomes(self, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """最近の結果記録を取得（辞書形式: record_id -> outcome）"""
        try:
            recent = (
                self.simple_outcome_store[-limit:] if self.simple_outcome_store else []
            )
            return {outcome["record_id"]: outcome for outcome in recent}
        except Exception as e:
            logger.error(f"最近の結果取得エラー: {e}")
            return {}

    async def record_daily_session(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """日次セッションを記録（簡易版）"""
        try:
            record_id = f"daily_{datetime.now().strftime('%Y%m%d')}"
            outcome_record = BusinessOutcomeRecord(
                record_id=record_id,
                session_id=session_data.get("day", "unknown"),
                related_action_id=None,
                timestamp=datetime.now(),
                outcome_type="daily_performance",
                metrics=session_data.get("performance", {}),
                success_level="good",  # デフォルト
                lessons_learned=session_data.get("learnings", []),
            )

            result = await self.record_outcome(outcome_record)
            return {"record_id": record_id, "success": True}
        except Exception as e:
            logger.error(f"日次セッション記録エラー: {e}")
            return {"success": False, "error": str(e)}


# グローバルインスタンス
recorder_agent = RecorderAgent()
