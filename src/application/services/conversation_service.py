import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from src.domain.models.product import Product
from src.domain.models.transaction import Transaction
from src.shared.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """会話メッセージ"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationSession:
    """会話セッション"""
    session_id: str
    customer_id: str
    machine_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages: List[ConversationMessage] = None
    context: Dict[str, Any] = None
    ai_insights: Dict[str, Any] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.context is None:
            self.context = {}
        if self.ai_insights is None:
            self.ai_insights = {}
        if self.tags is None:
            self.tags = []

class ConversationService:
    """会話サービス（NoSQL統合）"""

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.use_nosql = settings.use_nosql_for_conversations

        if self.use_nosql:
            self._initialize_nosql()
        else:
            self._initialize_json_storage()

    def _initialize_nosql(self):
        """NoSQL（MongoDB）の初期化"""
        try:
            # 実際の実装ではmotorを使用
            logger.info("NoSQL（MongoDB）会話ストレージを初期化しました")
        except Exception as e:
            logger.warning(f"NoSQL初期化エラー、JSONストレージにフォールバック: {e}")
            self.use_nosql = False
            self._initialize_json_storage()

    def _initialize_json_storage(self):
        """JSONファイルストレージの初期化"""
        import os
        self.storage_dir = "data/conversations"
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"JSON会話ストレージを初期化しました: {self.storage_dir}")

    async def create_session(self, customer_id: str, machine_id: str) -> str:
        """新しい会話セッションを作成"""
        session_id = f"{customer_id}_{machine_id}_{int(datetime.now().timestamp())}"

        session = ConversationSession(
            session_id=session_id,
            customer_id=customer_id,
            machine_id=machine_id,
            start_time=datetime.now()
        )

        self.sessions[session_id] = session

        if self.use_nosql:
            await self._save_session_nosql(session)
        else:
            self._save_session_json(session)

        logger.info(f"会話セッションを作成: {session_id}")
        return session_id

    async def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """メッセージを追加"""
        if session_id not in self.sessions:
            logger.warning(f"セッションが見つかりません: {session_id}")
            return

        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.sessions[session_id].messages.append(message)

        if self.use_nosql:
            await self._update_session_nosql(session_id)
        else:
            self._update_session_json(session_id)

        logger.debug(f"メッセージを追加: {session_id} - {role}")

    async def get_conversation_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """顧客の会話履歴を取得"""
        customer_sessions = [
            session for session in self.sessions.values()
            if session.customer_id == customer_id
        ]

        # 最新順にソート
        customer_sessions.sort(key=lambda s: s.start_time, reverse=True)

        conversations = []
        for session in customer_sessions[:limit]:
            formatted_conv = {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "message_count": len(session.messages),
                "context": session.context,
                "ai_insights": session.ai_insights,
                "summary": await self._generate_conversation_summary(session.messages)
            }
            conversations.append(formatted_conv)

        return conversations

    async def get_conversation_for_ai_agent(self, session_id: str) -> Dict[str, Any]:
        """AIエージェント用の会話データを取得"""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        # AIエージェントが処理しやすい形式に整形
        return {
            "session_id": session.session_id,
            "customer_context": session.context,
            "message_history": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in session.messages
            ],
            "previous_insights": session.ai_insights,
            "conversation_summary": await self._generate_conversation_summary(session.messages)
        }

    async def update_ai_insights(self, session_id: str, insights: Dict[str, Any]):
        """AI分析結果を更新"""
        if session_id not in self.sessions:
            logger.warning(f"セッションが見つかりません: {session_id}")
            return

        self.sessions[session_id].ai_insights.update(insights)

        if self.use_nosql:
            await self._update_session_nosql(session_id)
        else:
            self._update_session_json(session_id)

        logger.debug(f"AI分析結果を更新: {session_id}")

    async def search_conversations(self, query: str, customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """会話内容を検索"""
        results = []

        for session in self.sessions.values():
            if customer_id and session.customer_id != customer_id:
                continue

            # メッセージ内容で検索
            for message in session.messages:
                if query.lower() in message.content.lower():
                    results.append({
                        "session_id": session.session_id,
                        "customer_id": session.customer_id,
                        "message": message.content,
                        "timestamp": message.timestamp,
                        "role": message.role
                    })

        # タイムスタンプ順にソート
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    async def _generate_conversation_summary(self, messages: List[ConversationMessage]) -> str:
        """会話サマリを生成"""
        if not messages:
            return "会話なし"

        user_messages = [msg.content for msg in messages if msg.role == "user"]
        assistant_messages = [msg.content for msg in messages if msg.role == "assistant"]

        summary = f"ユーザー発言: {len(user_messages)}件, アシスタント応答: {len(assistant_messages)}件"

        if user_messages:
            # 最新のユーザー発言を追加
            summary += f", 最新発言: {user_messages[-1][:50]}..."

        return summary

    async def _save_session_nosql(self, session: ConversationSession):
        """NoSQLにセッションを保存"""
        try:
            # 実際の実装ではMongoDBに保存
            # await self.db.conversations.insert_one(session.dict())
            pass
        except Exception as e:
            logger.error(f"NoSQLセッション保存エラー: {e}")

    async def _update_session_nosql(self, session_id: str):
        """NoSQLのセッションを更新"""
        try:
            # 実際の実装ではMongoDBを更新
            # await self.db.conversations.update_one(...)
            pass
        except Exception as e:
            logger.error(f"NoSQLセッション更新エラー: {e}")

    def _save_session_json(self, session: ConversationSession):
        """JSONファイルにセッションを保存"""
        try:
            file_path = f"{self.storage_dir}/{session.session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": session.session_id,
                    "customer_id": session.customer_id,
                    "machine_id": session.machine_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "metadata": msg.metadata
                        }
                        for msg in session.messages
                    ],
                    "context": session.context,
                    "ai_insights": session.ai_insights,
                    "tags": session.tags
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSONセッション保存エラー: {e}")

    def _update_session_json(self, session_id: str):
        """JSONファイルのセッションを更新"""
        session = self.sessions.get(session_id)
        if session:
            self._save_session_json(session)

    def load_sessions_from_storage(self):
        """ストレージからセッションを読み込み"""
        if not self.use_nosql:
            try:
                import os
                for filename in os.listdir(self.storage_dir):
                    if filename.endswith('.json'):
                        file_path = f"{self.storage_dir}/{filename}"
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        session = ConversationSession(
                            session_id=data["session_id"],
                            customer_id=data["customer_id"],
                            machine_id=data["machine_id"],
                            start_time=datetime.fromisoformat(data["start_time"]),
                            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
                            messages=[
                                ConversationMessage(
                                    role=msg["role"],
                                    content=msg["content"],
                                    timestamp=datetime.fromisoformat(msg["timestamp"]),
                                    metadata=msg.get("metadata")
                                )
                                for msg in data["messages"]
                            ],
                            context=data.get("context", {}),
                            ai_insights=data.get("ai_insights", {}),
                            tags=data.get("tags", [])
                        )

                        self.sessions[session.session_id] = session

                logger.info(f"JSONストレージから{len(self.sessions)}件のセッションを読み込みました")
            except Exception as e:
                logger.error(f"セッションロードエラー: {e}")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """会話統計を取得"""
        if not self.sessions:
            return {"total_sessions": 0, "total_messages": 0}

        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())

        # 顧客別統計
        customer_stats = {}
        for session in self.sessions.values():
            customer_id = session.customer_id
            if customer_id not in customer_stats:
                customer_stats[customer_id] = {"sessions": 0, "messages": 0}
            customer_stats[customer_id]["sessions"] += 1
            customer_stats[customer_id]["messages"] += len(session.messages)

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / total_sessions,
            "customer_stats": customer_stats,
            "storage_type": "nosql" if self.use_nosql else "json"
        }

# グローバルインスタンス
conversation_service = ConversationService()
