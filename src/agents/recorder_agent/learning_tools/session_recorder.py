"""
session_recorder.py - セッション記録ツール

Agentの意思決定・行動・結果を詳細ログ化・学習データ蓄積Tool
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SessionData:
    """セッションデータ構造"""

    def __init__(
        self,
        session_id: str,
        agent_type: str,
        session_type: str,
        start_time: datetime,
        context: Dict[str, Any],
    ):
        self.session_id = session_id
        self.agent_type = agent_type
        self.session_type = session_type
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.context = context
        self.decisions: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.outcomes: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "session_type": self.session_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
            ),
            "context": self.context,
            "decisions": self.decisions,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "metadata": self.metadata,
        }


class SessionRecorder:
    """セッション記録マネージャー"""

    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self.completed_sessions: List[SessionData] = []
        logger.info("SessionRecorder initialized")

    def start_session(
        self,
        agent_type: str,
        session_type: str,
        context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """セッション開始"""
        if session_id is None:
            session_id = f"{agent_type}_{session_type}_{uuid4().hex[:8]}"

        session = SessionData(
            session_id=session_id,
            agent_type=agent_type,
            session_type=session_type,
            start_time=datetime.now(),
            context=context,
        )

        self.active_sessions[session_id] = session
        logger.info(f"Started session recording: {session_id}")
        return session_id

    def record_decision(
        self,
        session_id: str,
        decision_data: Dict[str, Any],
    ) -> bool:
        """決定内容を記録"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # 決定データを標準化
        standardized_decision = {
            "timestamp": datetime.now().isoformat(),
            "decision_id": f"{session_id}_decision_{len(session.decisions)}",
            "content": decision_data.get("decision", ""),
            "rationale": decision_data.get("rationale", ""),
            "confidence_score": decision_data.get("confidence_score", 0.5),
            "alternatives_considered": decision_data.get("alternatives", []),
            "raw_data": decision_data,
        }

        session.decisions.append(standardized_decision)

        logger.debug(
            f"Recorded decision for session {session_id}: {len(session.decisions)} decisions"
        )
        return True

    def record_action(
        self,
        session_id: str,
        action_data: Dict[str, Any],
    ) -> bool:
        """行動内容を記録"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # 行動データを標準化
        standardized_action = {
            "timestamp": datetime.now().isoformat(),
            "action_id": f"{session_id}_action_{len(session.actions)}",
            "action_type": action_data.get("type", "unknown"),
            "content": action_data.get("action", ""),
            "parameters": action_data.get("parameters", {}),
            "expected_impact": action_data.get("expected_impact", "unknown"),
            "raw_data": action_data,
        }

        session.actions.append(standardized_action)

        logger.debug(
            f"Recorded action for session {session_id}: {len(session.actions)} actions"
        )
        return True

    def record_outcome(
        self,
        session_id: str,
        outcome_data: Dict[str, Any],
    ) -> bool:
        """結果内容を記録"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # 結果データを標準化
        standardized_outcome = {
            "timestamp": datetime.now().isoformat(),
            "outcome_id": f"{session_id}_outcome_{len(session.outcomes)}",
            "outcome_type": outcome_data.get("type", "unknown"),
            "success": outcome_data.get("success", False),
            "metrics": outcome_data.get("metrics", {}),
            "feedback": outcome_data.get("feedback", ""),
            "lessons_learned": outcome_data.get("lessons", []),
            "raw_data": outcome_data,
        }

        session.outcomes.append(standardized_outcome)

        logger.debug(
            f"Recorded outcome for session {session_id}: {len(session.outcomes)} outcomes"
        )
        return True

    def end_session(
        self,
        session_id: str,
        final_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionData]:
        """セッション終了"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found for ending: {session_id}")
            return None

        session.end_time = datetime.now()
        if final_metadata:
            session.metadata.update(final_metadata)

        # アクティブから完了済みに移動
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]

        duration = (session.end_time - session.start_time).total_seconds()
        logger.info(
            f"Ended session recording: {session_id} "
            f"(duration: {duration:.1f}s, decisions: {len(session.decisions)}, "
            f"actions: {len(session.actions)}, outcomes: {len(session.outcomes)})"
        )

        return session

    def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """セッション取得"""
        return self.active_sessions.get(session_id) or next(
            (s for s in self.completed_sessions if s.session_id == session_id), None
        )

    def get_all_sessions(self, agent_type: Optional[str] = None) -> List[SessionData]:
        """全セッション取得 (フィルタリング可能)"""
        all_sessions = list(self.active_sessions.values()) + self.completed_sessions

        if agent_type:
            return [s for s in all_sessions if s.agent_type == agent_type]

        return all_sessions

    def get_recent_sessions(self, limit: int = 10) -> List[SessionData]:
        """最近のセッション取得"""
        all_sessions = self.get_all_sessions()
        # 終了時刻でソート (アクティブは現在時刻として扱う)
        for session in all_sessions:
            if session.end_time is None:
                session.end_time = datetime.now()

        sorted_sessions = sorted(
            all_sessions, key=lambda s: s.end_time or datetime.min, reverse=True
        )

        return sorted_sessions[:limit]

    def get_learning_quality_score(self, session: SessionData) -> float:
        """学習品質スコア計算 (再利用性・完全性を基準に)"""
        score_components = []

        # 決定の詳細度
        if session.decisions:
            avg_decisions_quality = sum(
                1 if d.get("rationale") else 0.5 for d in session.decisions
            ) / len(session.decisions)
            score_components.append(avg_decisions_quality)

        # 行動の追跡可能性
        if session.actions:
            avg_actions_quality = sum(
                1 if a.get("expected_impact", "unknown") != "unknown" else 0.5
                for a in session.actions
            ) / len(session.actions)
            score_components.append(avg_actions_quality)

        # 結果の学習価値
        if session.outcomes:
            avg_outcomes_quality = sum(
                1 if o.get("lessons_learned") else 0.7 for o in session.outcomes
            ) / len(session.outcomes)
            score_components.append(avg_outcomes_quality)

        return (
            sum(score_components) / len(score_components) if score_components else 0.5
        )


# グローバルインスタンス
session_recorder = SessionRecorder()


def record_session_start(
    agent_type: str,
    session_type: str,
    context: Dict[str, Any],
) -> str:
    """セッション記録開始"""
    return session_recorder.start_session(agent_type, session_type, context)


def record_decision(
    session_id: str,
    decision_data: Dict[str, Any],
) -> bool:
    """決定記録"""
    return session_recorder.record_decision(session_id, decision_data)


def record_action(
    session_id: str,
    action_data: Dict[str, Any],
) -> bool:
    """行動記録"""
    return session_recorder.record_action(session_id, action_data)


def record_outcome(
    session_id: str,
    outcome_data: Dict[str, Any],
) -> bool:
    """結果記録"""
    return session_recorder.record_outcome(session_id, outcome_data)


def record_session_end(
    session_id: str,
    final_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[SessionData]:
    """セッション記録終了"""
    return session_recorder.end_session(session_id, final_metadata)
