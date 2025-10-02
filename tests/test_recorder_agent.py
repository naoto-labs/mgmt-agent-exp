"""
RecorderAgentのテスト
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.recorder_agent import (
    BusinessOutcomeRecord,
    ManagementActionRecord,
    RecorderAgent,
)


class TestRecorderAgent:
    """RecorderAgentのテストクラス"""

    @pytest.fixture
    def agent(self):
        """テスト用のRecorderAgentインスタンスを作成"""
        with patch("src.agents.recorder_agent.OpenAIEmbeddings"):
            with patch("src.agents.recorder_agent.Chroma"):
                agent = RecorderAgent(persist_directory="./test_data/vector_store")
                # モックストアを設定
                agent.action_store = MagicMock()
                agent.decision_store = MagicMock()
                agent.outcome_store = MagicMock()
                return agent

    def test_initialization(self, agent):
        """初期化のテスト"""
        assert agent is not None
        assert agent.persist_directory == "./test_data/vector_store"

    @pytest.mark.asyncio
    async def test_record_action(self, agent):
        """行動記録のテスト"""
        action_record = ManagementActionRecord(
            record_id="action_001",
            session_id="session_001",
            timestamp=datetime.now(),
            action_type="decision",
            context={"sales": 100000},
            decision_process="在庫分析に基づき価格調整を決定",
            executed_action="価格を10%値上げ",
            expected_outcome="利益率5%向上",
        )

        # add_textsメソッドをモック
        agent.action_store.add_texts = MagicMock()

        result = await agent.record_action(action_record)

        assert result["success"] is True
        assert result["record_id"] == "action_001"
        agent.action_store.add_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_outcome(self, agent):
        """結果記録のテスト"""
        outcome_record = BusinessOutcomeRecord(
            record_id="outcome_001",
            session_id="session_001",
            related_action_id="action_001",
            timestamp=datetime.now(),
            outcome_type="sales",
            metrics={"sales": 110000, "profit_margin": 0.40},
            success_level="excellent",
            lessons_learned=["価格調整が効果的だった", "顧客満足度も維持できた"],
        )

        # add_textsメソッドをモック
        agent.outcome_store.add_texts = MagicMock()

        result = await agent.record_outcome(outcome_record)

        assert result["success"] is True
        assert result["record_id"] == "outcome_001"
        agent.outcome_store.add_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_actions(self, agent):
        """類似行動検索のテスト"""
        # モックの検索結果を設定
        mock_doc = MagicMock()
        mock_doc.page_content = "テスト行動"
        mock_doc.metadata = {"action_type": "decision"}

        agent.action_store.similarity_search = MagicMock(return_value=[mock_doc])

        results = await agent.search_similar_actions("価格調整", k=5)

        assert len(results) == 1
        assert "content" in results[0]
        assert "metadata" in results[0]
        agent.action_store.similarity_search.assert_called_once_with("価格調整", k=5)

    @pytest.mark.asyncio
    async def test_search_similar_outcomes(self, agent):
        """類似結果検索のテスト"""
        # モックの検索結果を設定
        mock_doc = MagicMock()
        mock_doc.page_content = "テスト結果"
        mock_doc.metadata = {"success_level": "excellent"}

        agent.outcome_store.similarity_search = MagicMock(return_value=[mock_doc])

        results = await agent.search_similar_outcomes("売上向上", k=5)

        assert len(results) == 1
        assert "content" in results[0]
        assert "metadata" in results[0]
        agent.outcome_store.similarity_search.assert_called_once_with("売上向上", k=5)

    @pytest.mark.asyncio
    async def test_extract_successful_patterns(self, agent):
        """成功パターン抽出のテスト"""
        # モックデータを設定
        mock_outcomes = [
            {
                "content": "成功事例1",
                "metadata": {
                    "session_id": "session_001",
                    "success_level": "excellent",
                },
            },
            {
                "content": "成功事例2",
                "metadata": {"session_id": "session_002", "success_level": "good"},
            },
        ]

        agent.search_similar_outcomes = AsyncMock(return_value=mock_outcomes)

        patterns = await agent.extract_successful_patterns("morning_routine")

        assert len(patterns) == 2
        assert patterns[0]["success_level"] == "excellent"
        assert patterns[1]["success_level"] == "good"

    @pytest.mark.asyncio
    async def test_extract_failure_lessons(self, agent):
        """失敗教訓抽出のテスト"""
        # モックデータを設定
        mock_outcomes = [
            {
                "content": "失敗事例1",
                "metadata": {"session_id": "session_003", "success_level": "poor"},
            },
            {
                "content": "失敗事例2",
                "metadata": {"session_id": "session_004", "success_level": "average"},
            },
        ]

        agent.search_similar_outcomes = AsyncMock(return_value=mock_outcomes)

        lessons = await agent.extract_failure_lessons("morning_routine")

        assert len(lessons) == 2
        assert lessons[0]["success_level"] == "poor"
        assert lessons[1]["success_level"] == "average"

    @pytest.mark.asyncio
    async def test_analyze_session_patterns(self, agent):
        """セッションパターン分析のテスト"""
        # モックデータを設定
        agent.extract_successful_patterns = AsyncMock(
            return_value=[{"session_id": "s1", "success_level": "excellent"}]
        )
        agent.extract_failure_lessons = AsyncMock(
            return_value=[{"session_id": "s2", "success_level": "poor"}]
        )

        # LLMのモック
        agent.analysis_llm.apredict = AsyncMock(return_value="洞察1\n洞察2\n洞察3")

        analysis = await agent.analyze_session_patterns("morning_routine")

        assert analysis["session_type"] == "morning_routine"
        assert analysis["successful_count"] == 1
        assert analysis["failure_count"] == 1
        assert analysis["success_rate"] == 0.5
        assert len(analysis["key_insights"]) > 0

    @pytest.mark.asyncio
    async def test_generate_session_recommendations(self, agent):
        """推奨事項生成のテスト"""
        # モックデータを設定
        agent.analyze_session_patterns = AsyncMock(
            return_value={
                "session_type": "morning_routine",
                "success_rate": 0.75,
                "key_insights": ["洞察1", "洞察2"],
            }
        )

        # LLMのモック
        agent.analysis_llm.apredict = AsyncMock(
            return_value="推奨1\n推奨2\n推奨3\n推奨4\n推奨5"
        )

        recommendations = await agent.generate_session_recommendations(
            "morning_routine"
        )

        assert len(recommendations) <= 5
        assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_update_daily_experience(self, agent):
        """日次経験更新のテスト"""
        daily_results = {
            "morning": {
                "session_id": "morning_001",
                "metrics": {"sales": 50000},
                "lessons_learned": ["朝は売上が好調"],
            },
            "evening": {
                "session_id": "evening_001",
                "metrics": {"sales": 100000},
                "lessons_learned": ["夕方も好調を維持"],
            },
        }

        agent.record_outcome = AsyncMock()

        await agent.update_daily_experience(daily_results)

        # 2回record_outcomeが呼ばれることを確認
        assert agent.record_outcome.call_count == 2

    @pytest.mark.asyncio
    async def test_record_action_without_store(self):
        """ストアがない場合の行動記録"""
        agent = RecorderAgent()
        agent.action_store = None

        action_record = ManagementActionRecord(
            record_id="action_001",
            session_id="session_001",
            timestamp=datetime.now(),
            action_type="decision",
            context={},
            decision_process="テスト",
            executed_action="テスト",
            expected_outcome="テスト",
        )

        result = await agent.record_action(action_record)

        assert result["success"] is False
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_search_without_store(self):
        """ストアがない場合の検索"""
        agent = RecorderAgent()
        agent.action_store = None

        results = await agent.search_similar_actions("テスト")

        assert len(results) == 0


class TestManagementActionRecord:
    """ManagementActionRecordモデルのテスト"""

    def test_action_record_creation(self):
        """行動記録の作成テスト"""
        record = ManagementActionRecord(
            record_id="action_001",
            session_id="session_001",
            timestamp=datetime.now(),
            action_type="decision",
            context={"key": "value"},
            decision_process="プロセス",
            executed_action="実行",
            expected_outcome="期待結果",
        )

        assert record.record_id == "action_001"
        assert record.session_id == "session_001"
        assert record.action_type == "decision"
        assert record.actual_outcome is None
        assert record.success_score is None


class TestBusinessOutcomeRecord:
    """BusinessOutcomeRecordモデルのテスト"""

    def test_outcome_record_creation(self):
        """結果記録の作成テスト"""
        record = BusinessOutcomeRecord(
            record_id="outcome_001",
            session_id="session_001",
            related_action_id="action_001",
            timestamp=datetime.now(),
            outcome_type="sales",
            metrics={"sales": 100000},
            success_level="excellent",
            lessons_learned=["教訓1", "教訓2"],
        )

        assert record.record_id == "outcome_001"
        assert record.session_id == "session_001"
        assert record.outcome_type == "sales"
        assert record.success_level == "excellent"
        assert len(record.lessons_learned) == 2
