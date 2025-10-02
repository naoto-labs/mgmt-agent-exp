"""
SessionBasedManagementAgentのテスト
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.management_agent import (
    BusinessMetrics,
    SessionBasedManagementAgent,
    SessionInfo,
)


class TestSessionBasedManagementAgentAnthropic:
    """Anthropicプロバイダー使用のSessionBasedManagementAgentテストクラス"""

    @pytest.fixture
    def agent(self):
        """テスト用のAgentインスタンスを作成"""
        with patch.object(SessionBasedManagementAgent, "_verify_llm_connection"):
            agent = SessionBasedManagementAgent(provider="anthropic")
            return agent


class TestSessionBasedManagementAgentOpenAI:
    """OpenAIプロバイダー使用のSessionBasedManagementAgentテストクラス"""

    @pytest.fixture
    def openai_agent(self):
        """テスト用のOpenAI Agentインスタンスを作成"""
        # model_managerのgenerate_responseをモックしてOpenAIレスポンスをシミュレート
        from unittest.mock import AsyncMock

        with patch(
            "src.ai.model_manager.model_manager.generate_response",
            new_callable=AsyncMock,
        ) as mock_generate:
            from src.ai.model_manager import AIResponse

            mock_response = AIResponse(
                content='{"decision": "OpenAIテスト決定", "rationale": "OpenAIによる分析", "actions": ["価格調整", "在庫補充"]}',
                model_used="openai",
                tokens_used=120,
                response_time=0.8,
                success=True,
            )
            mock_generate.return_value = mock_response
            agent = SessionBasedManagementAgent(provider="openai")
            yield agent

    @pytest.mark.asyncio
    async def test_openai_strategic_decision(self, openai_agent):
        """OpenAIを使用した戦略的意思決定テスト"""
        session_id = await openai_agent.start_management_session("test_session")
        assert openai_agent.current_session is not None

        decision = await openai_agent.make_strategic_decision(
            "売上向上のための戦略立案"
        )

        assert decision["decision"] == "OpenAIテスト決定"
        assert decision["rationale"] == "OpenAIによる分析"
        assert "価格調整" in decision["actions"]

        await openai_agent.end_management_session()

    @pytest.mark.asyncio
    async def test_openai_morning_routine(self, openai_agent):
        """OpenAIを使用した朝のルーチンテスト"""
        result = await openai_agent.morning_routine()

        assert result["session_type"] == "morning_routine"
        assert result["decisions"]["decision"] == "OpenAIテスト決定"
        assert result["status"] == "completed"

    def test_openai_business_metrics(self, openai_agent):
        """OpenAIプロバイダーのビジネスメトリクス取得テスト"""
        metrics = openai_agent.get_business_metrics()

        assert "sales" in metrics
        assert "profit_margin" in metrics
        assert isinstance(metrics["sales"], (int, float))


class TestSessionBasedManagementAgent:
    """全般的なSessionBasedManagementAgentのテストクラスの別名（後方互換用）"""

    # 既存のanthropicテストを保持するために、ラッパークラスとして使用

    @pytest.fixture
    def agent(self):
        """テスト用のAgentインスタンスを作成（anthropic）"""
        with patch.object(SessionBasedManagementAgent, "_verify_llm_connection"):
            agent = SessionBasedManagementAgent(provider="anthropic")
            return agent

    def test_initialization(self, agent):
        """初期化のテスト"""
        assert agent is not None
        assert agent.provider == "anthropic"
        assert agent.current_session is None
        assert len(agent.tools) > 0

    def test_get_business_metrics(self, agent):
        """ビジネスメトリクス取得のテスト"""
        metrics = agent.get_business_metrics()

        assert "sales" in metrics
        assert "profit_margin" in metrics
        assert "inventory_level" in metrics
        assert "customer_satisfaction" in metrics
        assert isinstance(metrics["sales"], (int, float))
        assert isinstance(metrics["inventory_level"], dict)

    def test_analyze_financial_performance(self, agent):
        """財務分析のテスト"""
        analysis = agent.analyze_financial_performance()

        assert "analysis" in analysis
        assert "recommendations" in analysis
        assert "metrics" in analysis
        assert isinstance(analysis["recommendations"], list)

    def test_check_inventory_status(self, agent):
        """在庫状況確認のテスト"""
        status = agent.check_inventory_status()

        assert "status" in status
        assert "low_stock_items" in status
        assert "reorder_needed" in status
        assert isinstance(status["low_stock_items"], list)

    def test_update_pricing_strategy(self, agent):
        """価格戦略更新のテスト"""
        result = agent.update_pricing_strategy("cola", 150.0)

        assert result["success"] is True
        assert result["product"] == "cola"
        assert result["new_price"] == 150.0
        assert "effective_date" in result

    def test_assign_restocking_task(self, agent):
        """補充タスク割り当てのテスト"""
        result = agent.assign_restocking_task(["water", "juice"], urgency="urgent")

        assert "task_id" in result
        assert result["task_type"] == "restocking"
        assert result["products"] == ["water", "juice"]
        assert result["urgency"] == "urgent"
        assert result["assigned"] is True

    def test_request_procurement(self, agent):
        """調達依頼のテスト"""
        result = agent.request_procurement(["water"], quantity={"water": 100})

        assert "order_id" in result
        assert result["products"] == ["water"]
        assert result["quantity"] == {"water": 100}
        assert result["status"] == "pending"

    def test_respond_to_customer_inquiry(self, agent):
        """顧客問い合わせ対応のテスト"""
        result = agent.respond_to_customer_inquiry(
            "customer_001", "商品の賞味期限について"
        )

        assert result["customer_id"] == "customer_001"
        assert result["inquiry"] == "商品の賞味期限について"
        assert "response" in result
        assert result["status"] == "responded"

    def test_handle_customer_complaint(self, agent):
        """顧客苦情処理のテスト"""
        result = agent.handle_customer_complaint("customer_002", "商品が出てこなかった")

        assert result["customer_id"] == "customer_002"
        assert result["complaint"] == "商品が出てこなかった"
        assert "resolution" in result
        assert result["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_start_management_session(self, agent):
        """セッション開始のテスト"""
        session_id = await agent.start_management_session("morning_routine")

        assert session_id is not None
        assert agent.current_session is not None
        assert agent.current_session.session_type == "morning_routine"
        assert agent.current_session.session_id == session_id

    @pytest.mark.asyncio
    async def test_end_management_session(self, agent):
        """セッション終了のテスト"""
        # セッションを開始
        await agent.start_management_session("morning_routine")

        # セッションを終了
        summary = await agent.end_management_session()

        assert "session_id" in summary
        assert "session_type" in summary
        assert summary["session_type"] == "morning_routine"
        assert "duration" in summary
        assert agent.current_session is None

    @pytest.mark.asyncio
    async def test_end_session_without_start_raises_error(self, agent):
        """セッション未開始で終了するとエラー"""
        with pytest.raises(ValueError, match="No active session"):
            await agent.end_management_session()

    @pytest.mark.asyncio
    async def test_make_strategic_decision(self, agent):
        """戦略的意思決定のテスト"""
        # セッションを開始
        await agent.start_management_session("morning_routine")

        # 意思決定
        context = "在庫が少なくなっています"
        decision = await agent.make_strategic_decision(context)

        assert "context" in decision
        assert "decision" in decision
        assert "rationale" in decision
        assert "actions" in decision
        assert decision["context"] == context

        # セッションを終了
        await agent.end_management_session()

    @pytest.mark.asyncio
    async def test_make_decision_without_session_raises_error(self, agent):
        """セッション未開始で意思決定するとエラー"""
        with pytest.raises(ValueError, match="No active session"):
            await agent.make_strategic_decision("テストコンテキスト")

    @pytest.mark.asyncio
    async def test_morning_routine(self, agent):
        """朝のルーチンのテスト"""
        result = await agent.morning_routine()

        assert "session_id" in result
        assert result["session_type"] == "morning_routine"
        assert "overnight_data" in result
        assert "decisions" in result
        assert result["status"] == "completed"
        # セッションは自動的に終了している
        assert agent.current_session is None

    @pytest.mark.asyncio
    async def test_midday_check(self, agent):
        """昼のチェックのテスト"""
        result = await agent.midday_check()

        assert "session_id" in result
        assert result["session_type"] == "midday_check"
        assert "metrics" in result
        assert "analysis" in result
        assert "decisions" in result
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_evening_summary(self, agent):
        """夕方の総括のテスト"""
        result = await agent.evening_summary()

        assert "session_id" in result
        assert result["session_type"] == "evening_summary"
        assert "daily_performance" in result
        assert "inventory_status" in result
        assert "decisions" in result
        assert "lessons_learned" in result
        assert result["status"] == "completed"

    def test_tools_creation(self, agent):
        """ツール作成のテスト"""
        # システム連携ツール
        system_tools = agent._create_system_integration_tools()
        assert len(system_tools) > 0
        assert any(tool.name == "get_business_data" for tool in system_tools)

        # 人間協働ツール
        human_tools = agent._create_human_collaboration_tools()
        assert len(human_tools) > 0
        assert any(tool.name == "assign_restocking" for tool in human_tools)

        # 顧客対応ツール
        customer_tools = agent._create_customer_service_tools()
        assert len(customer_tools) > 0
        assert any(tool.name == "customer_response" for tool in customer_tools)

    @pytest.mark.asyncio
    async def test_session_flow(self, agent):
        """完全なセッションフローのテスト"""
        # 1. セッション開始
        session_id = await agent.start_management_session("test_session")
        assert agent.current_session is not None

        # 2. ビジネスデータ取得
        metrics = agent.get_business_metrics()
        assert metrics is not None

        # 3. 意思決定
        decision = await agent.make_strategic_decision("テストコンテキスト")
        assert decision is not None
        assert len(agent.current_session.decisions_made) == 1

        # 4. セッション終了
        summary = await agent.end_management_session()
        assert summary["decisions_count"] == 1
        assert agent.current_session is None


class TestBusinessMetrics:
    """BusinessMetricsモデルのテスト"""

    def test_business_metrics_creation(self):
        """ビジネスメトリクスの作成テスト"""
        metrics = BusinessMetrics(
            sales=100000.0,
            profit_margin=0.3,
            inventory_level={"cola": 50, "water": 30},
            customer_satisfaction=4.5,
            timestamp=datetime.now(),
        )

        assert metrics.sales == 100000.0
        assert metrics.profit_margin == 0.3
        assert len(metrics.inventory_level) == 2
        assert metrics.customer_satisfaction == 4.5


class TestSessionInfo:
    """SessionInfoモデルのテスト"""

    def test_session_info_creation(self):
        """セッション情報の作成テスト"""
        session = SessionInfo(
            session_id="test_123",
            session_type="morning_routine",
            start_time=datetime.now(),
        )

        assert session.session_id == "test_123"
        assert session.session_type == "morning_routine"
        assert session.end_time is None
        assert len(session.decisions_made) == 0
        assert len(session.actions_executed) == 0
