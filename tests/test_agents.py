"""
新しいorchestratorアーキテクチャの統合テスト
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# New orchestrator imports
from src.agents.analytics_agent.advisory.orchestrator import analytics_orchestrator
from src.agents.management_agent.orchestrator import management_agent
from src.agents.recorder_agent.orchestrator import recorder_orchestrator


class TestManagementAgentOrchestrator:
    """Management Agent Orchestratorのテスト"""

    @pytest.mark.asyncio
    async def test_morning_routine(self):
        """朝の業務ルーチンテスト"""
        with patch.object(
            management_agent, "make_strategic_decision", new_callable=AsyncMock
        ) as mock_make_decision:
            mock_make_decision.return_value = {
                "decision": "在庫を最適化する",
                "rationale": "データに基づく判断",
                "actions": ["価格更新", "補充依頼"],
            }

            session_id = await management_agent.start_management_session(
                "morning_routine"
            )
            await management_agent.start_management_session("morning_routine")

            result = await management_agent.morning_routine()

            assert "session_id" in result
            assert "decisions" in result
            assert result["status"] == "completed"

            # セッション終了
            await management_agent.end_management_session()

    @pytest.mark.asyncio
    async def test_evening_summary(self):
        """夕方の業務総括テスト"""
        with patch.object(
            management_agent, "make_strategic_decision", new_callable=AsyncMock
        ) as mock_make_decision:
            mock_make_decision.return_value = {
                "decision": "1日の振り返りを実行",
                "rationale": "業績分析に基づく",
                "actions": ["レポート生成", "改善策立案"],
            }

            result = await management_agent.evening_summary()

            assert "daily_performance" in result
            assert "decisions" in result
            assert "lessons_learned" in result
            assert result["status"] == "completed"


class TestAnalyticsAgentOrchestrator:
    """Analytics Agent Orchestratorのテスト"""

    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self):
        """包括的分析テスト"""
        result = await analytics_orchestrator.run_comprehensive_analysis()

        assert "timestamp" in result
        assert "analysis_results" in result
        assert "session_type" in result
        assert result["session_type"] == "comprehensive_analysis"

    @pytest.mark.asyncio
    async def test_real_time_monitoring(self):
        """リアルタイム監視テスト"""
        result = await analytics_orchestrator.run_real_time_monitoring()

        assert "performance_health" in result
        assert "safety_status" in result
        assert "alerts" in result
        assert result["status"] == "monitoring_active"


class TestRecorderAgentOrchestrator:
    """Recorder Agent Orchestratorのテスト"""

    @pytest.mark.asyncio
    async def test_record_session_data(self):
        """セッションデータ記録テスト"""
        test_data = {
            "session_id": "test_session_123",
            "actions_taken": 3,
            "decisions_made": 2,
        }

        with patch(
            "src.agents.recorder_agent.learning_tools.session_recorder.record_session",
            new_callable=AsyncMock,
        ) as mock_record:
            mock_record.return_value = {"success": True, "recorded_data": test_data}

            result = await recorder_orchestrator.record_session_data(test_data)

            assert result["success"] is True
            assert result["recorded_data"]["session_id"] == "test_session_123"

    @pytest.mark.asyncio
    async def test_run_data_maintenance(self):
        """データメンテナンステスト"""
        result = await recorder_orchestrator.run_data_maintenance()

        assert "status" in result
        assert result["status"] == "maintenance_completed"
        assert "timestamp" in result


class TestAgentOrchestratorIntegration:
    """Agent Orchestrator統合テスト"""

    @pytest.mark.asyncio
    async def test_management_and_analytics_flow(self):
        """Management Agent → Analytics Agent連携テスト"""
        # Managementセッション実行
        with patch.object(
            management_agent, "make_strategic_decision", new_callable=AsyncMock
        ) as mock_make_decision:
            mock_make_decision.return_value = {
                "decision": "在庫戦略の調整",
                "rationale": "分析結果に基づく",
                "actions": ["在庫補充"],
            }

            session_result = await management_agent.morning_routine()
            session_data = session_result

            # Analytics分析実行
            analysis_result = await analytics_orchestrator.run_comprehensive_analysis()

            # Recorderデータ記録
            record_result = await recorder_orchestrator.record_session_data(
                session_data
            )

            # 統合結果確認
            assert session_result["status"] == "completed"
            assert "timestamp" in analysis_result
            assert record_result["success"] is True

    @pytest.mark.asyncio
    async def test_full_agent_workflow(self):
        """完全なAgentワークフローテスト"""
        # 朝のセッション
        morning_session = await management_agent.start_management_session(
            "morning_routine"
        )

        # 分析実行
        analytics_result = await analytics_orchestrator.run_real_time_monitoring()

        # 夕方の総括
        evening_result = await management_agent.evening_summary()

        # 記録
        log_result = await recorder_orchestrator.record_session_data(
            {"day": "test_day", "performance": evening_result["daily_performance"]}
        )

        assert morning_session is not None
        assert "performance_health" in analytics_result
        assert "daily_performance" in evening_result
        assert log_result["success"] is True


# パフォーマンステスト
@pytest.mark.asyncio
async def test_orchestrator_performance():
    """Orchestratorパフォーマンステスト"""
    import time

    start_time = time.time()

    # 複数の分析を実行
    tasks = [
        analytics_orchestrator.run_real_time_monitoring(),
        analytics_orchestrator.run_real_time_monitoring(),
        recorder_orchestrator.run_data_maintenance(),
    ]

    results = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    # 各タスクが完了しているか確認
    assert len(results) == 3
    assert all("timestamp" in result or "status" in result for result in results)

    # 所要時間が適当か確認（5秒以内）
    assert elapsed_time < 5.0


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
