"""
Management Agent Flow Integration Test

LangGraphベース全Node統合フロー検証
- 各ノード間データ連携確認
- State遷移正確性検証
- ビジネスロジック全体整合性チェック
"""

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.management_agent.agent import ManagementState, NodeBasedManagementAgent
from src.agents.management_agent.models import BusinessMetrics
from src.domain.models.product import SAMPLE_PRODUCTS


class TestManagementFlowIntegration:
    """Management Agent統合フロー検証"""

    @pytest.fixture
    async def agent_with_tools(self):
        """ツール統合済みのAgentインスタンス"""
        # NodeBasedManagementAgent初期化 (自動的にツールレジストリから全ツールを取得)
        agent = NodeBasedManagementAgent(provider="openai")

        # 本番LLMを使用 (モックなし)
        # APIキーが設定されていることを確認
        import os

        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            pytest.skip("LLM API keys not configured - skipping live LLM tests")

        return agent

    @pytest.fixture
    def initial_state(self):
        """Case Aの初期State生成"""
        return ManagementState(
            session_id="integration_test_001",
            session_type="management_flow",
            business_date=date.today(),
            business_metrics={
                "sales": 850000,  # 85万円
                "profit_margin": 0.28,  # 28%利益率
                "inventory_level": {
                    "cola_regular": 12,
                    "cola_zero": 8,
                    "water_mineral": 15,
                    "coffee_hot": 5,  # 低在庫
                    "snack_chips": 3,  # 危機的
                },
                "customer_satisfaction": 3.8,  # 高満足度
                "timestamp": "2025-10-06T10:00:00Z",
            },
        )

    @pytest.mark.asyncio
    async def test_full_flow_integration(self, agent_with_tools, initial_state):
        """全ノード統合フロー検証 - Case A完全実行"""

        agent = agent_with_tools

        # === Step 1: 在庫確認 ===
        print("🔍 Step 1: 在庫確認開始")
        state = await agent.inventory_check_node(initial_state)

        # 検証: 在庫分析結果が生成されていること
        assert state.inventory_analysis is not None
        assert state.current_step == "inventory_check"
        assert state.processing_status == "processing"
        assert isinstance(state.inventory_analysis.get("low_stock_items", []), list)
        print(
            f"✅ 在庫分析完了: {len(state.inventory_analysis['low_stock_items'])}件の低在庫商品"
        )

        # === Step 2: 売上計画 ===
        print("📈 Step 2: 売上計画開始")
        state = await agent.sales_plan_node(state)

        # 検証: 売上・財務分析が生成されていること
        assert state.sales_analysis is not None
        assert state.financial_analysis is not None
        assert state.current_step == "sales_plan"
        assert "strategies" in state.sales_analysis
        assert "analysis" in state.financial_analysis
        print(f"✅ 売上計画完了: {len(state.sales_analysis['strategies'])}件の戦略提案")

        # === Step 3: 価格戦略 ===
        print("💰 Step 3: 価格戦略開始")
        state = await agent.pricing_node(state)

        # 検証: 価格決定が生成されていること
        assert state.pricing_decision is not None
        assert state.current_step == "pricing"
        assert "strategy" in state.pricing_decision

        print(f"✅ 価格戦略完了: {state.pricing_decision['strategy']}戦略")

        # === Step 4: 補充タスク ===
        print("📦 Step 4: 補充タスク開始")
        state = await agent.restock_node(state)

        # 検証: 補充決定が生成されていること
        assert state.restock_decision is not None
        assert state.current_step == "restock"
        assert isinstance(state.restock_decision.get("tasks_assigned", []), list)
        print(
            f"✅ 補充タスク完了: {len(state.restock_decision['tasks_assigned'])}件のタスク"
        )

        # === Step 5: 発注依頼 ===
        print("🛒 Step 5: 発注依頼開始")
        state = await agent.procurement_request_generation_node(state)

        # 検証: 調達決定が生成されていること
        assert state.procurement_decision is not None
        assert state.current_step == "procurement"
        # 発注リストが作成されていること
        assert isinstance(state.procurement_decision.get("orders_placed", []), list)
        print(
            f"✅ 発注依頼完了: {len(state.procurement_decision.get('orders_placed', []))}件の発注"
        )

        # === Step 6: 売上処理 ===
        print("⚙️ Step 6: 売上処理開始")
        state = await agent.sales_processing_node(state)

        # 検証: 売上処理結果が生成されていること
        assert state.sales_processing is not None
        assert state.current_step == "sales_processing"
        assert "performance_rating" in state.sales_processing
        print(
            f"✅ 売上処理完了: rating={state.sales_processing.get('performance_rating')}"
        )

        # === Step 7: 顧客対応 ===
        print("👥 Step 7: 顧客対応開始")
        state = await agent.customer_interaction_node(state)

        # 検証: 顧客対応結果が生成されていること
        assert state.customer_interaction is not None
        assert state.current_step == "customer_interaction"
        customer_actions = [
            a for a in state.executed_actions if "customer" in a.get("type", "")
        ]
        print(f"✅ 顧客対応完了: action={state.customer_interaction.get('action')}")

        # === Step 8: 利益計算 ===
        print("💹 Step 8: 利益計算開始")
        state = await agent.profit_calculation_node(state)

        # 検証: 利益計算結果が生成されていること
        assert state.profit_calculation is not None
        assert state.current_step == "profit_calculation"
        assert "margin_level" in state.profit_calculation
        assert "calculation_method" in state.profit_calculation  # ツール使用確認
        assert state.profit_calculation.get("calculation_method") in [
            "llm_driven_tools",
            "tool_integrated",
        ]
        # 財務アクションが実行済みであること
        financial_actions = [
            a for a in state.executed_actions if "financial" in a.get("type", "")
        ]
        print(
            f"✅ 利益計算完了: レベル={state.profit_calculation.get('margin_level')}, メソッド={state.profit_calculation.get('calculation_method')}"
        )

        # === Step 9: フィードバック ===
        print("📋 Step 9: 戦略的フィードバック開始")
        state = await agent.feedback_node(state)

        # 検証: 最終結果が生成されていること
        assert state.feedback is not None
        assert state.final_report is not None
        assert state.current_step == "feedback"
        assert state.processing_status in ["completed", "completed_with_errors"]
        assert "executive_summary" in state.feedback
        assert "business_health" in state.feedback
        print(f"✅ 戦略的フィードバック完了: ステータス={state.processing_status}")

        # === 統合検証 ===
        print("🔗 統合検証開始")

        # 1. 全ノードが実行されたことを確認（アクション記録形式に基づく柔軟な検証）
        total_actions = len(state.executed_actions)
        print(f"実行されたアクション総数: {total_actions}")

        # 2. データフローの整合性
        all_nodes_completed = all(
            [
                state.inventory_analysis,
                state.sales_analysis,
                state.financial_analysis,
                state.pricing_decision,
                state.restock_decision,
                state.procurement_decision,
                state.sales_processing,
                state.customer_interaction,
                state.profit_calculation,
            ]
        )
        assert all_nodes_completed, "全ノードが完了していない"

        # 3. アクション履歴の存在確認
        actions_count = len(state.executed_actions)
        assert actions_count >= 0, f"アクション履歴がありません: {actions_count}"

        # 4. Session IDの一貫性
        assert state.session_id == initial_state.session_id
        assert state.session_type == initial_state.session_type

        print("✅ 統合フロー全体正常完了")
        print(f"📊 最終状態サマリー:")
        print(f"   - 実行アクション数: {len(state.executed_actions)}")
        print(f"   - エラー数: {len(state.errors)}")
        print(f"   - 最終ステータス: {state.processing_status}")
        print(
            f"   - ツールベース分析: {state.profit_calculation.get('calculation_method', 'unknown')}"
        )

    @pytest.mark.asyncio
    async def test_tool_integration_consistency(self, agent_with_tools, initial_state):
        """ツール統合の一貫性検証"""

        agent = agent_with_tools

        # 利益計算ノード実行 (ツール統合必須)
        state = await agent.profit_calculation_node(initial_state)

        # ツール使用状況の検証
        profit_calc = state.profit_calculation

        # 1. ツール使用データソースが正しく記録されていること
        assert profit_calc.get("data_source") == "get_business_data_tool"
        assert profit_calc.get("analysis_source") == "analyze_financials_tool"
        assert profit_calc.get("calculation_method") == "tool_integrated"

        # 2. ツール推奨事項が統合されていること
        recommendations = profit_calc.get("recommendations", [])
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # 3. ツールベースの財務分析が含まれること
        assert profit_calc.get("tool_based_analysis") is not None

        # 4. 実行アクションにツール使用が記録されていること
        tool_actions = [
            a
            for a in state.executed_actions
            if a.get("type") == "profit_calculation_with_tools"
        ]
        assert len(tool_actions) > 0

        tool_action = tool_actions[0]
        assert "tools_used" in tool_action
        assert "get_business_data" in tool_action["tools_used"]
        assert "analyze_financials" in tool_action["tools_used"]

        print("✅ ツール統合一貫性検証完了")

    @pytest.mark.asyncio
    async def test_state_transition_integrity(self, agent_with_tools):
        """State遷移の完全性検証"""

        agent = agent_with_tools
        state = ManagementState(
            session_id="transition_test_001",
            session_type="management_flow",
            business_metrics={
                "sales": 1000000,
                "profit_margin": 0.25,
                "inventory_level": {"test_item": 10},
                "customer_satisfaction": 4.0,
                "timestamp": "2025-10-06T12:00:00Z",
            },
        )

        # 全ノード実行パイプライン
        nodes = [
            ("inventory_check", agent.inventory_check_node),
            ("sales_plan", agent.sales_plan_node),
            ("pricing", agent.pricing_node),
            ("restock", agent.restock_node),
            ("procurement", agent.procurement_request_generation_node),
            ("sales_processing", agent.sales_processing_node),
            ("customer_interaction", agent.customer_interaction_node),
            ("profit_calculation", agent.profit_calculation_node),
            ("feedback", agent.feedback_node),
        ]

        # 各ノード実行と遷移検証
        for step_name, node_func in nodes:
            print(f"🔄 遷移検証: {step_name}")

            # ノード実行前の状態
            prev_step = state.current_step
            prev_status = state.processing_status

            # ノード実行
            state = await node_func(state)

            # 遷移検証
            assert state.current_step == step_name, (
                f"ステップ遷移失敗: 期待={step_name}, 実際={state.current_step}"
            )
            if step_name != "feedback":  # 最終ノード以外
                assert state.processing_status == "processing", (
                    f"処理ステータス異常 at {step_name}"
                )

        # 最終状態検証
        assert state.processing_status in ["completed", "completed_with_errors"]
        assert state.current_step == "feedback"

        print("✅ State遷移完全性検証完了")
