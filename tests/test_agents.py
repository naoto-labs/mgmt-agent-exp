import asyncio
import random
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.customer_agent import CustomerAgent
from src.agents.search_agent import SearchAgent, SearchResult, WebSearchService
from src.ai.model_manager import AIMessage, AIModelType, AIResponse
from src.services.conversation_service import ConversationService
from src.services.inventory_service import InventoryService


class TestSearchAgent:
    """検索エージェントのテスト"""

    @pytest.fixture
    def search_agent(self):
        """検索エージェントのフィクスチャ"""
        return SearchAgent()

    @pytest.fixture
    def sample_search_results(self):
        """サンプル検索結果のフィクスチャ"""
        return [
            SearchResult(
                query="コカ・コーラ 価格",
                title="コカ・コーラ 販売サイト",
                url="https://example.com/coke",
                snippet="高品質なコカ・コーラをお探しならこちら",
                price=150.0,
                availability="在庫あり",
                relevance_score=0.9,
            ),
            SearchResult(
                query="コカ・コーラ 価格",
                title="コカ・コーラ 通販",
                url="https://example2.com/coke",
                snippet="激安コカ・コーラ",
                price=140.0,
                availability="在庫あり",
                relevance_score=0.8,
            ),
        ]

    def test_search_agent_initialization(self, search_agent):
        """検索エージェントの初期化テスト"""
        assert search_agent.web_search is not None
        assert search_agent.model_manager is not None
        assert search_agent.search_history == []

    @pytest.mark.asyncio
    async def test_find_suppliers_success(self, search_agent, sample_search_results):
        """仕入れ先検索成功テスト"""
        with patch.object(
            search_agent.web_search, "search_products", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = sample_search_results

            results = await search_agent.find_suppliers("コカ・コーラ", max_results=5)

            assert len(results) == 2
            assert results[0].price == 150.0
            assert results[1].price == 140.0
            mock_search.assert_called_once_with("コカ・コーラ", 5)

    @pytest.mark.asyncio
    async def test_find_suppliers_no_results(self, search_agent):
        """仕入れ先検索結果なしテスト"""
        with patch.object(
            search_agent.web_search, "search_products", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []

            results = await search_agent.find_suppliers("存在しない商品")

            assert results == []

    @pytest.mark.asyncio
    async def test_compare_prices(self, search_agent, sample_search_results):
        """価格比較テスト"""
        with patch.object(
            search_agent, "find_suppliers", new_callable=AsyncMock
        ) as mock_find:
            mock_find.return_value = sample_search_results

            comparison = await search_agent.compare_prices("コカ・コーラ")

            assert comparison.product_name == "コカ・コーラ"
            assert comparison.best_price == 140.0  # 最安価格
            assert comparison.average_price == 145.0  # 平均価格
            assert comparison.price_range == (140.0, 150.0)
            assert len(comparison.search_results) == 2

    def test_search_stats(self, search_agent):
        """検索統計テスト"""
        # 検索履歴を追加
        search_agent.search_history = [
            {"product_name": "商品1", "results_count": 5, "timestamp": datetime.now()},
            {"product_name": "商品2", "results_count": 3, "timestamp": datetime.now()},
        ]

        stats = search_agent.get_search_stats()

        assert stats["total_searches"] == 2
        assert stats["avg_results_per_search"] == 4.0


class TestCustomerAgent:
    """顧客エージェントのテスト"""

    @pytest.fixture
    def customer_agent(self):
        """顧客エージェントのフィクスチャ"""
        return CustomerAgent()

    @pytest.fixture
    def mock_conversation_service(self):
        """モック会話サービスのフィクスチャ"""
        service = MagicMock(spec=ConversationService)
        service.get_conversation_history = AsyncMock(return_value=[])
        service.create_session = AsyncMock(return_value="session_123")
        service.add_message = AsyncMock()
        return service

    @pytest.fixture
    def mock_inventory_service(self):
        """モック在庫サービスのフィクスチャ"""
        service = MagicMock(spec=InventoryService)
        service.get_inventory_summary.return_value = MagicMock(
            total_slots=10, out_of_stock_slots=1
        )
        return service

    @pytest.mark.asyncio
    async def test_engage_customer_success(
        self, customer_agent, mock_conversation_service, mock_inventory_service
    ):
        """顧客エンゲージメント成功テスト"""
        customer_agent.conversation_service = mock_conversation_service
        customer_agent.inventory_service = mock_inventory_service

        # AI応答をモック
        mock_response = AIResponse(
            content='{"content": "こんにちは！", "engagement_type": "greeting", "suggested_products": [], "insights": {}}',
            model_used="mock",
            tokens_used=10,
            response_time=0.5,
            success=True,
        )

        with patch.object(
            customer_agent.model_manager, "generate_response", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = mock_response

            result = await customer_agent.engage_customer("customer_123", "VM001")

            assert result["success"] is True
            assert result["session_id"] == "session_123"
            assert "message" in result
            mock_conversation_service.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_engage_customer_ai_error(
        self, customer_agent, mock_conversation_service, mock_inventory_service
    ):
        """顧客エンゲージメントAIエラーテスト"""
        customer_agent.conversation_service = mock_conversation_service
        customer_agent.inventory_service = mock_inventory_service

        # AIエラーをモック
        mock_response = AIResponse(
            content="",
            model_used="mock",
            tokens_used=0,
            response_time=0.0,
            success=False,
            error_message="AIエラー",
        )

        with patch.object(
            customer_agent.model_manager, "generate_response", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = mock_response

            result = await customer_agent.engage_customer("customer_123", "VM001")

            assert result["success"] is True  # エラー時もデフォルトメッセージで成功扱い
            assert "message" in result

    @pytest.mark.asyncio
    async def test_handle_customer_message(
        self, customer_agent, mock_conversation_service
    ):
        """顧客メッセージ処理テスト"""
        customer_agent.conversation_service = mock_conversation_service

        # 会話データをモック
        conversation_data = {
            "session_id": "session_123",
            "customer_context": {},
            "message_history": [
                {
                    "role": "assistant",
                    "content": "こんにちは",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "previous_insights": {},
        }
        mock_conversation_service.get_conversation_for_ai_agent = AsyncMock(
            return_value=conversation_data
        )

        # AI応答をモック
        mock_response = AIResponse(
            content='{"content": "お役に立てることはありますか？", "engagement_type": "inquiry", "suggested_products": [], "insights": {}}',
            model_used="mock",
            tokens_used=15,
            response_time=0.3,
            success=True,
        )

        with patch.object(
            customer_agent.model_manager, "generate_response", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.return_value = mock_response

            result = await customer_agent.handle_customer_message(
                "session_123", "こんにちは"
            )

            assert result["success"] is True
            assert "お役に立てることはありますか？" in result["message"]
            mock_conversation_service.add_message.assert_called()

    def test_analyze_customer_preferences(self, customer_agent):
        """顧客嗜好分析テスト"""
        history = [
            {
                "summary": "コーヒーについて質問",
                "context": {"product_category": "drink"},
            },
            {
                "summary": "スナックについて会話",
                "context": {"product_category": "snack"},
            },
        ]

        preferences = customer_agent._analyze_customer_preferences(history)

        assert "コーヒー" in preferences
        assert "スナック" in preferences


class TestWebSearchService:
    """Web検索サービスのテスト"""

    @pytest.fixture
    def web_search(self):
        """Web検索サービスのフィクスチャ"""
        return WebSearchService()

    @pytest.mark.asyncio
    async def test_search_products_simulation(self, web_search):
        """商品検索シミュレーションテスト"""
        results = await web_search.search_products("コカ・コーラ", max_results=5)

        assert len(results) > 0
        assert all(result.price is not None for result in results)
        assert all(result.relevance_score > 0 for result in results)

    def test_estimate_base_price(self, web_search):
        """ベース価格推定テスト"""
        price = web_search._estimate_base_price("コカ・コーラ")
        assert price == 150.0

        price = web_search._estimate_base_price("ポテトチップス")
        assert price == 180.0

        price = web_search._estimate_base_price("不明な商品")
        assert price == 200.0  # デフォルト価格

    def test_deduplicate_results(self, web_search):
        """検索結果重複除去テスト"""
        results = [
            SearchResult(
                query="test",
                title="タイトル1",
                url="https://example1.com",
                snippet="説明1",
                relevance_score=0.9,
            ),
            SearchResult(
                query="test",
                title="タイトル2",
                url="https://example2.com",
                snippet="説明2",
                relevance_score=0.8,
            ),
            SearchResult(
                query="test",
                title="タイトル1",
                url="https://example1.com",
                snippet="説明1",
                relevance_score=0.9,
            ),  # 重複
        ]

        unique_results = web_search._deduplicate_results(results)

        assert len(unique_results) == 2  # 重複が除去される


# パフォーマンステスト
@pytest.mark.asyncio
async def test_search_agent_performance():
    """検索エージェントのパフォーマンステスト"""
    agent = SearchAgent()

    start_time = datetime.now()

    # 複数回の検索を実行
    for i in range(3):
        results = await agent.find_suppliers(f"商品{i}", max_results=5)
        assert len(results) >= 0

    elapsed_time = (datetime.now() - start_time).total_seconds()

    # 各検索が1秒以内に完了することを確認
    assert elapsed_time < 3.0


# エラーハンドリングテスト
@pytest.mark.asyncio
async def test_search_agent_error_handling():
    """検索エージェントのエラーハンドリングテスト"""
    agent = SearchAgent()

    # Web検索で例外が発生する場合
    with patch.object(
        agent.web_search, "search_products", new_callable=AsyncMock
    ) as mock_search:
        mock_search.side_effect = Exception("検索エラー")

        results = await agent.find_suppliers("エラー商品")

        assert results == []  # エラー時は空リストを返す


# インテグレーションテスト
@pytest.mark.asyncio
async def test_customer_agent_integration():
    """顧客エージェントの統合テスト"""
    agent = CustomerAgent()

    # エンゲージメントとメッセージ処理の流れをテスト
    engagement = await agent.engage_customer("test_customer", "VM001")
    assert engagement["success"] is True

    if engagement["session_id"]:
        response = await agent.handle_customer_message(
            engagement["session_id"], "テストメッセージ"
        )
        assert "success" in response


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
