"""
Search Products Tool - 既存のsearch_agent.pyをShared Toolに変換
"""

import logging
from typing import Any, Dict, List, Optional

from src.agents.search_agent import search_agent
from src.config.settings import settings
from src.tools.shared_tools import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SearchProductsTool(BaseTool):
    """商品検索Shared Tool"""

    def __init__(self):
        super().__init__(
            tool_id="search_products",
            category="market_tools",
            agent_access=[
                "management",
                "analytics",
            ],  # managementとanalyticsのみアクセス可能
        )

    def validate_input(self, **kwargs) -> bool:
        """入力パラメータ検証"""
        if "query" not in kwargs:
            logger.error("Missing required parameter: query")
            return False

        query = kwargs.get("query", "")
        if not isinstance(query, str) or len(query.strip()) == 0:
            logger.error("Query must be non-empty string")
            return False

        if len(query.strip()) > 200:  # 200文字制限
            logger.error("Query too long (max 200 characters)")
            return False

        max_results = kwargs.get("max_results", 5)
        if not isinstance(max_results, int) or not (1 <= max_results <= 20):
            logger.error("max_results must be integer between 1 and 20")
            return False

        return True

    async def execute(self, **kwargs) -> ToolResult:
        """商品検索実行"""
        try:
            query = kwargs.get("query", "").strip()
            max_results = kwargs.get("max_results", 5)

            logger.info(
                f"Executing search_products tool: query='{query}', max_results={max_results}"
            )

            # search_agentの機能を使用
            search_results = await search_agent.search_products(query, max_results)

            if search_results.get("error"):
                return ToolResult(
                    success=False,
                    error_message=f"Search failed: {search_results['error']}",
                    data={},
                    execution_time=0,
                    timestamp=None,
                )

            # 成功時の結果を整形
            results_data = {
                "query": search_results.get("query", query),
                "total_found": search_results.get("total_found", 0),
                "results": search_results.get("results", []),
                "search_type": "market_analysis"
                if "analytics" in self.agent_access
                else "management_decision",
            }

            return ToolResult(
                success=True,
                data=results_data,
                execution_time=0,  # execution_timeはToolResultのコンストラクタで設定
                timestamp=None,
            )

        except Exception as e:
            logger.error(f"SearchProductsTool execution failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                data={},
                execution_time=0,
                timestamp=None,
            )


class SupplierResearchTool(BaseTool):
    """仕入れ先調査Shared Tool"""

    def __init__(self):
        super().__init__(
            tool_id="supplier_research",
            category="market_tools",
            agent_access=[
                "management",
                "analytics",
            ],  # managementとanalyticsのみアクセス可能
        )

    def validate_input(self, **kwargs) -> bool:
        """入力パラメータ検証"""
        if "product_name" not in kwargs:
            logger.error("Missing required parameter: product_name")
            return False

        product_name = kwargs.get("product_name", "")
        if not isinstance(product_name, str) or len(product_name.strip()) == 0:
            logger.error("product_name must be non-empty string")
            return False

        if len(product_name.strip()) > 100:  # 100文字制限
            logger.error("product_name too long (max 100 characters)")
            return False

        max_results = kwargs.get("max_results", 10)
        if not isinstance(max_results, int) or not (1 <= max_results <= 50):
            logger.error("max_results must be integer between 1 and 50")
            return False

        return True

    async def execute(self, **kwargs) -> ToolResult:
        """仕入れ先調査実行"""
        try:
            product_name = kwargs.get("product_name", "").strip()
            max_results = kwargs.get("max_results", 10)

            logger.info(
                f"Executing supplier_research tool: product='{product_name}', max_results={max_results}"
            )

            # search_agentの機能を使用
            supplier_results = await search_agent.find_suppliers(
                product_name, max_results
            )

            # 結果を整形
            results_data = {
                "product_name": product_name,
                "suppliers_found": len(supplier_results),
                "suppliers": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "price": r.price,
                        "availability": r.availability,
                        "source": r.source,
                        "relevance_score": r.relevance_score,
                    }
                    for r in supplier_results
                ],
            }

            return ToolResult(
                success=True, data=results_data, execution_time=0, timestamp=None
            )

        except Exception as e:
            logger.error(f"SupplierResearchTool execution failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                data={},
                execution_time=0,
                timestamp=None,
            )


class MarketAnalysisTool(BaseTool):
    """市場分析Shared Tool"""

    def __init__(self):
        super().__init__(
            tool_id="market_analysis",
            category="market_tools",
            agent_access=[
                "management",
                "analytics",
            ],  # managementとanalyticsのみアクセス可能
        )

    def validate_input(self, **kwargs) -> bool:
        """入力パラメータ検証"""
        if "analysis_type" not in kwargs:
            logger.error("Missing required parameter: analysis_type")
            return False

        analysis_type = kwargs.get("analysis_type", "")
        valid_types = ["supply_demand", "price_trends", "competition", "opportunities"]
        if analysis_type not in valid_types:
            logger.error(f"analysis_type must be one of: {valid_types}")
            return False

        if "product_name" not in kwargs:
            logger.error("Missing required parameter: product_name")
            return False

        product_name = kwargs.get("product_name", "")
        if not isinstance(product_name, str) or len(product_name.strip()) == 0:
            logger.error("product_name must be non-empty string")
            return False

        return True

    async def execute(self, **kwargs) -> ToolResult:
        """市場分析実行"""
        try:
            analysis_type = kwargs.get("analysis_type", "")
            product_name = kwargs.get("product_name", "").strip()
            time_range = kwargs.get("time_range", "1month")

            logger.info(
                f"Executing market_analysis tool: type={analysis_type}, product='{product_name}'"
            )

            # 分析タイプに応じた処理
            if analysis_type == "supply_demand":
                analysis_result = await self._analyze_supply_demand(
                    product_name, time_range
                )
            elif analysis_type == "price_trends":
                analysis_result = await self._analyze_price_trends(
                    product_name, time_range
                )
            elif analysis_type == "competition":
                analysis_result = await self._analyze_competition(product_name)
            elif analysis_type == "opportunities":
                analysis_result = await self._analyze_opportunities(product_name)
            else:
                analysis_result = {"analysis": "Not implemented", "recommendations": []}

            results_data = {
                "analysis_type": analysis_type,
                "product_name": product_name,
                "time_range": time_range,
                "analysis_result": analysis_result,
            }

            return ToolResult(
                success=True, data=results_data, execution_time=0, timestamp=None
            )

        except Exception as e:
            logger.error(f"MarketAnalysisTool execution failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                data={},
                execution_time=0,
                timestamp=None,
            )

    async def _analyze_supply_demand(
        self, product_name: str, time_range: str
    ) -> Dict[str, Any]:
        """需給分析"""
        try:
            # search_agentを使用して市場データを収集
            market_data = await search_agent.search_products(
                f"{product_name} 供給 需給", 10
            )

            analysis = {
                "supply_status": "未知",
                "demand_trends": "安定",
                "recommendations": ["定期的な市場監視をおすすめします"],
            }

            if market_data and market_data.get("results"):
                results = market_data["results"]
                # 簡易分析: 価格安定性から需給を推定
                prices = [r.get("price") for r in results if r.get("price")]
                if len(prices) > 1:
                    price_variation = (
                        (max(prices) - min(prices)) / sum(prices) * len(prices)
                    )
                    if price_variation < 0.1:
                        analysis["supply_status"] = "安定供給"
                        analysis["demand_trends"] = "安定した需要"
                    elif price_variation > 0.3:
                        analysis["supply_status"] = "供給不安定"
                        analysis["demand_trends"] = "変動的な需要"

            return analysis
        except Exception as e:
            logger.error(f"Supply-demand analysis failed: {e}")
            return {"analysis": "分析失敗", "recommendations": []}

    async def _analyze_price_trends(
        self, product_name: str, time_range: str
    ) -> Dict[str, Any]:
        """価格トレンド分析"""
        try:
            # 価格比較を使用してトレンド分析
            price_comparison = await search_agent.compare_prices(product_name)

            if price_comparison.best_price:
                analysis = {
                    "current_best_price": price_comparison.best_price,
                    "trend": "安定的"
                    if price_comparison.price_range[1] - price_comparison.price_range[0]
                    < price_comparison.best_price * 0.2
                    else "変動的",
                    "recommendations": [price_comparison.recommendation],
                }
            else:
                analysis = {
                    "current_best_price": None,
                    "trend": "データ不足",
                    "recommendations": ["追加調査が必要"],
                }

            return analysis
        except Exception as e:
            logger.error(f"Price trends analysis failed: {e}")
            return {"analysis": "分析失敗", "recommendations": []}

    async def _analyze_competition(self, product_name: str) -> Dict[str, Any]:
        """競合分析"""
        try:
            search_results = await search_agent.search_products(
                f"{product_name} 競合", 15
            )

            competitors = []
            if search_results and search_results.get("results"):
                competitors = [
                    {
                        "name": r.get("title", "")[:50],
                        "price": r.get("price"),
                        "url": r.get("url"),
                    }
                    for r in search_results["results"][:5]  # 上位5件
                ]

            return {
                "competitors_found": len(competitors),
                "main_competitors": competitors,
                "recommendations": ["価格競争力の強化を検討"],
            }
        except Exception as e:
            logger.error(f"Competition analysis failed: {e}")
            return {"analysis": "分析失敗", "recommendations": []}

    async def _analyze_opportunities(self, product_name: str) -> Dict[str, Any]:
        """機会分析"""
        try:
            # 新製品検索
            opportunities = []
            search_results = await search_agent.search_products(
                f"{product_name} 新製品 イノベーション", 10
            )

            if search_results and search_results.get("results"):
                opportunities = [
                    {
                        "product_idea": r.get("title", "")[:100],
                        "description": r.get("snippet", "")[:200],
                        "source": r.get("url"),
                    }
                    for r in search_results["results"][:3]  # 上位3件
                ]

            market_opportunities = {
                "new_product_opportunities": opportunities,
                "market_gaps": ["注力商品の補完分野", "季節需要への対応"],
                "recommendations": ["新製品投入の検討", "市場ギャップの活用"],
            }

            return market_opportunities
        except Exception as e:
            logger.error(f"Opportunities analysis failed: {e}")
            return {"analysis": "分析失敗", "recommendations": []}
