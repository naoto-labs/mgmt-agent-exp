import asyncio
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from src.ai.model_manager import AIMessage, AIResponse, ModelManager, model_manager
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """検索結果"""

    query: str
    title: str
    url: str
    snippet: str
    price: Optional[float] = None
    currency: str = "JPY"
    availability: Optional[str] = None
    source: str = "web"
    relevance_score: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PriceComparison:
    """価格比較結果"""

    product_name: str
    search_results: List[SearchResult]
    best_price: Optional[float] = None
    average_price: Optional[float] = None
    price_range: Tuple[Optional[float], Optional[float]] = (None, None)
    recommendation: str = ""
    confidence: float = 0.0


class DuckDuckGoSearchClient:
    """DuckDuckGo検索クライアント（無料・APIキー不要）"""

    def __init__(self):
        self.base_url = "https://duckduckgo.com"
        self.search_url = "https://html.duckduckgo.com/html"
        self.ua = UserAgent()
        self.session_headers = {
            "User-Agent": self.ua.random,
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    async def search(
        self, query: str, max_results: int = 5, site_limit: Optional[str] = None
    ) -> List[SearchResult]:
        """DuckDuckGoで検索を実行"""
        try:
            enhanced_query = query
            if site_limit:
                if ":" in site_limit:  # site:domain.com形式
                    enhanced_query = f"{query} {site_limit}"
                else:  # ドメイン指定のみ
                    enhanced_query = f"{query} site:{site_limit}"

            params = {
                "q": enhanced_query,
                "kl": "jp-jp",  # 日本地域指定
            }

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.search_timeout),
                headers=self.session_headers,
                follow_redirects=True,
            ) as client:
                response = await client.post(self.search_url, data=params)

                if response.status_code not in [200, 202, 302, 303]:
                    logger.warning(f"DuckDuckGo search failed: {response.status_code}")
                    return []

                soup = BeautifulSoup(response.text, "html.parser")

                # デバッグ: HTML構造確認
                print(f"[DEBUG] DuckDuckGo response status: {response.status_code}")
                print(f"[DEBUG] DuckDuckGo response length: {len(response.text)}")
                print(
                    f"[DEBUG] DuckDuckGo HTML title: {soup.title.text if soup.title else 'No title'}"
                )

                # HTML構造の詳細確認
                result_selectors = [".result", ".web-result", ".result__item"]
                all_found = 0
                for selector in result_selectors:
                    count = len(soup.select(selector))
                    print(f"[DEBUG] Selector '{selector}': {count} elements")
                    all_found += count

                print(f"[DEBUG] Total result elements found: {all_found}")

                # HTMLファイルを保存して検査
                with open("duckduckgo_response.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                print("[DEBUG] Saved HTML response to duckduckgo_response.html")

                # 特定のセレクタで検索
                result_items = soup.select(".result,.web-result")
                print(f"[DEBUG] Final selector search found: {len(result_items)} items")

                return await self._parse_search_results(
                    soup, enhanced_query, max_results, site_limit
                )

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _parse_search_results(
        self,
        soup: BeautifulSoup,
        query: str,
        max_results: int,
        site_limit: Optional[str] = None,
    ) -> List[SearchResult]:
        """検索結果をパース"""
        results = []

        # DuckDuckGoの検索結果構造
        result_items = soup.select(".result,.web-result")

        for item in result_items[:max_results]:
            try:
                # タイトルとURLを取得
                title_elem = item.select_one(".result__title,.result__a")
                if not title_elem:
                    continue

                url_elem = item.select_one(".result__url,.result__a,.result__link")
                if not url_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = self._extract_url_from_element(url_elem)

                if not url or not title:
                    continue

                # スニペット取得
                snippet_elem = item.select_one(".result__snippet,.result__content")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                # ソース判定とスコアリング
                source = self._determine_source(url)
                relevance_score = self._calculate_relevance_score(source)

                # 価格抽出（スニペットから）
                price = self._extract_price_from_snippet(snippet)
                availability = self._extract_availability_from_snippet(snippet)

                result = SearchResult(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet[:200] + "..." if len(snippet) > 200 else snippet,
                    price=price,
                    availability=availability,
                    source=source,
                    relevance_score=relevance_score,
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse DuckDuckGo result: {e}")
                continue

        return results

    def _extract_url_from_element(self, elem) -> Optional[str]:
        """要素からURLを抽出"""
        try:
            # href属性がある場合
            if hasattr(elem, "get") and elem.get("href"):
                url = elem["href"]
                if url.startswith("//"):
                    return f"https:{url}"
                elif url.startswith("/"):
                    return f"{self.base_url}{url}"
                else:
                    return url

            # テキストからURLを抽出
            text = elem.get_text(strip=True)
            if text.startswith(("http://", "https://")):
                return text

        except Exception as e:
            logger.debug(f"URL extraction error: {e}")

        return None

    def _determine_source(self, url: str) -> str:
        """URLからソースを判定"""
        try:
            if "amazon.co.jp" in url:
                return "amazon"
            elif "shopping.yahoo.co.jp" in url:
                return "yahoo_shopping"
            elif "rakuten.co.jp" in url:
                return "rakuten"
            elif "aeon" in url:
                return "aeon"
            else:
                return "other"
        except:
            return "unknown"

    def _calculate_relevance_score(self, source: str) -> float:
        """ソースに基づいて関連度スコアを計算"""
        score_map = {
            "amazon": 0.9,
            "yahoo_shopping": 0.8,
            "rakuten": 0.8,
            "aeon": 0.7,
            "other": 0.6,
            "unknown": 0.5,
        }
        return score_map.get(source, 0.5)

    def _extract_price_from_snippet(self, snippet: str) -> Optional[float]:
        """スニペットから価格を抽出"""
        try:
            # 各種価格パターンを検索
            price_patterns = [
                r"¥[\s]*([0-9,]+(?:\.[0-9]{2})?)",
                r"([0-9,]+(?:\.[0-9]{2})?)\s*円",
                r"価格:\s*([0-9,]+(?:\.[0-9]{2})?)",
                r"([0-9,]+(?:\.[0-9]{2})?)\s*yen",
            ]

            for pattern in price_patterns:
                match = re.search(pattern, snippet, re.IGNORECASE)
                if match:
                    price_str = match.group(1).replace(",", "")
                    return float(price_str)

        except Exception as e:
            logger.debug(f"Price extraction from snippet error: {e}")

        return None

    def _extract_availability_from_snippet(self, snippet: str) -> Optional[str]:
        """スニペットから在庫状況を抽出"""
        try:
            snippet_lower = snippet.lower()

            if (
                "在庫切れ" in snippet
                or "out of stock" in snippet
                or "sold out" in snippet
            ):
                return "在庫切れ"
            elif (
                "在庫あり" in snippet or "in stock" in snippet or "available" in snippet
            ):
                return "在庫あり"
            elif "入荷待ち" in snippet or "back order" in snippet:
                return "入荷待ち"
            else:
                return None

        except Exception as e:
            logger.debug(f"Availability extraction error: {e}")
        return None


class TavilySearchClient:
    """Tavily検索クライアント"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.price_patterns = [
            r"¥[\s]*([0-9,]+(?:\.[0-9]{2})?)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*円",
            r"価格:\s*([0-9,]+(?:\.[0-9]{2})?)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*yen",
        ]

        if TavilyClient:
            self.client = TavilyClient(api_key=api_key)
        else:
            logger.warning("Tavily client not available")

    async def search(
        self, query: str, max_results: int = 5, site_limit: Optional[str] = None
    ) -> List[SearchResult]:
        """Tavilyで検索を実行"""
        if not self.client:
            logger.error("Tavily client not initialized")
            return []

        try:
            enhanced_query = query
            if site_limit:
                if ":" in site_limit:  # site:domain.com形式
                    enhanced_query = f"{query} {site_limit}"
                else:  # ドメイン指定のみ
                    enhanced_query = f"{query} site:{site_limit}"

            search_kwargs = {
                "query": enhanced_query,
                "search_depth": "advanced",
                "max_results": max_results,
            }

            # site_limitがある場合はinclude_domainsを使用
            if site_limit:
                # site:domain.comからドメインを抽出
                if site_limit.startswith("site:"):
                    domain = site_limit[5:]
                else:
                    domain = site_limit
                search_kwargs["include_domains"] = [domain]

            # 同期呼び出しなのでasyncio.run_in_executorを使用
            import concurrent.futures

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: self.client.search(**search_kwargs)
            )

            return self._parse_search_results(
                results, enhanced_query, max_results, site_limit
            )

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def _parse_search_results(
        self,
        results: dict,
        query: str,
        max_results: int,
        site_limit: Optional[str] = None,
    ) -> List[SearchResult]:
        """Tavily検索結果をパース"""
        search_results = []

        # Tavilyのレスポンス構造に基づいてパース
        results_list = results.get("results", [])

        for item in results_list[:max_results]:
            try:
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("content", "")

                if not title or not url:
                    continue

                # ソース判定とスコアリング
                source = self._determine_source(url)
                relevance_score = self._calculate_relevance_score(source)

                # 価格抽出
                price = self._extract_price_from_snippet(snippet)
                availability = self._extract_availability_from_snippet(snippet)

                search_result = SearchResult(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet[:200] + "..." if len(snippet) > 200 else snippet,
                    price=price,
                    availability=availability,
                    source=source,
                    relevance_score=relevance_score,
                )
                search_results.append(search_result)

            except Exception as e:
                logger.warning(f"Failed to parse Tavily result: {e}")
                continue

        return search_results

    def _determine_source(self, url: str) -> str:
        """URLからソースを判定"""
        try:
            if "amazon.co.jp" in url:
                return "amazon"
            elif "shopping.yahoo.co.jp" in url:
                return "yahoo_shopping"
            elif "rakuten.co.jp" in url:
                return "rakuten"
            elif "aeon" in url:
                return "aeon"
            else:
                return "other"
        except:
            return "unknown"

    def _calculate_relevance_score(self, source: str) -> float:
        """ソースに基づいて関連度スコアを計算"""
        score_map = {
            "amazon": 0.9,
            "yahoo_shopping": 0.8,
            "rakuten": 0.8,
            "aeon": 0.7,
            "other": 0.6,
            "unknown": 0.5,
        }
        return score_map.get(source, 0.5)

    def _extract_price_from_snippet(self, snippet: str) -> Optional[float]:
        """スニペットから価格を抽出"""
        try:
            for pattern in self.price_patterns:
                match = re.search(pattern, snippet, re.IGNORECASE)
                if match:
                    price_str = match.group(1).replace(",", "")
                    return float(price_str)
        except Exception as e:
            logger.debug(f"Price extraction from snippet error: {e}")
        return None

    def _extract_availability_from_snippet(self, snippet: str) -> Optional[str]:
        """スニペットから在庫状況を抽出"""
        try:
            snippet_lower = snippet.lower()

            if (
                "在庫切れ" in snippet
                or "out of stock" in snippet
                or "sold out" in snippet
            ):
                return "在庫切れ"
            elif (
                "在庫あり" in snippet or "in stock" in snippet or "available" in snippet
            ):
                return "在庫あり"
            elif "入荷待ち" in snippet or "back order" in snippet:
                return "入荷待ち"
            else:
                return None

        except Exception as e:
            logger.debug(f"Availability extraction error: {e}")
            return None


class WebSearchService:
    """Web検索サービス（実検索 + フォールバック）"""

    def __init__(self):
        self.search_engines = ["google", "yahoo", "bing"]
        self.price_patterns = [
            r"¥\s*([0-9,]+)",
            r"([0-9,]+)\s*円",
            r"価格:\s*([0-9,]+)",
            r"([0-9,]+)\s*yen",
            r"([0-9.]+)\s*USD",
        ]
        # 検索クライアントの初期化：Tavily優先、ない場合はDuckDuckGo
        self.search_client = self._initialize_search_client()
        self.target_sites = ["amazon.co.jp", "shopping.yahoo.co.jp"]

    def _initialize_search_client(self):
        """検索クライアントを初期化"""
        tavily_api_key = settings.tavily_api_key
        if tavily_api_key and TavilyClient:
            logger.info("Tavily検索クライアントを使用")
            return TavilySearchClient(tavily_api_key)
        else:
            logger.info("DuckDuckGo検索クライアントを使用")
            return DuckDuckGoSearchClient()

    async def search_products(
        self, product_name: str, max_results: int = 10
    ) -> List[SearchResult]:
        """商品を検索"""
        logger.info(f"商品検索開始: {product_name}")

        if settings.real_web_search:
            # 実検索を使用
            return await self._real_search_products(product_name, max_results)
        else:
            # シミュレーション検索を使用
            return await self._simulate_search_products(product_name, max_results)

    async def _real_search_products(
        self, product_name: str, max_results: int = 10
    ) -> List[SearchResult]:
        """DuckDuckGoを使った商品検索"""
        logger.info(f"DuckDuckGo実検索開始: {product_name}")
        all_results = []

        try:
            # 各ターゲットサイトに対して検索を実行
            search_tasks = []
            for site in self.target_sites:
                # サイトごとに結果数を割り当て
                site_results = max_results // len(self.target_sites) + (
                    max_results % len(self.target_sites) > len(search_tasks)
                )
                search_tasks.append(
                    self.search_client.search(product_name, site_results, site)
                )

            # 結果を収集（異常終了時は例外を返却）
            search_results_list = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            # 例外を処理して正常な結果のみを集める
            for results in search_results_list:
                if isinstance(results, Exception):
                    logger.error(f"DuckDuckGo search failed: {results}")
                    continue
                all_results.extend(results)

            # 重複を除去して関連度順にソート
            unique_results = self._deduplicate_results(all_results)
            sorted_results = sorted(
                unique_results, key=lambda x: x.relevance_score, reverse=True
            )

            logger.info(f"DuckDuckGo実検索完了: {len(sorted_results)}件の結果")
            return sorted_results[:max_results]

        except Exception as e:
            logger.error(f"実検索失敗、シミュレーションにフォールバック: {e}")
            # フォールバック
            return await self._simulate_search_products(product_name, max_results)

    async def _simulate_search_products(
        self, product_name: str, max_results: int = 10
    ) -> List[SearchResult]:
        """シミュレーション検索"""
        logger.info(f"シミュレーション検索開始: {product_name}")

        # 検索クエリの構築
        queries = [
            f"{product_name} 価格",
            f"{product_name} 販売",
            f"{product_name} 通販",
            f"{product_name} 購入",
        ]

        all_results = []

        for query in queries:
            # 検索結果をシミュレート
            results = await self._simulate_search(query, max_results // len(queries))
            all_results.extend(results)

        # 重複を除去して関連度順にソート
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(
            unique_results, key=lambda x: x.relevance_score, reverse=True
        )

        logger.info(f"シミュレーション検索完了: {len(sorted_results)}件の結果")
        return sorted_results[:max_results]

    async def _simulate_search(
        self, query: str, max_results: int
    ) -> List[SearchResult]:
        """検索をシミュレート"""
        # 実際の実装では、httpxやrequestsを使ってWeb検索APIを呼び出す
        # ここではシミュレーションとしてダミーデータを生成

        await asyncio.sleep(0.5)  # 検索時間をシミュレート

        results = []
        base_price = self._estimate_base_price(query)

        for i in range(min(max_results, 5)):
            price = base_price * (0.8 + random.random() * 0.4)  # ±20%の価格変動

            result = SearchResult(
                query=query,
                title=f"{query} - 販売サイト {i + 1}",
                url=f"https://example-shop{i + 1}.com/products/{random.randint(1000, 9999)}",
                snippet=f"高品質な{query}をお探しならこちら。価格: ¥{price:,.0f} 在庫あり。",
                price=price,
                availability="在庫あり" if random.random() > 0.1 else "在庫切れ",
                relevance_score=random.uniform(0.7, 0.95),
            )
            results.append(result)

        return results

    def _estimate_base_price(self, product_name: str) -> float:
        """商品のベース価格を推定"""
        # 簡易的な価格推定ロジック
        price_keywords = {
            "コカ・コーラ": 150,
            "ポテトチップス": 180,
            "カップヌードル": 200,
            "お茶": 120,
            "コーヒー": 300,
            "ジュース": 140,
            "チョコレート": 250,
            "キャンディー": 100,
        }

        for keyword, price in price_keywords.items():
            if keyword in product_name:
                return price

        # デフォルト価格
        return 200.0

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """検索結果の重複を除去"""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results


class SearchAgent:
    """検索エージェント"""

    def __init__(self):
        self.web_search = WebSearchService()
        self.model_manager = model_manager
        self.search_history: List[Dict[str, Any]] = []

    async def find_suppliers(
        self, product_name: str, max_results: int = 10
    ) -> List[SearchResult]:
        """仕入れ先を検索"""
        logger.info(f"仕入れ先検索開始: {product_name}")

        try:
            # Web検索を実行
            search_results = await self.web_search.search_products(
                product_name, max_results
            )

            # AIで検索結果を分析・構造化
            if search_results:
                analyzed_results = await self._analyze_search_results(
                    product_name, search_results
                )
            else:
                analyzed_results = search_results

            # 検索履歴を記録
            self._record_search(
                {
                    "product_name": product_name,
                    "results_count": len(analyzed_results),
                    "timestamp": datetime.now(),
                }
            )

            return analyzed_results

        except Exception as e:
            logger.error(f"仕入れ先検索エラー: {e}")
            return []

    async def search_products(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """商品を検索（辞書フォーマット対応）"""
        logger.info(f"商品検索: {query}")

        try:
            # search_productsを呼び出し
            web_results = await self.web_search.search_products(query, max_results)

            # 結果を整形
            results = []
            for result in web_results:
                results.append(
                    {
                        "title": result.title,
                        "price": result.price,
                        "availability": result.availability,
                        "url": result.url,
                        "relevance_score": result.relevance_score,
                    }
                )

            return {"results": results, "query": query, "total_found": len(results)}

        except Exception as e:
            logger.error(f"商品検索エラー: {e}")
            return {"results": [], "query": query, "total_found": 0, "error": str(e)}

    async def _analyze_search_results(
        self, product_name: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """AIで検索結果を分析"""
        try:
            # AIプロンプトの構築
            prompt = self._build_analysis_prompt(product_name, results)

            # AIメッセージの作成
            messages = [
                AIMessage(
                    role="system",
                    content="あなたは商品検索結果の分析エキスパートです。価格比較と仕入れ先評価を正確に行ってください。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            # AIで分析を実行
            response = await self.model_manager.generate_response(
                messages, max_tokens=500
            )

            if response.success:
                # AIの分析結果をパースして検索結果に適用
                return self._apply_ai_analysis(results, response.content)
            else:
                logger.warning(f"AI分析失敗、元の結果を返却: {response.error_message}")
                return results

        except Exception as e:
            logger.error(f"検索結果分析エラー: {e}")
            return results

    def _build_analysis_prompt(
        self, product_name: str, results: List[SearchResult]
    ) -> str:
        """分析プロンプトを構築"""
        results_text = "\n".join(
            [
                f"タイトル: {r.title}\n価格: {r.price}円\n説明: {r.snippet}\n在庫: {r.availability}\n"
                for r in results[:5]  # 上位5件のみ分析
            ]
        )

        return f"""
        商品名: {product_name}

        検索結果:
        {results_text}

        上記の検索結果を分析し、以下のJSON形式で応答してください：
        {{
            "price_analysis": {{
                "best_price": 最低価格,
                "average_price": 平均価格,
                "price_range": [最低価格, 最高価格],
                "recommendation": "価格比較に基づく推奨文"
            }},
            "supplier_evaluation": [
                {{
                    "index": 結果のインデックス,
                    "reliability_score": 信頼性スコア（0-1）,
                    "price_competitiveness": 価格競争力（0-1）,
                    "notes": "評価コメント"
                }}
            ]
        }}

        分析のポイント：
        1. 価格の妥当性を評価
        2. 仕入れ先の信頼性を評価
        3. 在庫状況を考慮
        4. 配送条件や最小注文量を推測
        """

    def _apply_ai_analysis(
        self, results: List[SearchResult], ai_response: str
    ) -> List[SearchResult]:
        """AI分析結果を検索結果に適用"""
        try:
            # JSONパース（簡易版）
            import json

            analysis = json.loads(ai_response)

            # 価格分析の適用
            if "price_analysis" in analysis:
                price_analysis = analysis["price_analysis"]

                # 各結果の関連度スコアを調整
                for i, result in enumerate(results):
                    if result.price:
                        # 価格に基づくスコア調整
                        if result.price <= price_analysis.get(
                            "best_price", float("inf")
                        ):
                            result.relevance_score = min(
                                1.0, result.relevance_score + 0.1
                            )

            # 仕入れ先評価の適用
            if "supplier_evaluation" in analysis:
                for evaluation in analysis["supplier_evaluation"]:
                    index = evaluation.get("index", 0)
                    if 0 <= index < len(results):
                        # 信頼性スコアを関連度に反映
                        reliability = evaluation.get("reliability_score", 0.5)
                        results[index].relevance_score = (
                            results[index].relevance_score + reliability
                        ) / 2

            return results

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"AI分析結果のパースエラー: {e}")
            return results

    async def compare_prices(self, product_name: str) -> PriceComparison:
        """価格比較を実行"""
        logger.info(f"価格比較開始: {product_name}")

        try:
            # 検索結果を取得
            search_results = await self.find_suppliers(product_name, max_results=15)

            if not search_results:
                return PriceComparison(
                    product_name=product_name,
                    search_results=[],
                    recommendation="検索結果が見つかりませんでした",
                )

            # 価格情報を抽出
            prices = [r.price for r in search_results if r.price is not None]

            if not prices:
                return PriceComparison(
                    product_name=product_name,
                    search_results=search_results,
                    recommendation="価格情報が見つかりませんでした",
                )

            # 統計計算
            best_price = min(prices)
            average_price = sum(prices) / len(prices)
            price_range = (min(prices), max(prices))

            # AIで推奨コメントを生成
            recommendation = await self._generate_price_recommendation(
                product_name, best_price, average_price, search_results
            )

            return PriceComparison(
                product_name=product_name,
                search_results=search_results,
                best_price=best_price,
                average_price=average_price,
                price_range=price_range,
                recommendation=recommendation,
                confidence=0.8,  # 簡易的な信頼度
            )

        except Exception as e:
            logger.error(f"価格比較エラー: {e}")
            return PriceComparison(
                product_name=product_name,
                search_results=[],
                recommendation=f"価格比較に失敗しました: {str(e)}",
            )

    async def _generate_price_recommendation(
        self,
        product_name: str,
        best_price: float,
        average_price: float,
        results: List[SearchResult],
    ) -> str:
        """価格推奨コメントを生成"""
        try:
            # AIプロンプトの構築
            prompt = f"""
            商品名: {product_name}
            最安価格: ¥{best_price:,.0f}
            平均価格: ¥{average_price:,.0f}
            検索結果数: {len(results)}

            上記の情報に基づいて、仕入れ担当者への推奨コメントを作成してください。
            以下のポイントを考慮：
            1. 価格の市場相場に対する評価
            2. 購入タイミングの提案
            3. 注意点やリスクの指摘
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは経験豊富な調達担当者です。価格分析に基づいて実践的なアドバイスを提供してください。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=200
            )

            if response.success:
                return response.content.strip()
            else:
                return f"最安価格 ¥{best_price:,.0f} で購入を検討してください。"

        except Exception as e:
            logger.error(f"価格推奨生成エラー: {e}")
            return f"最安価格 ¥{best_price:,.0f} で購入を検討してください。"

    def _record_search(self, search_data: Dict[str, Any]):
        """検索を記録"""
        self.search_history.append(search_data)

        # 履歴数の制限（最新50件）
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]

    def get_search_stats(self) -> Dict[str, Any]:
        """検索統計を取得"""
        if not self.search_history:
            return {"total_searches": 0, "avg_results": 0}

        total_searches = len(self.search_history)
        total_results = sum(search["results_count"] for search in self.search_history)

        return {
            "total_searches": total_searches,
            "avg_results_per_search": total_results / total_searches,
            "recent_searches": self.search_history[-5:] if self.search_history else [],
        }


# グローバルインスタンス
search_agent = SearchAgent()
