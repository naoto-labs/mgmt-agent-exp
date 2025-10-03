"""
セッション型経営管理Agent

LangChainで実装した統合経営管理システム
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

# 低レベルHTTPログを抑制（重要なManagement Agentログのみ表示）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.domain.models.product import SAMPLE_PRODUCTS
from src.infrastructure import model_manager
from src.shared import secure_config, settings

logger = logging.getLogger(__name__)


class SessionInfo(BaseModel):
    """セッション情報"""

    session_id: str
    session_type: str  # "morning_routine", "midday_check", "evening_summary"
    start_time: datetime
    end_time: Optional[datetime] = None
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    actions_executed: List[Dict[str, Any]] = Field(default_factory=list)


class BusinessMetrics(BaseModel):
    """事業メトリクス"""

    sales: float
    profit_margin: float
    inventory_level: Dict[str, int]
    customer_satisfaction: float
    timestamp: datetime


class SessionBasedManagementAgent:
    """セッション型経営管理Agent"""

    def __init__(self, provider: str = "openai"):
        """
        Args:
            provider: LLMプロバイダー ("anthropic" or "openai" - for model_manager selection)
        """
        self.provider = provider
        self.current_session: Optional[SessionInfo] = None
        self._system_prompt_logged = False  # システムプロンプトログ出力フラグ

        # model_managerを使用するため、プロバイダー指定は情報用途のみ
        logger.info(
            f"SessionBasedManagementAgent initialized with provider: {provider}"
        )

        # LLM接続確認
        self._verify_llm_connection()

        # 設定からAgent目的を取得してシステムプロンプト生成
        self.agent_objectives = settings.agent_objectives
        self.system_prompt = self._generate_system_prompt()

        # ツールの初期化
        self.tools = self._create_tools()

    async def _verify_llm_connection_async(self):
        """SessionBasedManagementAgent初期化時LLM接続確認（非同期版）"""
        logger.info("SessionBasedManagementAgentのLLM接続を確認しています...")

        try:
            # ヘルスチェックを実行
            health_results = await model_manager.check_all_models_health()

            # 結果のログ出力
            for model_name, is_healthy in health_results.items():
                if is_healthy:
                    logger.info(f"✅ AIモデル {model_name}: 接続確認成功")
                else:
                    logger.warning(f"❌ AIモデル {model_name}: 接続失敗")

            # 少なくとも1つのモデルが利用可能か確認
            available_models = [
                name for name, healthy in health_results.items() if healthy
            ]
            if not available_models:
                logger.warning(
                    "⚠️ 利用可能なAIモデルがありません。システムは制限モードで動作します。"
                )
            else:
                logger.info(
                    f"🚀 AI処理準備完了（利用可能モデル: {', '.join(available_models)}）"
                )

        except Exception as e:
            logger.error(f"LLM接続確認中にエラーが発生しました: {e}", exc_info=True)
            logger.warning("⚠️ AIモデル接続確認をスキップします。")

    def _verify_llm_connection(self):
        """SessionBasedManagementAgent初期化時LLM接続確認"""
        import asyncio

        try:
            # 新しいイベントループを作成して非同期関数を実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(self._verify_llm_connection_async())
            loop.run_until_complete(task)
            loop.close()
        except Exception as e:
            logger.error(f"LLM接続確認実行エラー: {e}")
            logger.warning("⚠️ AIモデル接続確認をスキップします。")

    def _generate_system_prompt(self) -> str:
        """Agent目的設定に基づいてシステムプロンプトを生成"""
        objectives = self.agent_objectives

        prompt = f"""
あなたは自動販売機事業の経営者です。以下の設定に基づいて意思決定を行ってください。

【主要目的】
{chr(10).join(f"- {obj}" for obj in objectives["primary"])}

【最適化期間枠設定】(戦略的優先度: {objectives["priority_weight"]})
"""

        for period_key, descriptions in objectives["optimization_period"].items():
            weight = objectives["priority_weight"].get(period_key, 0.0)
            prompt += f"- {period_key}: {descriptions} (重み: {weight})\n"

        prompt += f"""
【制約条件】
{chr(10).join(f"- {constraint}" for constraint in objectives["constraints"])}

【業務統括】
- 売上・財務データの分析と戦略立案
- 在庫状況の監視と補充計画
- 価格戦略の決定と実行指示
- 従業員への作業指示（補充、調達、メンテナンス）
- 顧客からの問い合わせ対応と苦情処理

【意思決定原則】
- 短期・中期・長期目標のバランスを考慮して収益性を最優先
- 顧客満足度を維持しつつ長期的な成長を図る
- リスクを適切に管理し、安定的な事業運営を行う
- データに基づいた戦略的判断を行う
"""

        return prompt

    def _test_llm_connection_sync(self):
        """model_manager経由でLLM接続確認（同期版）"""
        try:
            # 通常は非同期処理だが、initでは同期的にヘルスチェックのみを実行
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            health_results = loop.run_until_complete(
                model_manager.check_all_models_health()
            )

            for model_name, is_healthy in health_results.items():
                if is_healthy:
                    logger.info(f"✅ モデル {model_name}: ヘルスチェック成功")
                else:
                    logger.warning(f"❌ モデル {model_name}: ヘルスチェック失敗")

            loop.close()

        except Exception as e:
            logger.error(f"LLM接続テストに失敗: {e}", exc_info=True)

    def _create_tools(self) -> List[StructuredTool]:
        """LangChainツールの作成"""
        tools = []

        # システム連携ツール
        tools.extend(self._create_system_integration_tools())

        # 人間協働ツール
        tools.extend(self._create_human_collaboration_tools())

        # 顧客対応ツール
        tools.extend(self._create_customer_service_tools())

        return tools

    def _create_system_integration_tools(self) -> List[StructuredTool]:
        """システム連携ツール群"""
        return [
            StructuredTool.from_function(
                func=self.get_business_metrics,
                name="get_business_data",
                description="売上、在庫、顧客データをシステムから取得",
            ),
            StructuredTool.from_function(
                func=self.analyze_financial_performance,
                name="analyze_financials",
                description="財務実績を分析し、収益性を評価",
            ),
            StructuredTool.from_function(
                func=self.check_inventory_status,
                name="check_inventory",
                description="在庫レベルと回転率を確認",
            ),
            StructuredTool.from_function(
                func=self.update_pricing,
                name="update_pricing",
                description="価格戦略を決定し、システムに反映",
            ),
        ]

    def _create_human_collaboration_tools(self) -> List[StructuredTool]:
        """人間協働ツール群"""
        return [
            StructuredTool.from_function(
                func=self.assign_restocking_task,
                name="assign_restocking",
                description="従業員に商品補充作業を指示",
            ),
            StructuredTool.from_function(
                func=self.request_procurement,
                name="request_procurement",
                description="担当者に商品調達を依頼",
            ),
            StructuredTool.from_function(
                func=self.schedule_maintenance,
                name="schedule_maintenance",
                description="メンテナンス作業のスケジュール調整",
            ),
            StructuredTool.from_function(
                func=self.coordinate_employee_tasks,
                name="coordinate_tasks",
                description="従業員の業務配分と進捗管理",
            ),
        ]

    def _create_customer_service_tools(self) -> List[StructuredTool]:
        """顧客対応ツール群"""
        return [
            StructuredTool.from_function(
                func=self.respond_to_customer_inquiry,
                name="customer_response",
                description="顧客からの問い合わせに回答",
            ),
            StructuredTool.from_function(
                func=self.handle_customer_complaint,
                name="handle_complaint",
                description="顧客苦情の処理と解決策提案",
            ),
            StructuredTool.from_function(
                func=self.collect_customer_feedback,
                name="collect_feedback",
                description="顧客要望の収集と新商品検討",
            ),
            StructuredTool.from_function(
                func=self.create_customer_engagement_campaign,
                name="create_campaign",
                description="顧客エンゲージメント施策の企画",
            ),
        ]

    # ツール実装メソッド

    def get_business_metrics(self) -> Dict[str, Any]:
        """ビジネスメトリクスを取得（実際のシステムと連携）"""
        logger.info("Getting business metrics from actual systems")

        try:
            # 各種サービスをインポート
            from datetime import date, timedelta

            from src.application.services.inventory_service import inventory_service
            from src.domain.accounting.management_accounting import management_analyzer

            # 在庫情報を取得
            inventory_summary = inventory_service.get_inventory_summary()
            inventory_level = {}

            # 商品別在庫を集計
            for slot in inventory_service.vending_machine_slots.values():
                product_name = slot.product_name.lower()
                if product_name not in inventory_level:
                    inventory_level[product_name] = 0
                inventory_level[product_name] += slot.current_quantity

            # 財務情報を取得（管理会計から）
            end_date = date.today()
            start_date = end_date - timedelta(days=30)

            # 売上情報を取得（会計システムから）
            sales = abs(
                management_analyzer.journal_processor.get_account_balance(
                    "4001", start_date, end_date
                )
            )

            period_profitability = management_analyzer.analyze_period_profitability(
                start_date, end_date
            )
            profit_margin = period_profitability.get("gross_margin", 0.35)

            # 顧客満足度の計算
            # 在庫充足率と売上実績から推定
            total_inventory = sum(inventory_level.values())
            max_inventory = (
                len(inventory_service.vending_machine_slots) * 50
            )  # 想定最大在庫
            inventory_score = (
                min(total_inventory / max_inventory, 1.0) if max_inventory > 0 else 0.5
            )

            # 売上目標との比較（月間目標: 100万円）
            monthly_target = 1000000
            sales_score = min(sales / monthly_target, 1.0)

            # 総合満足度（3.0-5.0のスケール）
            customer_satisfaction = 3.0 + (inventory_score * 1.0 + sales_score * 1.0)

            metrics_result = {
                "sales": round(sales, 2),
                "profit_margin": round(profit_margin, 3),
                "inventory_level": inventory_level,
                "customer_satisfaction": round(customer_satisfaction, 2),
                "timestamp": datetime.now().isoformat(),
                "inventory_status": {
                    "total_slots": len(inventory_service.vending_machine_slots),
                    "low_stock_count": len(inventory_service.get_low_stock_slots()),
                    "out_of_stock_count": len(
                        inventory_service.get_out_of_stock_slots()
                    ),
                },
                "sales_stats": {
                    "total_revenue": sales,  # 会計システムから取得
                },
            }

            # デバッグログ: 取得したビジネスデータをログ出力
            logger.debug("=== BUSINESS METRICS RETRIEVED ===")
            logger.debug(f"Sales (accounting_system): ¥{sales:.2f}")
            logger.debug(f"Profit Margin: {profit_margin:.3f}")
            logger.debug(f"Inventory Level: {inventory_level}")
            logger.debug(f"Inventory Status: {metrics_result['inventory_status']}")
            logger.debug(f"Customer Satisfaction: {customer_satisfaction:.2f}")
            logger.debug("=== END BUSINESS METRICS ===")

            return metrics_result

        except Exception as e:
            logger.error(f"ビジネスメトリクス取得エラー: {e}", exc_info=True)
            # エラー時はフォールバック値を返す
            return {
                "sales": 0.0,
                "profit_margin": 0.0,
                "inventory_level": {},
                "customer_satisfaction": 3.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def analyze_financial_performance(self) -> Dict[str, Any]:
        """財務パフォーマンスを分析（model_manager経由）"""
        logger.info("Analyzing financial performance using LLM")
        try:
            from src.ai.model_manager import AIMessage

            from src.infrastructure import model_manager

            metrics = self.get_business_metrics()

            messages = [
                AIMessage(role="system", content=self.system_prompt),
                AIMessage(
                    role="user",
                    content=f"""
以下の財務データを分析し、パフォーマンス評価と改善提案を行ってください。

【財務データ】
- 売上: ¥{metrics["sales"]:,}
- 利益率: {metrics["profit_margin"]:.1%}
- 在庫状況: {metrics["inventory_level"]}
- 顧客満足度: {metrics["customer_satisfaction"]}/5.0

【出力形式】
JSON形式で回答してください：
```json
{{
    "analysis": "財務状況の全体的な評価と分析",
    "recommendations": ["改善提案1", "改善提案2", "改善提案3"]
}}
```
""",
                ),
            ]

            response = await model_manager.generate_response(messages, max_tokens=1000)

            if response.success:
                try:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    llm_response = json.loads(content)
                    return {
                        "analysis": llm_response.get(
                            "analysis", "分析できませんでした"
                        ),
                        "recommendations": llm_response.get("recommendations", []),
                        "metrics": metrics,
                    }
                except json.JSONDecodeError:
                    logger.warning(
                        f"財務分析LLMレスポンスパース失敗: {response.content}"
                    )

            # LLM失敗時はハードコードされたレスポンス
            logger.warning("LLM財務分析失敗、ハードコードレスポンスを使用")
            return {
                "analysis": "売上は予算比95%で推移。利益率は良好。",
                "recommendations": ["在庫回転率の改善", "高利益商品の強化"],
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"財務パフォーマンス分析エラー: {e}")
            metrics = self.get_business_metrics()
            return {
                "analysis": f"分析エラー: {str(e)}",
                "recommendations": ["管理者へ連絡してください"],
                "metrics": metrics,
            }

    async def check_inventory_status(self) -> Dict[str, Any]:
        """在庫状況を確認（model_manager経由）"""
        logger.info("Checking inventory status using LLM")
        try:
            from src.ai.model_manager import AIMessage

            from src.infrastructure import model_manager

            metrics = self.get_business_metrics()
            inventory_level = metrics["inventory_level"]

            messages = [
                AIMessage(role="system", content=self.system_prompt),
                AIMessage(
                    role="user",
                    content=f"""
以下の在庫状況を分析し、在庫管理の推奨事項を提案してください。

【現在の在庫状況】
{inventory_level}

【出力形式】
JSON形式で回答してください：
```json
{{
    "status": "在庫状況の全体評価 (normal/critical/low)",
    "low_stock_items": ["在庫の少ない商品名リスト"],
    "reorder_needed": ["発注が必要な商品名リスト"],
    "estimated_stockout": {{"商品名": "在庫切れ予測日"}}
}}
```
""",
                ),
            ]

            response = await model_manager.generate_response(messages, max_tokens=1000)

            if response.success:
                try:
                    import json

                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    llm_response = json.loads(content)
                    return llm_response
                except json.JSONDecodeError:
                    logger.warning(
                        f"在庫状況LLMレスポンスパース失敗: {response.content}"
                    )

            # LLM失敗時はハードコードされたレスポンス
            logger.warning("LLM在庫分析失敗、ハードコードレスポンスを使用")
            return {
                "status": "normal",
                "low_stock_items": ["water"],
                "reorder_needed": ["water"],
                "estimated_stockout": {"water": "2日後"},
            }

        except Exception as e:
            logger.error(f"在庫状況確認エラー: {e}")
            return {
                "status": "error",
                "low_stock_items": [],
                "reorder_needed": [],
                "estimated_stockout": {},
            }

    def update_pricing(self, product: str, price: float) -> Dict[str, Any]:
        """価格戦略を更新"""
        logger.info("Tool update_pricing called")
        logger.info(f"Updating pricing for {product} to {price}")
        return {
            "success": True,
            "product": product,
            "new_price": price,
            "effective_date": datetime.now().isoformat(),
        }

    def assign_restocking_task(
        self, products: List[str], urgency: str = "normal"
    ) -> Dict[str, Any]:
        """補充タスクを割り当て"""
        logger.info("Tool assign_restocking called")
        logger.info(f"Assigning restocking task for {products} with urgency {urgency}")
        task_id = str(uuid4())
        return {
            "task_id": task_id,
            "task_type": "restocking",
            "products": products,
            "urgency": urgency,
            "assigned": True,
            "deadline": (
                datetime.now() + timedelta(hours=2 if urgency == "urgent" else 24)
            ).isoformat(),
        }

    def request_procurement(
        self, products: List[str], quantity: Dict[str, int]
    ) -> Dict[str, Any]:
        """調達を依頼"""
        logger.info(f"Requesting procurement for {products}")
        order_id = str(uuid4())
        return {
            "order_id": order_id,
            "products": products,
            "quantity": quantity,
            "status": "pending",
            "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
        }

    def schedule_maintenance(self, task: str, date: str) -> Dict[str, Any]:
        """メンテナンスをスケジュール"""
        logger.info(f"Scheduling maintenance: {task} on {date}")
        return {"success": True, "task": task, "scheduled_date": date}

    async def coordinate_employee_tasks(self) -> Dict[str, Any]:
        """発注/補充が必要な場合に従業員1人にメール通知 + 新商品発注処理"""
        logger.info("Coordinating employee tasks")

        notifications = []
        employees_status = {}

        # === 1. 在庫補充タスク ===
        inventory_status = await self.check_inventory_status()
        low_stock_items = inventory_status.get("low_stock_items", [])

        if low_stock_items:
            notification = {
                "recipient": "employee@vending-company.com",
                "subject": "在庫補充依頼",
                "body": f"以下の商品が在庫不足です。補充をお願いします: {', '.join(low_stock_items)}",
                "priority": "normal",
                "timestamp": datetime.now().isoformat(),
                "task_type": "restock",
            }
            notifications.append(notification)
            employees_status["restock"] = low_stock_items
            logger.info(f"在庫補充通知送信: {low_stock_items}")

        # === 2. 新商品発注タスク ===
        # 在庫データを基に新商品検索クエリを生成
        try:
            # 在庫状況から商品カテゴリを把握
            metrics = self.get_business_metrics()
            inventory_level = metrics.get("inventory_level", {})
            sales = metrics.get("sales", 0)

            # カテゴリ別在庫を確認
            drink_categories = [
                item
                for item in inventory_level.keys()
                if "コーラ" in item or "飲料" in item or "ジュース" in item
            ]
            food_categories = [
                item
                for item in inventory_level.keys()
                if "チップス" in item or "ヌードル" in item or "お菓子" in item
            ]

            # 売上実績に基づいて検索クエリを決定
            if sales > 1000:  # 売上が良い場合
                search_query = "人気飲料 新商品"
                logger.info("売上好調のため、新商品飲料を検索")
            elif (
                drink_categories
                and min([inventory_level.get(cat, 0) for cat in drink_categories]) < 5
            ):  # 飲料在庫が少ない場合
                search_query = "人気清涼飲料 ボトル飲料"
                logger.info("飲料在庫不足のため、供給安定した飲料を検索")
            elif food_categories:
                search_query = "人気スナック 健康志向"
                logger.info("既存食品を補完する人気スナックを検索")
            else:
                search_query = "人気飲料"
                logger.info("デフォルトで人気飲料を検索")

            logger.info(f"生成された検索クエリ: {search_query}")

            # Shared Toolsから商品検索機能を使用
            from src.agents.shared_tools import shared_registry

            search_tool = shared_registry.get_tool("market_search")
            if search_tool:
                search_results = await search_tool.asearch(query=search_query)
                logger.info(
                    f"検索結果取得: {len(search_results) if search_results else 0}件 (クエリ: {search_query})"
                )
                recommended_products = (
                    search_results[:2] if search_results else []
                )  # 上位2つ
            else:
                recommended_products = []
                logger.warning("検索ツールが利用できません")

            if recommended_products:
                procurement_tasks = []
                for product in recommended_products[:2]:  # dict形式を想定
                    # Procurement AgentからShared Toolsに変更
                    procurement_tool = shared_registry.get_tool("procurement_order")
                    if procurement_tool:
                        procurement_result = await procurement_tool.aexecute(
                            product_info={
                                "product_name": product.get("name", "") or product,
                                "recommended_quantity": 10,
                            },
                            supplier_info={
                                "name": "Search Supplier",
                                "url": product.get("url", ""),
                                "price": product.get("price", 150),
                            },
                        )

                        if procurement_result.get("success"):
                            order = procurement_result.get("order", {})
                            procurement_tasks.append(
                                {
                                    "product": product.get("name", "") or product,
                                    "order_id": order.get("order_id", "unknown"),
                                }
                            )

                if procurement_tasks:
                    procurement_notification = {
                        "recipient": "employee@vending-company.com",
                        "subject": "新商品発注完了通知",
                        "body": f"以下の新商品を発注しました。入荷管理をお願いします:\n"
                        + "\n".join(
                            [
                                f"- {t['product']} (注文ID: {t['order_id']})"
                                for t in procurement_tasks
                            ]
                        ),
                        "priority": "high",
                        "timestamp": datetime.now().isoformat(),
                        "task_type": "new_procurement",
                        "orders": procurement_tasks,
                    }
                    notifications.append(procurement_notification)
                    employees_status["new_procurement"] = [
                        t["product"] for t in procurement_tasks
                    ]
                    logger.info(f"新商品発注通知送信: {len(procurement_tasks)}件")

        except Exception as e:
            logger.error(f"新商品発注プロセスエラー: {e}")

        # === 結果返却 ===
        if notifications:
            return {
                "active_tasks": len(notifications),
                "completed_today": 0,
                "pending": len(notifications),
                "notifications_sent": notifications,
                "employees": {"employee@vending-company.com": employees_status},
            }
        else:
            return {
                "active_tasks": 0,
                "completed_today": 0,
                "pending": 0,
                "notifications_sent": [],
                "employees": {"employee@vending-company.com": "特記事項なし"},
            }

    def respond_to_customer_inquiry(
        self, customer_id: str, inquiry: str
    ) -> Dict[str, Any]:
        """顧客問い合わせに対応"""
        logger.info(f"Responding to customer {customer_id} inquiry")
        return {
            "customer_id": customer_id,
            "inquiry": inquiry,
            "response": "お問い合わせありがとうございます。担当者が確認して折り返しご連絡いたします。",
            "status": "responded",
        }

    def handle_customer_complaint(
        self, customer_id: str, complaint: str
    ) -> Dict[str, Any]:
        """顧客苦情を処理"""
        logger.info(f"Handling complaint from customer {customer_id}")
        return {
            "customer_id": customer_id,
            "complaint": complaint,
            "resolution": "商品の返金処理を行い、次回使用可能なクーポンを発行しました。",
            "status": "resolved",
            "compensation": "500円クーポン",
        }

    def collect_customer_feedback(self) -> Dict[str, Any]:
        """顧客フィードバックを収集"""
        logger.info("Collecting customer feedback")
        return {
            "feedback_count": 15,
            "average_rating": 4.2,
            "top_requests": ["新しいフレーバー", "温かい飲み物", "健康志向商品"],
            "trends": "健康志向商品への関心が高まっている",
        }

    def create_customer_engagement_campaign(self, campaign_type: str) -> Dict[str, Any]:
        """エンゲージメントキャンペーンを作成"""
        logger.info(f"Creating {campaign_type} campaign")
        return {
            "campaign_type": campaign_type,
            "target": "全顧客",
            "duration": "2週間",
            "expected_impact": "売上10%増",
            "status": "planned",
        }

    async def start_management_session(self, session_type: str) -> str:
        """管理セッションを開始"""
        session_id = str(uuid4())
        self.current_session = SessionInfo(
            session_id=session_id, session_type=session_type, start_time=datetime.now()
        )

        logger.info(f"Started {session_type} session: {session_id}")
        return session_id

    async def end_management_session(self) -> Dict[str, Any]:
        """管理セッションを終了"""
        if not self.current_session:
            raise ValueError("No active session")

        self.current_session.end_time = datetime.now()
        duration = self.current_session.end_time - self.current_session.start_time

        session_summary = {
            "session_id": self.current_session.session_id,
            "session_type": self.current_session.session_type,
            "duration": str(duration),
            "decisions_count": len(self.current_session.decisions_made),
            "actions_count": len(self.current_session.actions_executed),
        }

        logger.info(f"Ended session {self.current_session.session_id}")
        self.current_session = None

        return session_summary

    async def make_strategic_decision(self, context: str) -> Dict[str, Any]:
        """戦略的意思決定を行う（model_manager経由）"""
        if not self.current_session:
            raise ValueError("No active session. Start a session first.")

        logger.info("Making strategic decision using model_manager")

        try:
            from src.ai.model_manager import AIMessage

            from src.infrastructure import model_manager

            # LLMに渡すメッセージを作成
            user_content = f"""
以下のビジネス状況を分析し、戦略的意思決定を行ってください。

【状況】
{context}

【出力形式】
JSON形式で以下の構造で回答してください：
```json
{{
    "decision": "決定事項の簡潔な要約",
    "rationale": "決定の根拠と理由",
    "actions": ["具体的なアクション1", "具体的なアクション2"]
}}
```

注意: JSON形式のみで回答し、他のテキストは含めないでください。
"""

            messages = [
                AIMessage(role="system", content=self.system_prompt),
                AIMessage(role="user", content=user_content),
            ]

            # デバッグログ: LLMプロンプト内容を出力（初回のみ）
            if not self._system_prompt_logged:
                logger.debug("=== LLM PROMPT ===")
                logger.debug(
                    f"System Prompt: {self.system_prompt[:500]}..."
                )  # システムプロンプトは長すぎるので一部のみ
                logger.debug(f"User Content: {user_content}")
                logger.debug("=== END PROMPT ===")
                self._system_prompt_logged = True
            else:
                logger.debug("LLM called with established system prompt")

            # model_manager経由でLLM呼び出し
            response = await model_manager.generate_response(messages, max_tokens=1000)

            if not response.success:
                # フォールバックとしてハードコードされた決定を使用
                logger.warning(
                    f"LLM呼び出し失敗 ({response.error_message})、ハードコードされた決定を使用"
                )
                decision = {
                    "context": context,
                    "decision": "在庫水準を維持しつつ、売れ筋商品の価格を最適化する",
                    "rationale": "データ分析の結果、価格調整により利益率5%改善が見込める",
                    "actions": ["価格更新", "在庫補充依頼"],
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # LLMレスポンスをパース
                try:
                    import json

                    # JSON部分を抽出（コードブロックがある場合）
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:]  # ```jsonを削除
                    if content.endswith("```"):
                        content = content[:-3]  # ```を削除
                    content = content.strip()

                    # デバッグログ: LLMレスポンス内容を出力
                    logger.debug("=== LLM RESPONSE ===")
                    logger.debug(f"Raw Response: {content}")
                    logger.debug("=== END RESPONSE ===")

                    llm_response = json.loads(content)

                    decision = {
                        "context": context,
                        "decision": llm_response.get(
                            "decision", "決定できませんでした"
                        ),
                        "rationale": llm_response.get("rationale", "理由不明"),
                        "actions": llm_response.get("actions", []),
                        "timestamp": datetime.now().isoformat(),
                        "llm_used": response.model_used,
                    }

                    logger.info(f"LLM意思決定完了: {decision['decision']}")
                    logger.debug(
                        f"LLM意思決定詳細: rationale='{decision['rationale']}', actions={decision.get('actions', [])}"
                    )

                except json.JSONDecodeError as e:
                    logger.error(f"LLMレスポンスのパース失敗: {e}")
                    logger.error(f"LLMレスポンス内容: {response.content}")
                    # フォールバック
                    decision = {
                        "context": context,
                        "decision": "在庫水準を維持しつつ、売れ筋商品の価格を最適化する",
                        "rationale": f"LLMレスポンスのパースに失敗したため、デフォルト決定を使用: {response.content[:200]}",
                        "actions": ["価格更新", "在庫補充依頼"],
                        "timestamp": datetime.now().isoformat(),
                    }

        except Exception as e:
            logger.error(f"戦略的意思決定中にエラー発生: {e}", exc_info=True)
            decision = {
                "context": context,
                "decision": "エラーが発生したため、デフォルト意思決定",
                "rationale": f"システムエラー: {str(e)}",
                "actions": ["管理者への連絡"],
                "timestamp": datetime.now().isoformat(),
            }

        # Execute actions using system data
        executed_actions = []
        for action in decision.get("actions", []):
            if "在庫補充" in action or "補充" in action:
                try:
                    inventory_status = await self.check_inventory_status()
                    low_stock_items = inventory_status.get("low_stock_items", [])
                    if low_stock_items:
                        result = self.assign_restocking_task(
                            low_stock_items, urgency="normal"
                        )
                        executed_actions.append(
                            f"Executed restocking for {low_stock_items}, task_id: {result.get('task_id')}"
                        )
                except Exception as e:
                    logger.error(f"Error executing restocking: {e}")
            elif "価格更新" in action or "価格" in action:
                try:
                    if SAMPLE_PRODUCTS:
                        product = SAMPLE_PRODUCTS[0]  # Using first registered product
                        new_price = round(product.price * 1.05, 0)  # Example adjustment
                        result = self.update_pricing(product.product_id, new_price)
                        executed_actions.append(
                            f"Executed pricing update for {product.product_id} to ¥{new_price}"
                        )
                except Exception as e:
                    logger.error(f"Error executing pricing update: {e}")
            else:
                executed_actions.append(f"Action '{action}' not executable")

        # Log executed actions
        for ea in executed_actions:
            logger.info(f"Executed action: {ea}")

        decision["executed_actions"] = executed_actions

        self.current_session.decisions_made.append(decision)
        return decision

    async def morning_routine(self) -> Dict[str, Any]:
        """朝の業務ルーチン"""
        session_id = await self.start_management_session("morning_routine")

        try:
            # 夜間データ確認
            overnight_data = self.get_business_metrics()

            # 朝の分析
            morning_analysis = f"""
            昨夜の事業データを確認し、今日の業務優先順位を決定してください。
            
            【夜間データ】
            - 売上実績: {overnight_data["sales"]}
            - 在庫状況: {overnight_data["inventory_level"]}
            - 顧客満足度: {overnight_data["customer_satisfaction"]}
            
            【判断項目】
            1. 緊急対応が必要な事項
            2. 今日の重点業務
            3. 従業員への指示事項
            """

            decisions = await self.make_strategic_decision(morning_analysis)

            return {
                "session_id": session_id,
                "session_type": "morning_routine",
                "overnight_data": overnight_data,
                "decisions": decisions,
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def midday_check(self) -> Dict[str, Any]:
        """昼の業務チェック"""
        session_id = await self.start_management_session("midday_check")

        try:
            metrics = self.get_business_metrics()
            financial_analysis = await self.analyze_financial_performance()

            midday_analysis = f"""
            午前中の業績を確認し、午後の調整を行ってください。
            
            【午前実績】
            - 売上: {metrics["sales"]}
            - 利益率: {metrics["profit_margin"]}
            """

            decisions = await self.make_strategic_decision(midday_analysis)

            return {
                "session_id": session_id,
                "session_type": "midday_check",
                "metrics": metrics,
                "analysis": financial_analysis,
                "decisions": decisions,
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def evening_summary(self) -> Dict[str, Any]:
        """夕方の業務総括"""
        session_id = await self.start_management_session("evening_summary")

        try:
            daily_performance = self.get_business_metrics()
            inventory_status = await self.check_inventory_status()

            evening_analysis = f"""
            今日一日の業績を総括し、明日への改善点を特定してください。
            
            【今日の実績】
            - 売上: {daily_performance["sales"]}
            - 利益率: {daily_performance["profit_margin"]}
            - 在庫状況: {inventory_status["status"]}
            
            【分析項目】
            1. 今日の成功要因
            2. 改善が必要な領域
            3. 明日の重点課題
            """

            decisions = await self.make_strategic_decision(evening_analysis)

            return {
                "session_id": session_id,
                "session_type": "evening_summary",
                "daily_performance": daily_performance,
                "inventory_status": inventory_status,
                "decisions": decisions,
                "lessons_learned": [
                    "在庫管理の改善が必要",
                    "顧客満足度を維持できた",
                ],
                "status": "completed",
            }

        finally:
            await self.end_management_session()

    async def feedback_engine(self) -> Dict[str, Any]:
        """夕方の業務総括"""
        session_id = await self.start_management_session("evening_summary")

        try:
            daily_performance = self.get_business_metrics()
            inventory_status = await self.check_inventory_status()

            evening_analysis = f"""
            今日一日の業績を総括し、明日への改善点を特定してください。
            
            【今日の実績】
            - 売上: {daily_performance["sales"]}
            - 利益率: {daily_performance["profit_margin"]}
            - 在庫状況: {inventory_status["status"]}
            
            【分析項目】
            1. 今日の成功要因
            2. 改善が必要な領域
            3. 明日の重点課題
            """

            decisions = await self.make_strategic_decision(evening_analysis)

            return {
                "session_id": session_id,
                "session_type": "evening_summary",
                "daily_performance": daily_performance,
                "inventory_status": inventory_status,
                "decisions": decisions,
                "lessons_learned": [
                    "在庫管理の改善が必要",
                    "顧客満足度を維持できた",
                ],
                "status": "completed",
            }

        finally:
            await self.end_management_session()


# グローバルインスタンス
management_agent = SessionBasedManagementAgent(provider="openai")
