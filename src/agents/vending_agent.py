"""
自動販売機運営エージェント

このエージェントは自動販売機の全体的な運営を管理し、
他のエージェントと連携して店舗運営を行います。
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.accounting.journal_entry import journal_processor
from src.agents.customer_agent import customer_agent
from src.agents.search_agent import search_agent
from src.ai.model_manager import AIMessage, AIResponse, ModelManager, model_manager
from src.config.settings import settings
from src.models.product import SAMPLE_PRODUCTS, Product
from src.models.transaction import PaymentMethod, Transaction, TransactionStatus
from src.services.inventory_service import inventory_service
from src.services.payment_service import payment_service

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """運営モード"""

    NORMAL = "normal"  # 通常運営
    LOW_STOCK = "low_stock"  # 在庫不足警戒
    HIGH_DEMAND = "high_demand"  # 高需要モード
    MAINTENANCE = "maintenance"  # メンテナンスモード
    EMERGENCY = "emergency"  # 緊急モード


@dataclass
class OperationStatus:
    """運営状況"""

    mode: OperationMode
    total_products: int
    available_products: int
    out_of_stock_products: int
    total_revenue_today: float
    transaction_count_today: int
    system_health: float  # 0-1の範囲
    last_updated: datetime
    alerts: List[str]


@dataclass
class VendingDecision:
    """販売決定"""

    action_type: str  # "price_adjustment", "promotion", "restock", "maintenance"
    product_id: Optional[str]
    decision: str
    reasoning: str
    confidence: float
    priority: int  # 1-10の優先度
    timestamp: datetime


class VendingAgent:
    """
    自動販売機運営エージェント

    自動販売機の全体的な運営を管理し、以下の役割を果たします：
    - 商品販売の管理と最適化
    - 在庫状況の監視と補充計画
    - 価格戦略の決定
    - 他のエージェントとの連携統括
    - 運営状況の監視と異常検出
    """

    def __init__(self):
        self.model_manager = model_manager
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.search_agent = search_agent
        self.customer_agent = customer_agent
        self.journal_processor = journal_processor

        # 運営状態
        self.current_mode = OperationMode.NORMAL
        self.operation_history: List[VendingDecision] = []
        self.performance_metrics: Dict[str, Any] = {}

        # 設定
        self.restock_threshold = 5  # 在庫補充閾値
        self.price_adjustment_range = 0.1  # 価格調整範囲（±10%）
        self.maintenance_check_interval = 3600  # メンテナンスチェック間隔（秒）

    async def initialize(self) -> bool:
        """エージェントの初期化"""
        try:
            logger.info("自動販売機運営エージェントを初期化中...")

            # システム状態チェック
            system_ready = await self._check_system_readiness()
            if not system_ready:
                logger.error("システム準備が完了していません")
                return False

            # 初期在庫確認
            await self._initialize_inventory()

            # 初期運営モード設定
            await self._determine_initial_mode()

            logger.info("自動販売機運営エージェントの初期化完了")
            return True

        except Exception as e:
            logger.error(f"エージェント初期化エラー: {e}")
            return False

    async def _check_system_readiness(self) -> bool:
        """システム準備状態の確認"""
        try:
            # 必須サービスの確認
            services_status = []

            # 在庫サービス
            try:
                summary = self.inventory_service.get_inventory_summary()
                services_status.append(True)
            except Exception as e:
                logger.error(f"在庫サービス確認エラー: {e}")
                services_status.append(False)

            # 決済サービス
            try:
                # 決済サービスはモデルベースなので常に利用可能
                services_status.append(True)
            except Exception as e:
                logger.error(f"決済サービス確認エラー: {e}")
                services_status.append(False)

            # 検索エージェント
            try:
                stats = self.search_agent.get_search_stats()
                services_status.append(True)
            except Exception as e:
                logger.error(f"検索エージェント確認エラー: {e}")
                services_status.append(False)

            return all(services_status)

        except Exception as e:
            logger.error(f"システム準備状態確認エラー: {e}")
            return False

    async def _initialize_inventory(self):
        """初期在庫の設定"""
        try:
            logger.info("初期在庫を設定中...")

            # サンプル商品の在庫を設定
            for product in SAMPLE_PRODUCTS:
                # 在庫サービスに商品を登録（実際の実装ではデータベースに保存）
                logger.info(
                    f"商品登録: {product.name} - 在庫: {product.stock_quantity}"
                )

            logger.info("初期在庫設定完了")

        except Exception as e:
            logger.error(f"初期在庫設定エラー: {e}")

    async def _determine_initial_mode(self):
        """初期運営モードの決定"""
        try:
            # 在庫状況に基づいてモードを決定
            summary = self.inventory_service.get_inventory_summary()

            if summary.out_of_stock_slots > 0:
                self.current_mode = OperationMode.LOW_STOCK
                logger.warning("在庫不足モードで開始")
            else:
                self.current_mode = OperationMode.NORMAL
                logger.info("通常モードで開始")

        except Exception as e:
            logger.error(f"初期モード決定エラー: {e}")
            self.current_mode = OperationMode.NORMAL

    async def process_sale(
        self, product_id: str, payment_method: PaymentMethod, customer_id: str = None
    ) -> Dict[str, Any]:
        """
        商品販売を処理

        Args:
            product_id: 商品ID
            payment_method: 決済方法
            customer_id: 顧客ID（オプション）

        Returns:
            販売結果
        """
        try:
            logger.info(f"商品販売処理開始: {product_id}")

            # 1. 在庫確認
            inventory_check = await self._check_inventory_availability(product_id)
            if not inventory_check["available"]:
                return {
                    "success": False,
                    "error": "在庫不足",
                    "message": f"{inventory_check['product_name']}の在庫が不足しています",
                }

            # 2. 価格取得
            product = inventory_check["product"]
            current_price = await self._get_current_price(product_id)

            # 3. 決済処理
            payment_result = await self.payment_service.process_payment(
                current_price, payment_method
            )

            if not payment_result.success:
                return {
                    "success": False,
                    "error": "決済失敗",
                    "message": "決済処理に失敗しました",
                }

            # 4. 在庫更新
            dispense_result = await self._dispense_product(product_id)

            # 5. 取引記録
            transaction = await self._record_transaction(
                product_id, current_price, payment_method, customer_id
            )

            # 6. 会計処理
            await self._process_accounting(transaction)

            # 7. 顧客対応（オプション）
            if customer_id:
                await self._handle_customer_interaction(customer_id, product_id)

            # 8. 運営状況更新
            await self._update_operation_status()

            logger.info(f"商品販売処理完了: {product_id}")

            return {
                "success": True,
                "transaction_id": transaction.transaction_id,
                "product_name": product.name,
                "amount": current_price,
                "change": payment_result.change
                if hasattr(payment_result, "change")
                else 0,
                "message": "購入ありがとうございます",
            }

        except Exception as e:
            logger.error(f"商品販売処理エラー: {e}")
            return {
                "success": False,
                "error": "システムエラー",
                "message": "販売処理中にエラーが発生しました",
            }

    async def _check_inventory_availability(self, product_id: str) -> Dict[str, Any]:
        """在庫利用可能性の確認"""
        try:
            # 商品情報の取得
            product = next(
                (p for p in SAMPLE_PRODUCTS if p.product_id == product_id), None
            )

            if not product:
                return {"available": False, "product_name": "不明な商品"}

            # 在庫数の確認（簡易実装）
            # 実際の実装ではデータベースから在庫情報を取得
            current_stock = getattr(product, "stock_quantity", 0)

            return {
                "available": current_stock > 0,
                "product": product,
                "product_name": product.name,
                "current_stock": current_stock,
            }

        except Exception as e:
            logger.error(f"在庫確認エラー: {e}")
            return {"available": False, "product_name": "エラー"}

    async def _get_current_price(self, product_id: str) -> float:
        """現在の価格を取得"""
        try:
            product = next(
                (p for p in SAMPLE_PRODUCTS if p.product_id == product_id), None
            )

            if not product:
                raise ValueError(f"商品が見つかりません: {product_id}")

            # 現在の価格戦略に基づいて価格を調整（簡易実装）
            base_price = product.price

            # 需要予測に基づく価格調整
            demand_factor = await self._calculate_demand_factor(product_id)

            # 価格調整を適用
            adjusted_price = base_price * (1 + demand_factor * 0.1)  # ±10%の調整

            return max(adjusted_price, base_price * 0.5)  # 最低価格制限

        except Exception as e:
            logger.error(f"価格取得エラー: {e}")
            # エラー時はベース価格を返す
            product = next(
                (p for p in SAMPLE_PRODUCTS if p.product_id == product_id), None
            )
            return product.price if product else 0

    async def _calculate_demand_factor(self, product_id: str) -> float:
        """需要要因を計算（簡易実装）"""
        try:
            # 現在の時間による需要変動
            current_hour = datetime.now().hour

            # 時間帯別需要パターン
            if 7 <= current_hour <= 9:  # 朝
                demand_factor = 0.2
            elif 11 <= current_hour <= 14:  # 昼
                demand_factor = 0.3
            elif 17 <= current_hour <= 19:  # 夕方
                demand_factor = 0.25
            else:  # その他
                demand_factor = 0.0

            # 商品別需要調整
            if "飲み物" in product_id or "drink" in product_id:
                demand_factor *= 1.2  # 飲み物は需要高め
            elif "スナック" in product_id or "snack" in product_id:
                demand_factor *= 0.8  # スナックは需要控えめ

            return demand_factor

        except Exception as e:
            logger.error(f"需要要因計算エラー: {e}")
            return 0.0

    async def _dispense_product(self, product_id: str) -> bool:
        """商品を排出"""
        try:
            # 在庫サービスで商品排出を処理
            # 実際の実装ではハードウェア制御を行う

            logger.info(f"商品排出処理: {product_id}")

            # シミュレーションでは常に成功
            return True

        except Exception as e:
            logger.error(f"商品排出エラー: {e}")
            return False

    async def _record_transaction(
        self,
        product_id: str,
        amount: float,
        payment_method: PaymentMethod,
        customer_id: str = None,
    ) -> Transaction:
        """取引を記録"""
        try:
            transaction = Transaction(
                transaction_id=f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product_id}",
                product_id=product_id,
                amount=amount,
                payment_method=payment_method,
                status=TransactionStatus.COMPLETED,
                customer_id=customer_id,
                timestamp=datetime.now(),
            )

            logger.info(f"取引記録完了: {transaction.transaction_id}")
            return transaction

        except Exception as e:
            logger.error(f"取引記録エラー: {e}")
            # エラー時はダミーの取引を返す
            return Transaction(
                transaction_id="error_transaction",
                product_id=product_id,
                amount=amount,
                payment_method=payment_method,
                status=TransactionStatus.FAILED,
                timestamp=datetime.now(),
            )

    async def _process_accounting(self, transaction: Transaction):
        """会計処理を実行"""
        try:
            # 仕訳処理を呼び出し
            await self.journal_processor.record_sale(
                {
                    "transaction_id": transaction.transaction_id,
                    "product_name": transaction.product_id,
                    "amount": transaction.amount,
                    "payment_method": transaction.payment_method.value,
                }
            )

            logger.info(f"会計処理完了: {transaction.transaction_id}")

        except Exception as e:
            logger.error(f"会計処理エラー: {e}")

    async def _handle_customer_interaction(self, customer_id: str, product_id: str):
        """顧客対応を処理"""
        try:
            # 顧客エージェントで対応を処理
            # 実際の実装では会話サービスを呼び出し

            logger.info(f"顧客対応処理: {customer_id} - {product_id}")

        except Exception as e:
            logger.error(f"顧客対応処理エラー: {e}")

    async def _update_operation_status(self):
        """運営状況を更新"""
        try:
            # 在庫状況の確認
            summary = self.inventory_service.get_inventory_summary()

            # 運営モードの更新
            if summary.out_of_stock_slots > 0:
                if self.current_mode != OperationMode.LOW_STOCK:
                    self.current_mode = OperationMode.LOW_STOCK
                    logger.warning("運営モードを在庫不足モードに変更")
            elif (
                summary.out_of_stock_slots == 0
                and self.current_mode == OperationMode.LOW_STOCK
            ):
                self.current_mode = OperationMode.NORMAL
                logger.info("運営モードを通常モードに変更")

        except Exception as e:
            logger.error(f"運営状況更新エラー: {e}")

    async def check_and_plan_restocking(self) -> Dict[str, Any]:
        """
        在庫補充計画を作成

        Returns:
            補充計画
        """
        try:
            logger.info("在庫補充計画を作成中...")

            restock_plan = {
                "required_actions": [],
                "priority_products": [],
                "estimated_cost": 0,
                "recommendations": [],
            }

            # 在庫不足商品の確認
            low_stock_products = []

            for product in SAMPLE_PRODUCTS:
                # 在庫数の確認（簡易実装）
                current_stock = getattr(product, "stock_quantity", 0)

                if current_stock <= self.restock_threshold:
                    low_stock_products.append(
                        {
                            "product_id": product.product_id,
                            "product_name": product.name,
                            "current_stock": current_stock,
                            "recommended_quantity": 20,  # 推奨補充量
                            "priority": "high" if current_stock <= 2 else "medium",
                        }
                    )

            if not low_stock_products:
                return {
                    "success": True,
                    "message": "在庫は十分です",
                    "restock_plan": restock_plan,
                }

            # 価格調査
            total_estimated_cost = 0

            for product_info in low_stock_products:
                try:
                    # 検索エージェントで価格調査
                    price_comparison = await self.search_agent.compare_prices(
                        product_info["product_name"]
                    )

                    if price_comparison.best_price:
                        estimated_cost = (
                            price_comparison.best_price
                            * product_info["recommended_quantity"]
                        )
                        product_info["estimated_unit_cost"] = (
                            price_comparison.best_price
                        )
                        product_info["estimated_total_cost"] = estimated_cost
                        total_estimated_cost += estimated_cost

                        restock_plan["required_actions"].append(
                            {
                                "action": "procurement",
                                "product_id": product_info["product_id"],
                                "product_name": product_info["product_name"],
                                "quantity": product_info["recommended_quantity"],
                                "estimated_cost": estimated_cost,
                                "supplier_recommendation": price_comparison.recommendation,
                            }
                        )

                except Exception as e:
                    logger.error(
                        f"価格調査エラー ({product_info['product_name']}): {e}"
                    )

            restock_plan["estimated_cost"] = total_estimated_cost
            restock_plan["priority_products"] = low_stock_products

            # AIで補充優先度を分析
            if low_stock_products:
                priority_analysis = await self._analyze_restock_priority(
                    low_stock_products
                )
                restock_plan["recommendations"] = priority_analysis

            logger.info(f"在庫補充計画作成完了: {len(low_stock_products)}商品")

            return {
                "success": True,
                "message": f"在庫補充が必要な商品が{len(low_stock_products)}件あります",
                "restock_plan": restock_plan,
            }

        except Exception as e:
            logger.error(f"在庫補充計画作成エラー: {e}")
            return {"success": False, "error": str(e), "restock_plan": {}}

    async def _analyze_restock_priority(
        self, low_stock_products: List[Dict]
    ) -> List[str]:
        """補充優先度の分析"""
        try:
            # AIプロンプトの構築
            products_text = "\n".join(
                [
                    f"- {p['product_name']}: 在庫{p['current_stock']}個, 優先度{p['priority']}"
                    for p in low_stock_products
                ]
            )

            prompt = f"""
            在庫不足商品リスト:
            {products_text}

            上記の商品の補充優先度を分析し、以下のJSON形式で応答してください：
            {{
                "priority_order": [
                    {{"product_name": "商品名", "priority": "high/medium/low", "reasoning": "理由"}}
                ],
                "overall_strategy": "全体戦略の提案",
                "cost_considerations": "コスト考慮事項"
            }}

            分析のポイント：
            1. 商品の人気度と売上への影響
            2. 在庫切れリスクの評価
            3. コスト効率の考慮
            4. 補充タイミングの提案
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは経験豊富な店舗運営マネージャーです。在庫管理と補充戦略に詳しく、データに基づいた意思決定を行います。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=400
            )

            if response.success:
                # 応答をパースして推奨事項を抽出
                recommendations = self._parse_priority_analysis(response.content)
                return recommendations
            else:
                return [
                    f"AI分析が失敗したため、標準的な優先度で処理します: {response.error_message}"
                ]

        except Exception as e:
            logger.error(f"補充優先度分析エラー: {e}")
            return ["優先度分析に失敗しました。在庫数の少ない順で処理してください。"]

    def _parse_priority_analysis(self, ai_response: str) -> List[str]:
        """優先度分析結果のパース"""
        try:
            # 簡易的なパース（実際の実装ではより堅牢なJSONパースを行う）
            recommendations = []

            if "優先度" in ai_response:
                # 日本語の応答から推奨事項を抽出
                lines = ai_response.split("\n")
                for line in lines:
                    if any(
                        keyword in line for keyword in ["優先", "推奨", "提案", "考慮"]
                    ):
                        recommendations.append(line.strip())

            if not recommendations:
                recommendations = ["AI分析結果をパースできませんでした"]

            return recommendations

        except Exception as e:
            logger.error(f"優先度分析結果パースエラー: {e}")
            return ["パースエラーにより標準優先度を適用します"]

    async def optimize_pricing(self) -> Dict[str, Any]:
        """
        価格最適化を実行

        Returns:
            価格最適化結果
        """
        try:
            logger.info("価格最適化を実行中...")

            optimization_results = {
                "price_adjustments": [],
                "overall_strategy": "",
                "expected_impact": {},
                "recommendations": [],
            }

            # 商品ごとに価格最適化を分析
            for product in SAMPLE_PRODUCTS:
                try:
                    # 需要予測を取得
                    demand_forecast = self.payment_service.get_demand_forecast(
                        product.product_id, days=7
                    )

                    # 現在の価格戦略を評価
                    current_price = product.price
                    optimal_price = (
                        self.payment_service.sales_model.calculate_optimal_price(
                            product.product_id, product.cost
                        )
                    )

                    # 価格調整の必要性を判断
                    price_diff_ratio = (
                        abs(optimal_price - current_price) / current_price
                    )

                    if price_diff_ratio > 0.05:  # 5%以上の差がある場合
                        try:
                            expected_demand_change = demand_forecast.get(
                                "summary", {}
                            ).get("total_predicted_demand", 0)
                        except (AttributeError, KeyError):
                            expected_demand_change = 0

                        adjustment = {
                            "product_id": product.product_id,
                            "product_name": product.name,
                            "current_price": current_price,
                            "optimal_price": optimal_price,
                            "adjustment_ratio": (optimal_price - current_price)
                            / current_price,
                            "expected_demand_change": expected_demand_change,
                            "priority": "high" if price_diff_ratio > 0.1 else "medium",
                        }

                        optimization_results["price_adjustments"].append(adjustment)

                except Exception as e:
                    logger.error(f"価格最適化エラー ({product.name}): {e}")

            # AIで全体戦略を生成
            if optimization_results["price_adjustments"]:
                strategy = await self._generate_pricing_strategy(
                    optimization_results["price_adjustments"]
                )
                optimization_results["overall_strategy"] = strategy

            logger.info(
                f"価格最適化完了: {len(optimization_results['price_adjustments'])}件の調整候補"
            )

            return {"success": True, "optimization_results": optimization_results}

        except Exception as e:
            logger.error(f"価格最適化エラー: {e}")
            return {"success": False, "error": str(e), "optimization_results": {}}

    async def _generate_pricing_strategy(self, adjustments: List[Dict]) -> str:
        """価格戦略を生成"""
        try:
            # AIプロンプトの構築
            adjustments_text = "\n".join(
                [
                    f"- {adj['product_name']}: 現在{adj['current_price']}円 → 最適{adj['optimal_price']:.0f}円 ({adj['adjustment_ratio']:+.1%})"
                    for adj in adjustments
                ]
            )

            prompt = f"""
            価格調整候補:
            {adjustments_text}

            上記の価格調整候補に基づいて、全体的な価格戦略を提案してください。
            以下のポイントを考慮：
            1. 顧客満足度への影響
            2. 売上・利益への影響
            3. 競合他社との関係
            4. 実施タイミングと段階的アプローチ
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは価格戦略の専門家です。データに基づいた最適な価格戦略を提案します。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=300
            )

            if response.success:
                return response.content.strip()
            else:
                return "価格調整の分析に失敗しました。慎重に調整を実施してください。"

        except Exception as e:
            logger.error(f"価格戦略生成エラー: {e}")
            return "価格戦略の生成に失敗しました。"

    async def get_operation_status(self) -> OperationStatus:
        """現在の運営状況を取得"""
        try:
            # 在庫状況の取得
            summary = self.inventory_service.get_inventory_summary()

            # 本日の売上情報（簡易実装）
            today_revenue = 0  # 実際の実装ではデータベースから取得
            today_transactions = 0  # 実際の実装ではデータベースから取得

            # システム健全性の評価
            system_health = await self._evaluate_system_health()

            # アラートの生成
            alerts = await self._generate_alerts(summary)

            return OperationStatus(
                mode=self.current_mode,
                total_products=len(SAMPLE_PRODUCTS),
                available_products=len(SAMPLE_PRODUCTS) - summary.out_of_stock_slots,
                out_of_stock_products=summary.out_of_stock_slots,
                total_revenue_today=today_revenue,
                transaction_count_today=today_transactions,
                system_health=system_health,
                last_updated=datetime.now(),
                alerts=alerts,
            )

        except Exception as e:
            logger.error(f"運営状況取得エラー: {e}")
            return OperationStatus(
                mode=OperationMode.EMERGENCY,
                total_products=0,
                available_products=0,
                out_of_stock_products=0,
                total_revenue_today=0,
                transaction_count_today=0,
                system_health=0.0,
                last_updated=datetime.now(),
                alerts=[f"運営状況取得エラー: {str(e)}"],
            )

    async def _evaluate_system_health(self) -> float:
        """システム健全性を評価（0-1の範囲）"""
        try:
            health_score = 1.0

            # 在庫状況による減点
            summary = self.inventory_service.get_inventory_summary()
            if summary.out_of_stock_slots > 0:
                health_score -= 0.2 * min(
                    summary.out_of_stock_slots / len(SAMPLE_PRODUCTS), 1.0
                )

            # サービス状態による減点（簡易評価）
            # 実際の実装では各サービスの状態をチェック

            return max(health_score, 0.0)

        except Exception as e:
            logger.error(f"システム健全性評価エラー: {e}")
            return 0.5  # エラー時は中間値

    async def _generate_alerts(self, summary) -> List[str]:
        """アラートを生成"""
        alerts = []

        try:
            # 在庫切れアラート
            if summary.out_of_stock_slots > 0:
                alerts.append(f"在庫切れ商品が{summary.out_of_stock_slots}件あります")

            # 在庫不足アラート
            low_stock_count = sum(
                1
                for p in SAMPLE_PRODUCTS
                if getattr(p, "stock_quantity", 0) <= self.restock_threshold
            )
            if low_stock_count > 0:
                alerts.append(f"在庫不足商品が{low_stock_count}件あります")

            # 運営モードアラート
            if self.current_mode != OperationMode.NORMAL:
                alerts.append(f"運営モード: {self.current_mode.value}")

            return alerts

        except Exception as e:
            logger.error(f"アラート生成エラー: {e}")
            return [f"アラート生成エラー: {str(e)}"]

    async def make_strategic_decision(self, context: Dict[str, Any]) -> VendingDecision:
        """
        戦略的な意思決定を行う

        Args:
            context: 決定コンテキスト

        Returns:
            決定結果
        """
        try:
            logger.info("戦略的意思決定を実行中...")

            # AIプロンプトの構築
            prompt = self._build_decision_prompt(context)

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは自動販売機の運営責任者です。データに基づいて最適な戦略的意思決定を行います。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=400
            )

            if response.success:
                # AI応答から決定を抽出
                decision = self._parse_ai_decision(response.content, context)
            else:
                # AI決定失敗時はデフォルト決定
                decision = VendingDecision(
                    action_type="monitoring",
                    product_id=None,
                    decision="通常運営を継続",
                    reasoning=f"AI決定システムが利用できないため、手動モードで運営を継続します: {response.error_message}",
                    confidence=0.5,
                    priority=5,
                    timestamp=datetime.now(),
                )

            # 決定履歴を記録
            self.operation_history.append(decision)

            # 履歴数の制限
            if len(self.operation_history) > 100:
                self.operation_history = self.operation_history[-100:]

            logger.info(f"戦略的意思決定完了: {decision.action_type}")

            return decision

        except Exception as e:
            logger.error(f"戦略的意思決定エラー: {e}")
            return VendingDecision(
                action_type="error",
                product_id=None,
                decision="エラーによる決定中断",
                reasoning=f"決定プロセスでエラーが発生しました: {str(e)}",
                confidence=0.0,
                priority=1,
                timestamp=datetime.now(),
            )

    def _build_decision_prompt(self, context: Dict[str, Any]) -> str:
        """決定プロンプトを構築"""
        return f"""
        現在の運営状況:
        - 運営モード: {self.current_mode.value}
        - 在庫状況: {context.get("inventory_status", "不明")}
        - 売上状況: {context.get("sales_status", "不明")}
        - システム健全性: {context.get("system_health", "不明")}

        利用可能な情報:
        - 商品情報: {len(SAMPLE_PRODUCTS)}商品
        - サービス状態: 在庫・決済・検索サービス稼働中
        - 時間帯: {datetime.now().strftime("%H:%M")}
        - 曜日: {datetime.now().strftime("%A")}

        上記の状況を分析し、最適な運営戦略を決定してください。

        以下のJSON形式で応答してください：
        {{
            "action_type": "決定の種類（例: price_adjustment, promotion, restock, maintenance）",
            "product_id": "対象商品ID（該当する場合）",
            "decision": "具体的な決定内容",
            "reasoning": "決定の理由",
            "confidence": "信頼度（0-1）",
            "priority": "優先度（1-10）"
        }}

        決定のポイント：
        1. 現在の運営効率を最大化
        2. 顧客満足度を維持・向上
        3. 在庫リスクを最小化
        4. 利益を最適化
        """

    def _parse_ai_decision(
        self, ai_response: str, context: Dict[str, Any]
    ) -> VendingDecision:
        """AI決定をパース"""
        try:
            # 簡易的なパース（実際の実装ではより堅牢なJSONパースを行う）
            # ここではデフォルトの決定を返す
            return VendingDecision(
                action_type="monitoring",
                product_id=None,
                decision="通常運営を継続",
                reasoning="AI応答を正常に処理しました",
                confidence=0.8,
                priority=5,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"AI決定パースエラー: {e}")
            return VendingDecision(
                action_type="monitoring",
                product_id=None,
                decision="通常運営を継続（AI決定パースエラー）",
                reasoning=f"決定パースエラー: {str(e)}",
                confidence=0.3,
                priority=3,
                timestamp=datetime.now(),
            )

    async def run_maintenance_check(self) -> Dict[str, Any]:
        """メンテナンスチェックを実行"""
        try:
            logger.info("メンテナンスチェックを実行中...")

            maintenance_report = {
                "timestamp": datetime.now(),
                "checks_performed": [],
                "issues_found": [],
                "recommendations": [],
                "overall_status": "healthy",
            }

            # 在庫システムチェック
            try:
                summary = self.inventory_service.get_inventory_summary()
                maintenance_report["checks_performed"].append("inventory_system")

                if summary.out_of_stock_slots > 0:
                    maintenance_report["issues_found"].append(
                        {
                            "type": "inventory",
                            "severity": "high",
                            "description": f"在庫切れ商品が{summary.out_of_stock_slots}件あります",
                        }
                    )

            except Exception as e:
                maintenance_report["issues_found"].append(
                    {
                        "type": "inventory_system_error",
                        "severity": "critical",
                        "description": f"在庫システムエラー: {str(e)}",
                    }
                )

            # システム全体のステータス判定
            if len(maintenance_report["issues_found"]) == 0:
                maintenance_report["overall_status"] = "healthy"
            elif any(
                issue["severity"] == "critical"
                for issue in maintenance_report["issues_found"]
            ):
                maintenance_report["overall_status"] = "critical"
            else:
                maintenance_report["overall_status"] = "warning"

            logger.info(
                f"メンテナンスチェック完了: {maintenance_report['overall_status']}"
            )

            return {"success": True, "maintenance_report": maintenance_report}

        except Exception as e:
            logger.error(f"メンテナンスチェックエラー: {e}")
            return {"success": False, "error": str(e), "maintenance_report": {}}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """性能指標を取得"""
        return {
            "operation_mode": self.current_mode.value,
            "decisions_made": len(self.operation_history),
            "last_decision": self.operation_history[-1]
            if self.operation_history
            else None,
            "system_health": 0.8,  # 簡易実装
            "uptime": "99.5%",  # 簡易実装
            "error_rate": "0.1%",  # 簡易実装
        }


# グローバルインスタンス
vending_agent = VendingAgent()
