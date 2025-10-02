"""
調達エージェント

このエージェントは自動販売機の調達・在庫管理を担当し、
最適な仕入れ先選定と発注処理を行います。
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agents.search_agent import search_agent
from src.ai.model_manager import AIMessage, AIResponse, ModelManager, model_manager
from src.config.settings import settings
from src.models.product import SAMPLE_PRODUCTS, Product
from src.services.inventory_service import inventory_service

logger = logging.getLogger(__name__)


class ProcurementStatus(Enum):
    """調達ステータス"""

    IDLE = "idle"  # 待機中
    MONITORING = "monitoring"  # 監視中
    PRICE_CHECKING = "price_checking"  # 価格確認中
    ORDER_PENDING = "order_pending"  # 発注待機中
    ORDER_PLACED = "order_placed"  # 発注済み
    DELIVERY_WAITING = "delivery_waiting"  # 配送待ち
    COMPLETED = "completed"  # 完了
    CANCELLED = "cancelled"  # キャンセル


@dataclass
class ProcurementOrder:
    """発注情報"""

    order_id: str
    product_id: str
    product_name: str
    supplier_name: str
    supplier_url: str
    quantity: int
    unit_price: float
    total_amount: float
    order_date: datetime
    expected_delivery: datetime
    status: ProcurementStatus
    priority: str  # "high", "medium", "low"
    notes: str = ""


@dataclass
class InventoryAlert:
    """在庫アラート"""

    product_id: str
    product_name: str
    current_stock: int
    threshold: int
    alert_type: str  # "low_stock", "out_of_stock", "overstock"
    severity: str  # "critical", "high", "medium", "low"
    message: str
    timestamp: datetime


class ProcurementAgent:
    """
    調達エージェント

    自動販売機の調達・在庫管理を担当し、以下の役割を果たします：
    - 在庫状況の監視とアラート生成
    - 最適な仕入れ先の選定と価格比較
    - 発注指示書の自動生成と管理
    - 在庫補充タイミングの最適化
    - コスト効率の分析と改善提案
    """

    def __init__(self):
        self.model_manager = model_manager
        self.inventory_service = inventory_service
        self.search_agent = search_agent

        # 調達設定
        self.restock_threshold = 5  # 再発注閾値
        self.critical_threshold = 2  # 緊急発注閾値
        self.max_stock_level = 50  # 最大在庫数
        self.price_comparison_count = 5  # 価格比較対象数

        # 発注管理
        self.active_orders: Dict[str, ProcurementOrder] = {}
        self.order_history: List[ProcurementOrder] = []
        self.inventory_alerts: List[InventoryAlert] = []

        # 監視設定
        self.monitoring_interval = 300  # 監視間隔（秒）
        self.is_monitoring = False

    async def initialize(self) -> bool:
        """エージェントの初期化"""
        try:
            logger.info("調達エージェントを初期化中...")

            # システム状態チェック
            system_ready = await self._check_system_readiness()
            if not system_ready:
                logger.error("システム準備が完了していません")
                return False

            # 初期在庫調査
            await self._initialize_inventory_monitoring()

            # 初期発注チェック
            await self._check_initial_orders()

            logger.info("調達エージェントの初期化完了")
            return True

        except Exception as e:
            logger.error(f"エージェント初期化エラー: {e}")
            return False

    async def _check_system_readiness(self) -> bool:
        """システム準備状態の確認"""
        try:
            services_status = []

            # 在庫サービス
            try:
                summary = self.inventory_service.get_inventory_summary()
                services_status.append(True)
            except Exception as e:
                logger.error(f"在庫サービス確認エラー: {e}")
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

    async def _initialize_inventory_monitoring(self):
        """在庫監視の初期化"""
        try:
            logger.info("在庫監視を初期化中...")

            # 現在の在庫状況をチェック
            summary = self.inventory_service.get_inventory_summary()

            # 商品ごとの在庫レベルをチェック
            for product in SAMPLE_PRODUCTS:
                current_stock = getattr(product, "stock_quantity", 0)

                if current_stock <= self.critical_threshold:
                    alert = InventoryAlert(
                        product_id=product.product_id,
                        product_name=product.name,
                        current_stock=current_stock,
                        threshold=self.critical_threshold,
                        alert_type="out_of_stock"
                        if current_stock == 0
                        else "low_stock",
                        severity="critical" if current_stock == 0 else "high",
                        message=f"{product.name}の在庫が{'切れ' if current_stock == 0 else '不足'}しています",
                        timestamp=datetime.now(),
                    )
                    self.inventory_alerts.append(alert)

            logger.info(f"在庫監視初期化完了: {len(self.inventory_alerts)}件のアラート")

        except Exception as e:
            logger.error(f"在庫監視初期化エラー: {e}")

    async def _check_initial_orders(self):
        """初期発注チェック"""
        try:
            # 緊急在庫不足商品の確認
            emergency_products = []

            for product in SAMPLE_PRODUCTS:
                current_stock = getattr(product, "stock_quantity", 0)
                if current_stock <= self.critical_threshold:
                    emergency_products.append(product)

            if emergency_products:
                logger.warning(
                    f"緊急発注が必要な商品が{len(emergency_products)}件あります"
                )
                # 緊急発注プロセスを開始
                await self._process_emergency_orders(emergency_products)
            else:
                logger.info("初期発注チェック完了: 緊急発注は不要です")

        except Exception as e:
            logger.error(f"初期発注チェックエラー: {e}")

    async def start_monitoring(self):
        """在庫監視を開始"""
        if self.is_monitoring:
            logger.warning("監視は既に開始されています")
            return

        self.is_monitoring = True
        logger.info("在庫監視を開始します")

        while self.is_monitoring:
            try:
                # 在庫状況のチェック
                await self._monitor_inventory_levels()

                # 発注状況の更新
                await self._update_order_status()

                # アラートの処理
                await self._process_alerts()

                # 指定間隔待機
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                logger.info("在庫監視を停止します")
                break
            except Exception as e:
                logger.error(f"在庫監視エラー: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def stop_monitoring(self):
        """在庫監視を停止"""
        self.is_monitoring = False
        logger.info("在庫監視を停止しました")

    async def _monitor_inventory_levels(self):
        """在庫レベルを監視"""
        try:
            logger.debug("在庫レベル監視を実行中...")

            # 在庫状況の取得
            summary = self.inventory_service.get_inventory_summary()

            # 商品ごとの詳細チェック
            for product in SAMPLE_PRODUCTS:
                current_stock = getattr(product, "stock_quantity", 0)

                # アラートチェック
                if current_stock == 0:
                    await self._create_stock_alert(product, "out_of_stock", "critical")
                elif current_stock <= self.critical_threshold:
                    await self._create_stock_alert(product, "low_stock", "high")
                elif current_stock <= self.restock_threshold:
                    await self._create_stock_alert(product, "low_stock", "medium")

            logger.debug("在庫レベル監視完了")

        except Exception as e:
            logger.error(f"在庫レベル監視エラー: {e}")

    async def _create_stock_alert(
        self, product: Product, alert_type: str, severity: str
    ):
        """在庫アラートを作成"""
        try:
            # 重複アラートをチェック
            existing_alert = next(
                (
                    alert
                    for alert in self.inventory_alerts
                    if alert.product_id == product.product_id
                    and alert.alert_type == alert_type
                ),
                None,
            )

            if existing_alert:
                return  # 既に同じアラートが存在する場合はスキップ

            alert = InventoryAlert(
                product_id=product.product_id,
                product_name=product.name,
                current_stock=getattr(product, "stock_quantity", 0),
                threshold=self.critical_threshold
                if "critical" in severity
                else self.restock_threshold,
                alert_type=alert_type,
                severity=severity,
                message=f"{product.name}の在庫が{'切れ' if alert_type == 'out_of_stock' else '不足'}しています",
                timestamp=datetime.now(),
            )

            self.inventory_alerts.append(alert)
            logger.warning(f"在庫アラート作成: {alert.message}")

        except Exception as e:
            logger.error(f"在庫アラート作成エラー: {e}")

    async def _update_order_status(self):
        """発注状況を更新"""
        try:
            # 発注済み注文のステータス更新（簡易実装）
            for order_id, order in self.active_orders.items():
                if order.status == ProcurementStatus.ORDER_PLACED:
                    # 発注から3日経過したら配送待ち状態に変更（簡易ロジック）
                    if datetime.now() - order.order_date > timedelta(days=3):
                        order.status = ProcurementStatus.DELIVERY_WAITING
                        logger.info(f"発注{order_id}を配送待ち状態に更新")

        except Exception as e:
            logger.error(f"発注状況更新エラー: {e}")

    async def _process_alerts(self):
        """アラートを処理"""
        try:
            # 緊急アラートの処理
            critical_alerts = [
                alert
                for alert in self.inventory_alerts
                if alert.severity == "critical"
                and alert.alert_type in ["out_of_stock", "low_stock"]
            ]

            for alert in critical_alerts:
                # 緊急発注プロセスを開始
                await self._process_emergency_procurement(alert)

        except Exception as e:
            logger.error(f"アラート処理エラー: {e}")

    async def _process_emergency_procurement(self, alert: InventoryAlert):
        """緊急調達を処理"""
        try:
            logger.info(f"緊急調達プロセス開始: {alert.product_name}")

            # 価格比較と最適仕入れ先選定
            price_comparison = await self.search_agent.compare_prices(
                alert.product_name
            )

            if not price_comparison.search_results:
                logger.error(f"価格情報が見つかりません: {alert.product_name}")
                return

            # 最適な仕入れ先を選択
            best_supplier = await self._select_best_supplier(
                price_comparison, emergency=True
            )

            if not best_supplier:
                logger.error(f"適切な仕入れ先が見つかりません: {alert.product_name}")
                return

            # 発注数量の決定（緊急時は多めに発注）
            emergency_quantity = 30  # 緊急時の発注量

            # 発注を作成
            order = ProcurementOrder(
                order_id=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert.product_id}",
                product_id=alert.product_id,
                product_name=alert.product_name,
                supplier_name=best_supplier.get("name", "不明"),
                supplier_url=best_supplier.get("url", ""),
                quantity=emergency_quantity,
                unit_price=best_supplier.get("price", 0),
                total_amount=best_supplier.get("price", 0) * emergency_quantity,
                order_date=datetime.now(),
                expected_delivery=datetime.now()
                + timedelta(days=1),  # 緊急時は翌日配送想定
                status=ProcurementStatus.ORDER_PENDING,
                priority="high",
                notes=f"緊急発注: {alert.message}",
            )

            # 発注を登録
            self.active_orders[order.order_id] = order

            # AIで発注優先度を分析
            await self._analyze_order_priority(order)

            logger.info(f"緊急発注作成完了: {order.order_id}")

        except Exception as e:
            logger.error(f"緊急調達処理エラー: {e}")

    async def monitor_inventory_and_procure(self) -> Dict[str, Any]:
        """
        在庫監視と自動調達を実行

        Returns:
            調達結果
        """
        try:
            logger.info("在庫監視と自動調達を実行中...")

            procurement_results = {
                "inventory_status": {},
                "procurement_actions": [],
                "cost_analysis": {},
                "recommendations": [],
            }

            # 1. 在庫状況の確認
            inventory_status = await self._analyze_inventory_status()
            procurement_results["inventory_status"] = inventory_status

            # 2. 調達が必要な商品の特定
            products_needing_procurement = await self._identify_procurement_needs(
                inventory_status
            )

            if not products_needing_procurement:
                return {
                    "success": True,
                    "message": "調達が必要な商品はありません",
                    "procurement_results": procurement_results,
                }

            # 3. 商品ごとの調達計画作成
            total_estimated_cost = 0

            for product_info in products_needing_procurement:
                try:
                    # 価格比較を実行
                    price_comparison = await self.search_agent.compare_prices(
                        product_info["product_name"]
                    )

                    if price_comparison.best_price:
                        # 最適な調達計画を作成
                        procurement_plan = await self._create_procurement_plan(
                            product_info, price_comparison
                        )

                        if procurement_plan:
                            procurement_results["procurement_actions"].append(
                                procurement_plan
                            )
                            total_estimated_cost += procurement_plan["total_cost"]

                except Exception as e:
                    logger.error(
                        f"調達計画作成エラー ({product_info['product_name']}): {e}"
                    )

            # 4. コスト分析
            cost_analysis = await self._analyze_procurement_costs(
                procurement_results["procurement_actions"]
            )
            procurement_results["cost_analysis"] = cost_analysis

            # 5. AIで全体戦略を生成
            if procurement_results["procurement_actions"]:
                strategy = await self._generate_procurement_strategy(
                    procurement_results
                )
                procurement_results["recommendations"] = strategy

            logger.info(
                f"在庫監視と自動調達完了: {len(procurement_results['procurement_actions'])}件の調達計画"
            )

            return {
                "success": True,
                "message": f"{len(procurement_results['procurement_actions'])}件の調達計画を作成しました",
                "procurement_results": procurement_results,
            }

        except Exception as e:
            logger.error(f"在庫監視と自動調達エラー: {e}")
            return {"success": False, "error": str(e), "procurement_results": {}}

    async def _analyze_inventory_status(self) -> Dict[str, Any]:
        """在庫状況を分析"""
        try:
            summary = self.inventory_service.get_inventory_summary()

            inventory_analysis = {
                "total_products": len(SAMPLE_PRODUCTS),
                "total_slots": summary.total_slots,
                "active_slots": summary.active_slots,
                "out_of_stock_slots": summary.out_of_stock_slots,
                "low_stock_products": 0,
                "healthy_products": 0,
                "product_details": [],
            }

            # 商品ごとの詳細分析
            for product in SAMPLE_PRODUCTS:
                current_stock = getattr(product, "stock_quantity", 0)

                product_detail = {
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "current_stock": current_stock,
                    "max_capacity": self.max_stock_level,
                    "stock_level": current_stock / self.max_stock_level
                    if self.max_stock_level > 0
                    else 0,
                    "status": "healthy",
                }

                if current_stock == 0:
                    product_detail["status"] = "out_of_stock"
                    inventory_analysis["out_of_stock_slots"] += 1
                elif current_stock <= self.critical_threshold:
                    product_detail["status"] = "critical"
                    inventory_analysis["low_stock_products"] += 1
                elif current_stock <= self.restock_threshold:
                    product_detail["status"] = "low"
                    inventory_analysis["low_stock_products"] += 1
                else:
                    product_detail["status"] = "healthy"
                    inventory_analysis["healthy_products"] += 1

                inventory_analysis["product_details"].append(product_detail)

            return inventory_analysis

        except Exception as e:
            logger.error(f"在庫状況分析エラー: {e}")
            return {}

    async def _identify_procurement_needs(
        self, inventory_status: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """調達が必要な商品を特定"""
        try:
            needs_procurement = []

            for product_detail in inventory_status.get("product_details", []):
                current_stock = product_detail["current_stock"]

                # 調達が必要な条件
                if current_stock <= self.restock_threshold:
                    # 推奨発注量を計算
                    recommended_quantity = self.max_stock_level - current_stock

                    need_info = {
                        "product_id": product_detail["product_id"],
                        "product_name": product_detail["product_name"],
                        "current_stock": current_stock,
                        "recommended_quantity": max(
                            recommended_quantity, 10
                        ),  # 最低発注量
                        "priority": "high"
                        if current_stock <= self.critical_threshold
                        else "medium",
                        "urgency": "緊急" if current_stock == 0 else "通常",
                    }

                    needs_procurement.append(need_info)

            return needs_procurement

        except Exception as e:
            logger.error(f"調達必要商品特定エラー: {e}")
            return []

    async def _create_procurement_plan(
        self, product_info: Dict[str, Any], price_comparison
    ) -> Optional[Dict[str, Any]]:
        """調達計画を作成"""
        try:
            if not price_comparison.best_price:
                return None

            # 発注情報を構築
            procurement_plan = {
                "product_id": product_info["product_id"],
                "product_name": product_info["product_name"],
                "quantity": product_info["recommended_quantity"],
                "unit_price": price_comparison.best_price,
                "total_cost": price_comparison.best_price
                * product_info["recommended_quantity"],
                "supplier_info": self._extract_supplier_info(
                    price_comparison.search_results[0]
                ),
                "priority": product_info["priority"],
                "urgency": product_info["urgency"],
                "estimated_delivery": datetime.now()
                + timedelta(days=3),  # 標準配送日数
                "cost_effectiveness": await self._calculate_cost_effectiveness(
                    price_comparison
                ),
            }

            return procurement_plan

        except Exception as e:
            logger.error(f"調達計画作成エラー: {e}")
            return None

    def _extract_supplier_info(self, search_result) -> Dict[str, str]:
        """検索結果から仕入れ先情報を抽出"""
        return {
            "name": search_result.title,
            "url": search_result.url,
            "price": str(search_result.price) if search_result.price else "不明",
            "availability": search_result.availability or "不明",
        }

    async def _calculate_cost_effectiveness(self, price_comparison) -> float:
        """コスト効率を計算（0-1の範囲）"""
        try:
            if not price_comparison.best_price or not price_comparison.average_price:
                return 0.5

            # 最安価格と平均価格の比較
            if price_comparison.average_price > 0:
                effectiveness = 1.0 - (
                    price_comparison.best_price / price_comparison.average_price
                )
                return max(0.0, min(1.0, effectiveness))

            return 0.5

        except Exception as e:
            logger.error(f"コスト効率計算エラー: {e}")
            return 0.5

    async def _select_best_supplier(
        self, price_comparison, emergency: bool = False
    ) -> Optional[Dict[str, Any]]:
        """最適な仕入れ先を選択"""
        try:
            if not price_comparison.search_results:
                return None

            # 価格でソート
            sorted_results = sorted(
                price_comparison.search_results,
                key=lambda x: x.price if x.price else float("inf"),
            )

            # 上位候補から選択
            candidates = sorted_results[:3]

            # AIで総合評価
            best_supplier = await self._evaluate_suppliers(candidates, emergency)

            return best_supplier

        except Exception as e:
            logger.error(f"最適仕入れ先選択エラー: {e}")
            return None

    async def _evaluate_suppliers(
        self, candidates, emergency: bool = False
    ) -> Dict[str, Any]:
        """仕入れ先を評価"""
        try:
            # 簡易評価（価格優先）
            best_candidate = min(
                candidates, key=lambda x: x.price if x.price else float("inf")
            )

            return {
                "name": best_candidate.title,
                "url": best_candidate.url,
                "price": best_candidate.price,
                "availability": best_candidate.availability,
                "relevance_score": best_candidate.relevance_score,
            }

        except Exception as e:
            logger.error(f"仕入れ先評価エラー: {e}")
            return {}

    async def _analyze_procurement_costs(
        self, procurement_actions: List[Dict]
    ) -> Dict[str, Any]:
        """調達コストを分析"""
        try:
            if not procurement_actions:
                return {"total_cost": 0, "average_unit_cost": 0, "cost_breakdown": {}}

            total_cost = sum(action["total_cost"] for action in procurement_actions)
            average_unit_cost = total_cost / sum(
                action["quantity"] for action in procurement_actions
            )

            # 商品別コスト内訳
            cost_breakdown = {}
            for action in procurement_actions:
                cost_breakdown[action["product_name"]] = {
                    "quantity": action["quantity"],
                    "unit_cost": action["unit_price"],
                    "total_cost": action["total_cost"],
                }

            return {
                "total_cost": total_cost,
                "average_unit_cost": average_unit_cost,
                "cost_breakdown": cost_breakdown,
                "action_count": len(procurement_actions),
            }

        except Exception as e:
            logger.error(f"調達コスト分析エラー: {e}")
            return {"total_cost": 0, "average_unit_cost": 0, "cost_breakdown": {}}

    async def _generate_procurement_strategy(
        self, procurement_results: Dict[str, Any]
    ) -> List[str]:
        """調達戦略を生成"""
        try:
            # AIプロンプトの構築
            actions_text = "\n".join(
                [
                    f"- {action['product_name']}: {action['quantity']}個 × ¥{action['unit_price']} = ¥{action['total_cost']:,}"
                    for action in procurement_results["procurement_actions"]
                ]
            )

            total_cost = procurement_results["cost_analysis"].get("total_cost", 0)

            prompt = f"""
            調達計画:
            {actions_text}

            総調達コスト: ¥{total_cost:,}

            上記の調達計画に基づいて、戦略的な提案を作成してください。
            以下のポイントを考慮：
            1. コスト効率の最適化
            2. 発注タイミングの調整
            3. 仕入れ先の多様化
            4. 在庫リスクの管理
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは調達戦略の専門家です。コスト効率とリスク管理を考慮した最適な調達戦略を提案します。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=300
            )

            if response.success:
                # 応答をパースして推奨事項を抽出
                recommendations = self._parse_procurement_strategy(response.content)
                return recommendations
            else:
                return [f"調達戦略の生成に失敗しました: {response.error_message}"]

        except Exception as e:
            logger.error(f"調達戦略生成エラー: {e}")
            return ["調達戦略の生成に失敗しました。標準的な手順で進めてください。"]

    def _parse_procurement_strategy(self, ai_response: str) -> List[str]:
        """調達戦略をパース"""
        try:
            recommendations = []

            if "戦略" in ai_response or "提案" in ai_response:
                # 日本語の応答から推奨事項を抽出
                lines = ai_response.split("\n")
                for line in lines:
                    if any(
                        keyword in line
                        for keyword in ["戦略", "提案", "検討", "推奨", "考慮"]
                    ):
                        recommendations.append(line.strip())

            if not recommendations:
                recommendations = ["AI戦略をパースできませんでした"]

            return recommendations

        except Exception as e:
            logger.error(f"調達戦略パースエラー: {e}")
            return ["パースエラーにより標準戦略を適用します"]

    async def create_order_instruction(
        self, product_info: Dict[str, Any], supplier_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        発注指示書を作成

        Args:
            product_info: 商品情報
            supplier_info: 仕入れ先情報

        Returns:
            発注指示書
        """
        try:
            order_instruction = {
                "order_id": f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product_info['product_id']}",
                "product_id": product_info["product_id"],
                "product_name": product_info["product_name"],
                "supplier_name": supplier_info.get("name", "不明"),
                "supplier_url": supplier_info.get("url", ""),
                "quantity": product_info["recommended_quantity"],
                "unit_price": supplier_info.get("price", 0),
                "total_amount": supplier_info.get("price", 0)
                * product_info["recommended_quantity"],
                "order_date": datetime.now(),
                "expected_delivery": datetime.now() + timedelta(days=3),
                "priority": product_info["priority"],
                "special_instructions": self._generate_special_instructions(
                    product_info
                ),
                "status": "draft",
            }

            logger.info(f"発注指示書作成完了: {order_instruction['order_id']}")

            return {"success": True, "order_instruction": order_instruction}

        except Exception as e:
            logger.error(f"発注指示書作成エラー: {e}")
            return {"success": False, "error": str(e), "order_instruction": {}}

    def _generate_special_instructions(self, product_info: Dict[str, Any]) -> str:
        """特別指示を生成"""
        instructions = []

        if product_info["priority"] == "high":
            instructions.append("緊急発注のため、優先的に処理してください")

        if product_info["urgency"] == "緊急":
            instructions.append("至急対応をお願いします")

        instructions.append("品質確認後、速やかに納品してください")

        return " | ".join(instructions)

    async def get_procurement_report(self) -> Dict[str, Any]:
        """調達レポートを取得"""
        try:
            # 現在の在庫状況
            inventory_status = await self._analyze_inventory_status()

            # 発注状況
            order_summary = {
                "active_orders": len(self.active_orders),
                "completed_orders": len(
                    [
                        o
                        for o in self.order_history
                        if o.status == ProcurementStatus.COMPLETED
                    ]
                ),
                "total_cost_this_month": sum(
                    o.total_amount
                    for o in self.order_history
                    if o.order_date.month == datetime.now().month
                ),
            }

            # アラート状況
            alert_summary = {
                "total_alerts": len(self.inventory_alerts),
                "critical_alerts": len(
                    [a for a in self.inventory_alerts if a.severity == "critical"]
                ),
                "high_alerts": len(
                    [a for a in self.inventory_alerts if a.severity == "high"]
                ),
                "recent_alerts": self.inventory_alerts[-5:]
                if self.inventory_alerts
                else [],
            }

            return {
                "timestamp": datetime.now(),
                "inventory_status": inventory_status,
                "order_summary": order_summary,
                "alert_summary": alert_summary,
                "recommendations": await self._generate_procurement_recommendations(
                    inventory_status
                ),
            }

        except Exception as e:
            logger.error(f"調達レポート取得エラー: {e}")
            return {"timestamp": datetime.now(), "error": str(e)}

    async def _generate_procurement_recommendations(
        self, inventory_status: Dict[str, Any]
    ) -> List[str]:
        """調達推奨事項を生成"""
        try:
            recommendations = []

            # 在庫切れ商品の推奨
            out_of_stock = [
                p
                for p in inventory_status.get("product_details", [])
                if p["status"] == "out_of_stock"
            ]
            if out_of_stock:
                recommendations.append(
                    f"在庫切れ商品{len(out_of_stock)}件の緊急調達を優先してください"
                )

            # 在庫不足商品の推奨
            low_stock = [
                p
                for p in inventory_status.get("product_details", [])
                if p["status"] == "low"
            ]
            if low_stock:
                recommendations.append(
                    f"在庫不足商品{len(low_stock)}件の調達計画を作成してください"
                )

            # コスト効率の推奨
            if inventory_status.get("low_stock_products", 0) > 3:
                recommendations.append(
                    "複数の商品をまとめて発注し、配送コストを削減してください"
                )

            return recommendations

        except Exception as e:
            logger.error(f"調達推奨事項生成エラー: {e}")
            return ["推奨事項の生成に失敗しました"]

    def get_inventory_alerts(self) -> List[InventoryAlert]:
        """在庫アラートを取得"""
        return self.inventory_alerts.copy()

    def clear_resolved_alerts(self, product_ids: List[str]):
        """解決済みアラートをクリア"""
        self.inventory_alerts = [
            alert
            for alert in self.inventory_alerts
            if alert.product_id not in product_ids
        ]
        logger.info(f"解決済みアラートをクリア: {len(product_ids)}件")


    async def check_supplier_inventory(self, product_name: str) -> Dict[str, Any]:
        """サプライヤ在庫を確認（簡易版）"""
        try:
            # 簡易的な在庫確認（実際には外部API呼び出し）
            # サンプルデータ
            supplier_stock = {"cola": 50, "water": 30, "juice": 20}.get(product_name, 10)
            return {
                "product_name": product_name,
                "supplier_stock": supplier_stock,
                "estimated_delivery": "1-2営業日",
            }
        except Exception as e:
            logger.error(f"サプライヤ在庫確認エラー: {e}")
            return {"supplier_stock": 0, "error": str(e)}


# グローバルインスタンス
procurement_agent = ProcurementAgent()
