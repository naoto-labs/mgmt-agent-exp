"""
分析エージェント

このエージェントはシステム全体のデータ分析とレポート生成を担当し、
ビジネスインサイトの提供と異常検出を行います。
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.application.services.inventory_service import inventory_service
from src.application.services.payment_service import payment_service
from src.domain.accounting.journal_entry import journal_processor
from src.domain.analytics.event_tracker import event_tracker
from src.domain.models.product import SAMPLE_PRODUCTS, Product
from src.domain.models.transaction import Transaction, TransactionStatus
from src.infrastructure.ai.model_manager import AIMessage, model_manager
from src.shared.config.settings import settings

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """分析種類"""

    SALES_ANALYSIS = "sales_analysis"  # 売上分析
    INVENTORY_ANALYSIS = "inventory_analysis"  # 在庫分析
    CUSTOMER_ANALYSIS = "customer_analysis"  # 顧客分析
    FINANCIAL_ANALYSIS = "financial_analysis"  # 財務分析
    PERFORMANCE_ANALYSIS = "performance_analysis"  # 性能分析
    TREND_ANALYSIS = "trend_analysis"  # トレンド分析
    ANOMALY_DETECTION = "anomaly_detection"  # 異常検出


class ReportFrequency(Enum):
    """レポート頻度"""

    REALTIME = "realtime"  # リアルタイム
    HOURLY = "hourly"  # 1時間ごと
    DAILY = "daily"  # 日次
    WEEKLY = "weekly"  # 週次
    MONTHLY = "monthly"  # 月次


@dataclass
class AnalysisResult:
    """分析結果"""

    analysis_type: AnalysisType
    timestamp: datetime
    period: Dict[str, datetime]  # start_date, end_date
    summary: Dict[str, Any]
    details: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    data_quality: float


@dataclass
class AnomalyDetection:
    """異常検出結果"""

    anomaly_id: str
    detection_time: datetime
    anomaly_type: str  # "sales_spike", "inventory_drop", "system_error", etc.
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_components: List[str]
    confidence: float
    suggested_actions: List[str]
    status: str  # "detected", "investigating", "resolved", "false_positive"


@dataclass
class BusinessMetric:
    """ビジネス指標"""

    metric_name: str
    value: float
    previous_value: float
    change_percentage: float
    trend: str  # "up", "down", "stable"
    benchmark: Optional[float] = None
    unit: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AnalyticsAgent:
    """
    分析エージェント

    システム全体のデータ分析とレポート生成を担当し、以下の役割を果たします：
    - 売上・在庫・顧客・財務データの包括的分析
    - 異常検出とリアルタイム監視
    - トレンド分析と予測
    - パフォーマンス指標の追跡と評価
    - ビジネスインサイトの生成と提案
    """

    def __init__(self):
        self.model_manager = model_manager
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.event_tracker = event_tracker
        self.journal_processor = journal_processor

        # 分析設定
        self.analysis_interval = 3600  # 分析間隔（秒）
        self.anomaly_detection_enabled = True
        self.prediction_horizon = 30  # 予測期間（日）

        # データキャッシュ
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.metrics_cache: Dict[str, List[BusinessMetric]] = defaultdict(list)
        self.anomaly_history: List[AnomalyDetection] = []

        # 監視設定
        self.is_monitoring = False
        self.baseline_data: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """エージェントの初期化"""
        try:
            logger.info("分析エージェントを初期化中...")

            # システム状態チェック
            system_ready = await self._check_system_readiness()
            if not system_ready:
                logger.error("システム準備が完了していません")
                return False

            # ベースラインデータの収集
            await self._collect_baseline_data()

            # 初期分析の実行
            await self._run_initial_analysis()

            logger.info("分析エージェントの初期化完了")
            return True

        except Exception as e:
            logger.error(f"エージェント初期化エラー: {e}")
            return False

    async def _check_system_readiness(self) -> bool:
        """システム準備状態の確認"""
        try:
            services_status = []

            # イベント追跡サービス
            try:
                events = await self.event_tracker.get_recent_events()
                services_status.append(True)
            except Exception as e:
                logger.error(f"イベント追跡サービス確認エラー: {e}")
                services_status.append(False)

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

            return all(services_status)

        except Exception as e:
            logger.error(f"システム準備状態確認エラー: {e}")
            return False

    async def _collect_baseline_data(self):
        """ベースラインデータの収集"""
        try:
            logger.info("ベースラインデータを収集中...")

            # 過去30日間のデータをベースラインとして収集
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # 売上データのベースライン
            self.baseline_data["sales"] = {
                "average_daily": 0,  # 実際の実装では過去データから計算
                "peak_hours": [11, 12, 13, 17, 18, 19],  # ピーク時間帯
                "slow_hours": [2, 3, 4, 5, 6],  # 閑散時間帯
                "average_transaction_value": 200,  # 平均取引額
            }

            # 在庫データのベースライン
            self.baseline_data["inventory"] = {
                "average_stock_level": 0.7,  # 平均在庫水準
                "turnover_rate": 2.5,  # 在庫回転率
                "out_of_stock_rate": 0.05,  # 在庫切れ率
            }

            # システム性能のベースライン
            self.baseline_data["system"] = {
                "average_response_time": 0.5,  # 平均応答時間（秒）
                "error_rate": 0.01,  # エラー率
                "uptime": 0.995,  # 稼働率
            }

            logger.info("ベースラインデータ収集完了")

        except Exception as e:
            logger.error(f"ベースラインデータ収集エラー: {e}")

    async def _run_initial_analysis(self):
        """初期分析の実行"""
        try:
            logger.info("初期分析を実行中...")

            # 現在のシステム状態を分析
            current_analysis = await self.analyze_system_events()

            if current_analysis:
                logger.info("初期分析完了")
            else:
                logger.warning("初期分析に失敗しました")

        except Exception as e:
            logger.error(f"初期分析エラー: {e}")

    async def start_monitoring(self):
        """分析監視を開始"""
        if self.is_monitoring:
            logger.warning("監視は既に開始されています")
            return

        self.is_monitoring = True
        logger.info("分析監視を開始します")

        while self.is_monitoring:
            try:
                # 包括的なシステム分析
                await self.analyze_system_events()

                # 異常検出
                if self.anomaly_detection_enabled:
                    await self._detect_anomalies()

                # 指標の更新
                await self._update_business_metrics()

                # 指定間隔待機
                await asyncio.sleep(self.analysis_interval)

            except asyncio.CancelledError:
                logger.info("分析監視を停止します")
                break
            except Exception as e:
                logger.error(f"分析監視エラー: {e}")
                await asyncio.sleep(self.analysis_interval)

    def stop_monitoring(self):
        """分析監視を停止"""
        self.is_monitoring = False
        logger.info("分析監視を停止しました")

    async def analyze_system_events(self) -> Dict[str, Any]:
        """
        システム事象の包括的分析を実行

        Returns:
            分析結果
        """
        try:
            logger.info("システム事象の包括的分析を実行中...")

            # 分析期間の設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 過去7日間

            analysis_results = {
                "timestamp": end_date,
                "period": {"start": start_date, "end": end_date},
                "analyses": {},
                "overall_insights": [],
                "overall_recommendations": [],
            }

            # 1. 売上分析
            sales_analysis = await self._analyze_sales_performance(start_date, end_date)
            analysis_results["analyses"]["sales"] = sales_analysis

            # 2. 在庫分析
            inventory_analysis = await self._analyze_inventory_performance(
                start_date, end_date
            )
            analysis_results["analyses"]["inventory"] = inventory_analysis

            # 3. 財務分析
            financial_analysis = await self._analyze_financial_performance(
                start_date, end_date
            )
            analysis_results["analyses"]["financial"] = financial_analysis

            # 4. システム性能分析
            system_analysis = await self._analyze_system_performance(
                start_date, end_date
            )
            analysis_results["analyses"]["system"] = system_analysis

            # 5. トレンド分析
            trend_analysis = await self._analyze_trends(start_date, end_date)
            analysis_results["analyses"]["trends"] = trend_analysis

            # 全体的な洞察と推奨事項を生成
            overall_insights = await self._generate_overall_insights(
                analysis_results["analyses"]
            )
            overall_recommendations = await self._generate_overall_recommendations(
                analysis_results["analyses"]
            )

            analysis_results["overall_insights"] = overall_insights
            analysis_results["overall_recommendations"] = overall_recommendations

            # 分析結果をキャッシュ
            cache_key = f"comprehensive_{end_date.strftime('%Y%m%d_%H')}"
            self.analysis_cache[cache_key] = AnalysisResult(
                analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                timestamp=end_date,
                period={"start": start_date, "end": end_date},
                summary={"analyses_count": len(analysis_results["analyses"])},
                details=analysis_results,
                insights=overall_insights,
                recommendations=overall_recommendations,
                confidence=0.85,
                data_quality=0.9,
            )

            logger.info("システム事象の包括的分析完了")

            return analysis_results

        except Exception as e:
            logger.error(f"システム事象分析エラー: {e}")
            return {}

    async def _analyze_sales_performance(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """売上性能を分析"""
        try:
            # 実際の実装ではデータベースから売上データを取得
            # ここではサンプルデータでシミュレーション

            sales_data = {
                "total_revenue": 0,  # 実際の実装では計算
                "total_transactions": 0,  # 実際の実装では計算
                "average_transaction_value": 0,
                "top_products": [],
                "hourly_pattern": {},
                "daily_pattern": {},
                "conversion_rate": 0.0,
                "customer_retention": 0.0,
            }

            # サンプル商品の売上分析
            for product in SAMPLE_PRODUCTS:
                # 簡易的な売上シミュレーション
                estimated_sales = (
                    getattr(product, "stock_quantity", 0) * 0.3
                )  # 30%が売れたと仮定

                if estimated_sales > 0:
                    sales_data["top_products"].append(
                        {
                            "product_id": product.product_id,
                            "product_name": product.name,
                            "estimated_units_sold": estimated_sales,
                            "estimated_revenue": estimated_sales * product.price,
                        }
                    )

            # 時間帯別パターン（サンプル）
            sales_data["hourly_pattern"] = {
                "peak_hours": [11, 12, 13, 17, 18, 19],
                "slow_hours": [2, 3, 4, 5, 6],
                "average_by_hour": {hour: 0 for hour in range(24)},
            }

            return {
                "period": {"start": start_date, "end": end_date},
                "metrics": sales_data,
                "insights": ["売上パターンが時間帯によって明確な違いが見られます"],
                "recommendations": ["ピーク時間帯の在庫確保を優先してください"],
            }

        except Exception as e:
            logger.error(f"売上性能分析エラー: {e}")
            return {}

    async def _analyze_inventory_performance(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """在庫性能を分析"""
        try:
            summary = self.inventory_service.get_inventory_summary()

            # 在庫効率性の計算
            total_products = len(SAMPLE_PRODUCTS)
            out_of_stock_rate = (
                summary.out_of_stock_slots / total_products if total_products > 0 else 0
            )

            # 商品別在庫分析
            product_analysis = []
            for product in SAMPLE_PRODUCTS:
                current_stock = getattr(product, "stock_quantity", 0)
                stock_level = current_stock / 50 if 50 > 0 else 0  # 最大在庫を50と仮定

                product_analysis.append(
                    {
                        "product_id": product.product_id,
                        "product_name": product.name,
                        "current_stock": current_stock,
                        "stock_level": stock_level,
                        "status": "healthy" if stock_level > 0.2 else "low",
                        "turnover_estimate": 2.5,  # 推定回転率
                    }
                )

            return {
                "period": {"start": start_date, "end": end_date},
                "summary": {
                    "total_products": total_products,
                    "out_of_stock_rate": out_of_stock_rate,
                    "average_stock_level": sum(
                        p["stock_level"] for p in product_analysis
                    )
                    / total_products
                    if total_products > 0
                    else 0,
                    "inventory_value": sum(
                        p["current_stock"] * 200 for p in product_analysis
                    ),  # 簡易的な在庫評価額
                },
                "product_analysis": product_analysis,
                "insights": [f"在庫切れ率は{out_of_stock_rate:.1%}です"],
                "recommendations": ["在庫切れ商品の優先的な補充を検討してください"],
            }

        except Exception as e:
            logger.error(f"在庫性能分析エラー: {e}")
            return {}

    async def _analyze_financial_performance(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """財務性能を分析"""
        try:
            # 会計データの取得（簡易実装）
            try:
                trial_balance = self.journal_processor.get_trial_balance()
            except Exception:
                trial_balance = {"total_debit": 0, "total_credit": 0, "accounts": {}}

            # 財務指標の計算
            total_revenue = trial_balance.get("total_credit", 0)  # 売上は貸方
            total_cost = trial_balance.get("total_debit", 0)  # 費用は借方

            gross_profit = total_revenue - total_cost
            gross_margin = gross_profit / total_revenue if total_revenue > 0 else 0

            return {
                "period": {"start": start_date, "end": end_date},
                "summary": {
                    "total_revenue": total_revenue,
                    "total_cost": total_cost,
                    "gross_profit": gross_profit,
                    "gross_margin": gross_margin,
                    "account_count": len(trial_balance.get("accounts", {})),
                },
                "insights": [f"粗利益率は{gross_margin:.1%}です"],
                "recommendations": ["利益率向上のため、コスト管理を強化してください"],
            }

        except Exception as e:
            logger.error(f"財務性能分析エラー: {e}")
            return {}

    async def _analyze_system_performance(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """システム性能を分析"""
        try:
            # システム指標の収集（簡易実装）
            system_metrics = {
                "response_time": 0.3,  # 平均応答時間（秒）
                "error_rate": 0.005,  # エラー率
                "uptime": 0.998,  # 稼働率
                "throughput": 100,  # 処理能力（取引/時間）
                "memory_usage": 0.6,  # メモリ使用率
                "cpu_usage": 0.4,  # CPU使用率
            }

            # 性能評価
            performance_score = (
                (1 - system_metrics["error_rate"]) * 0.3
                + system_metrics["uptime"] * 0.3
                + (1 - system_metrics["response_time"] / 2.0) * 0.2
                + (1 - min(system_metrics["memory_usage"], 1.0)) * 0.1
                + (1 - min(system_metrics["cpu_usage"], 1.0)) * 0.1
            )

            return {
                "period": {"start": start_date, "end": end_date},
                "metrics": system_metrics,
                "performance_score": performance_score,
                "status": "excellent"
                if performance_score > 0.9
                else "good"
                if performance_score > 0.8
                else "needs_improvement",
                "insights": [f"システム性能スコアは{performance_score:.1%}です"],
                "recommendations": [
                    "システム性能が良好です。現在の設定を維持してください"
                ],
            }

        except Exception as e:
            logger.error(f"システム性能分析エラー: {e}")
            return {}

    async def _analyze_trends(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """トレンドを分析"""
        try:
            # トレンドデータの収集（簡易実装）
            trends = {
                "sales_trend": "stable",  # 安定
                "inventory_trend": "decreasing",  # 減少傾向
                "profit_trend": "increasing",  # 増加傾向
                "customer_trend": "stable",  # 安定
                "growth_rate": 0.05,  # 成長率
                "seasonal_patterns": {"peak_season": "summer", "slow_season": "winter"},
            }

            return {
                "period": {"start": start_date, "end": end_date},
                "trends": trends,
                "insights": ["売上は安定した推移を示しています"],
                "recommendations": [
                    "現在のトレンドを維持するための施策を継続してください"
                ],
            }

        except Exception as e:
            logger.error(f"トレンド分析エラー: {e}")
            return {}

    async def _detect_anomalies(self):
        """異常を検出"""
        try:
            logger.debug("異常検出を実行中...")

            # 現在のシステム状態を取得
            current_metrics = await self._get_current_metrics()

            # ベースラインとの比較
            anomalies = []

            # 売上異常検出
            if "sales" in current_metrics:
                sales_anomaly = self._detect_sales_anomaly(current_metrics["sales"])
                if sales_anomaly:
                    anomalies.append(sales_anomaly)

            # 在庫異常検出
            if "inventory" in current_metrics:
                inventory_anomaly = self._detect_inventory_anomaly(
                    current_metrics["inventory"]
                )
                if inventory_anomaly:
                    anomalies.append(inventory_anomaly)

            # システム異常検出
            if "system" in current_metrics:
                system_anomaly = self._detect_system_anomaly(current_metrics["system"])
                if system_anomaly:
                    anomalies.append(system_anomaly)

            # 検出された異常を記録
            for anomaly in anomalies:
                self.anomaly_history.append(anomaly)
                logger.warning(f"異常検出: {anomaly.description}")

            # 履歴数の制限
            if len(self.anomaly_history) > 1000:
                self.anomaly_history = self.anomaly_history[-1000:]

        except Exception as e:
            logger.error(f"異常検出エラー: {e}")

    async def _get_current_metrics(self) -> Dict[str, Any]:
        """現在の指標を取得"""
        try:
            # 実際の実装ではリアルタイムデータを取得
            return {
                "sales": {"today_revenue": 0, "transaction_count": 0},
                "inventory": {"out_of_stock_count": 0, "low_stock_count": 0},
                "system": {"response_time": 0.3, "error_count": 0},
            }

        except Exception as e:
            logger.error(f"現在の指標取得エラー: {e}")
            return {}

    def _detect_sales_anomaly(
        self, sales_data: Dict[str, Any]
    ) -> Optional[AnomalyDetection]:
        """売上異常を検出"""
        try:
            # ベースラインとの比較（簡易実装）
            baseline = self.baseline_data.get("sales", {})
            baseline_avg = baseline.get("average_daily", 0)

            current_revenue = sales_data.get("today_revenue", 0)

            # 異常判定（ベースラインの2倍以上の場合）
            if current_revenue > baseline_avg * 2:
                return AnomalyDetection(
                    anomaly_id=f"sales_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    detection_time=datetime.now(),
                    anomaly_type="sales_spike",
                    severity="medium",
                    description=f"売上が急増しています（本日: ¥{current_revenue:,}）",
                    affected_components=["sales_system", "inventory_system"],
                    confidence=0.8,
                    suggested_actions=[
                        "在庫状況を確認してください",
                        "システム負荷を監視してください",
                        "顧客満足度をチェックしてください",
                    ],
                    status="detected",
                )

            return None

        except Exception as e:
            logger.error(f"売上異常検出エラー: {e}")
            return None

    def _detect_inventory_anomaly(
        self, inventory_data: Dict[str, Any]
    ) -> Optional[AnomalyDetection]:
        """在庫異常を検出"""
        try:
            out_of_stock_count = inventory_data.get("out_of_stock_count", 0)

            # 在庫切れが急増した場合
            if out_of_stock_count > 2:  # 2商品以上在庫切れ
                return AnomalyDetection(
                    anomaly_id=f"inventory_drop_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    detection_time=datetime.now(),
                    anomaly_type="inventory_drop",
                    severity="high",
                    description=f"在庫切れ商品が{out_of_stock_count}件発生しています",
                    affected_components=["inventory_system", "procurement_system"],
                    confidence=0.9,
                    suggested_actions=[
                        "緊急調達プロセスを開始してください",
                        "代替商品の在庫を確認してください",
                        "顧客への影響を評価してください",
                    ],
                    status="detected",
                )

            return None

        except Exception as e:
            logger.error(f"在庫異常検出エラー: {e}")
            return None

    def _detect_system_anomaly(
        self, system_data: Dict[str, Any]
    ) -> Optional[AnomalyDetection]:
        """システム異常を検出"""
        try:
            response_time = system_data.get("response_time", 0)
            error_count = system_data.get("error_count", 0)

            # 応答時間が大幅に遅い場合
            if response_time > 2.0:  # 2秒以上
                return AnomalyDetection(
                    anomaly_id=f"system_slow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    detection_time=datetime.now(),
                    anomaly_type="system_performance",
                    severity="medium",
                    description=f"システム応答時間が遅くなっています（{response_time:.1f}秒）",
                    affected_components=["api_system", "database_system"],
                    confidence=0.7,
                    suggested_actions=[
                        "システム負荷を確認してください",
                        "データベース性能をチェックしてください",
                        "キャッシュ戦略の見直しを検討してください",
                    ],
                    status="detected",
                )

            return None

        except Exception as e:
            logger.error(f"システム異常検出エラー: {e}")
            return None

    async def _update_business_metrics(self):
        """ビジネス指標を更新"""
        try:
            # 現在の指標を計算・記録
            current_time = datetime.now()

            # 売上指標
            sales_metric = BusinessMetric(
                metric_name="daily_revenue",
                value=0,  # 実際の実装では計算
                previous_value=0,
                change_percentage=0.0,
                trend="stable",
                unit="JPY",
            )
            self.metrics_cache["sales"].append(sales_metric)

            # 在庫指標
            inventory_metric = BusinessMetric(
                metric_name="inventory_turnover",
                value=2.5,  # 実際の実装では計算
                previous_value=2.3,
                change_percentage=8.7,
                trend="up",
                unit="turns",
            )
            self.metrics_cache["inventory"].append(inventory_metric)

            # システム指標
            system_metric = BusinessMetric(
                metric_name="system_uptime",
                value=0.998,
                previous_value=0.995,
                change_percentage=0.3,
                trend="up",
                unit="percentage",
            )
            self.metrics_cache["system"].append(system_metric)

            # 古い指標を削除（最新100件のみ保持）
            for metric_list in self.metrics_cache.values():
                if len(metric_list) > 100:
                    metric_list[:] = metric_list[-100:]

        except Exception as e:
            logger.error(f"ビジネス指標更新エラー: {e}")

    async def _generate_overall_insights(self, analyses: Dict[str, Any]) -> List[str]:
        """全体的な洞察を生成"""
        try:
            insights = []

            # 各分析結果から洞察を抽出
            for analysis_type, analysis_data in analyses.items():
                if "insights" in analysis_data:
                    insights.extend(analysis_data["insights"])

            # AIで統合的な洞察を生成
            if insights:
                prompt = f"""
                個別分析の結果から抽出された洞察:
                {chr(10).join(f"- {insight}" for insight in insights)}

                上記の洞察を統合し、全体的なビジネス洞察を生成してください。
                """

                messages = [
                    AIMessage(
                        role="system",
                        content="あなたはビジネスアナリストです。複数のデータソースから統合的な洞察を生成します。",
                    ),
                    AIMessage(role="user", content=prompt),
                ]

                response = await self.model_manager.generate_response(
                    messages, max_tokens=200
                )

                if response.success:
                    integrated_insights = self._parse_integrated_insights(
                        response.content
                    )
                    return integrated_insights

            return insights[:3]  # 上位3件を返す

        except Exception as e:
            logger.error(f"全体洞察生成エラー: {e}")
            return ["洞察の生成に失敗しました"]

    async def _generate_overall_recommendations(
        self, analyses: Dict[str, Any]
    ) -> List[str]:
        """全体的な推奨事項を生成"""
        try:
            recommendations = []

            # 各分析結果から推奨事項を抽出
            for analysis_type, analysis_data in analyses.items():
                if "recommendations" in analysis_data:
                    recommendations.extend(analysis_data["recommendations"])

            # AIで優先順位付け
            if recommendations:
                prompt = f"""
                個別分析の結果から抽出された推奨事項:
                {chr(10).join(f"- {rec}" for rec in recommendations)}

                上記の推奨事項を優先順位付けし、最も重要なものから順にリストアップしてください。
                """

                messages = [
                    AIMessage(
                        role="system",
                        content="あなたはビジネスコンサルタントです。複数の推奨事項から優先順位を決定します。",
                    ),
                    AIMessage(role="user", content=prompt),
                ]

                response = await self.model_manager.generate_response(
                    messages, max_tokens=250
                )

                if response.success:
                    prioritized_recommendations = (
                        self._parse_prioritized_recommendations(response.content)
                    )
                    return prioritized_recommendations

            return recommendations[:3]  # 上位3件を返す

        except Exception as e:
            logger.error(f"全体推奨事項生成エラー: {e}")
            return ["推奨事項の生成に失敗しました"]

    def _parse_integrated_insights(self, ai_response: str) -> List[str]:
        """統合洞察をパース"""
        try:
            insights = []

            if "洞察" in ai_response or "分析" in ai_response:
                lines = ai_response.split("\n")
                for line in lines:
                    if any(
                        keyword in line
                        for keyword in ["洞察", "分析", "考察", "ポイント"]
                    ):
                        insights.append(line.strip())

            return insights if insights else ["統合的な洞察を抽出できませんでした"]

        except Exception as e:
            logger.error(f"統合洞察パースエラー: {e}")
            return ["パースエラーにより洞察を抽出できませんでした"]

    def _parse_prioritized_recommendations(self, ai_response: str) -> List[str]:
        """優先順位付き推奨事項をパース"""
        try:
            recommendations = []

            if "優先" in ai_response or "推奨" in ai_response:
                lines = ai_response.split("\n")
                for line in lines:
                    if any(
                        keyword in line for keyword in ["優先", "推奨", "実施", "検討"]
                    ):
                        recommendations.append(line.strip())

            return (
                recommendations
                if recommendations
                else ["優先順位を決定できませんでした"]
            )

        except Exception as e:
            logger.error(f"優先推奨事項パースエラー: {e}")
            return ["パースエラーにより優先順位を決定できませんでした"]

    async def generate_comprehensive_report(
        self, report_type: str = "weekly"
    ) -> Dict[str, Any]:
        """
        包括的なレポートを生成

        Args:
            report_type: レポート種類 ("daily", "weekly", "monthly")

        Returns:
            レポートデータ
        """
        try:
            logger.info(f"包括的レポートを生成中: {report_type}")

            # レポート期間の設定
            end_date = datetime.now()
            if report_type == "daily":
                start_date = end_date - timedelta(days=1)
            elif report_type == "weekly":
                start_date = end_date - timedelta(days=7)
            elif report_type == "monthly":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=7)  # デフォルトは週次

            # 包括的な分析を実行
            comprehensive_analysis = await self.analyze_system_events()

            # レポート構造の作成
            report = {
                "report_id": f"report_{report_type}_{end_date.strftime('%Y%m%d_%H%M%S')}",
                "report_type": report_type,
                "period": {"start": start_date, "end": end_date},
                "generated_at": end_date,
                "executive_summary": await self._generate_executive_summary(
                    comprehensive_analysis
                ),
                "detailed_analysis": comprehensive_analysis,
                "key_metrics": await self._get_key_metrics(start_date, end_date),
                "anomalies": [
                    a
                    for a in self.anomaly_history
                    if start_date <= a.detection_time <= end_date
                ],
                "trends": await self._analyze_long_term_trends(start_date, end_date),
                "forecasts": await self._generate_forecasts(),
                "strategic_recommendations": await self._generate_strategic_recommendations(
                    comprehensive_analysis
                ),
            }

            logger.info(f"包括的レポート生成完了: {report['report_id']}")

            return {"success": True, "report": report}

        except Exception as e:
            logger.error(f"包括的レポート生成エラー: {e}")
            return {"success": False, "error": str(e), "report": {}}

    async def _generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """エグゼクティブサマリーを生成"""
        try:
            # AIでサマリーを生成
            prompt = f"""
            システム分析結果の概要:
            - 売上分析: {analysis.get("analyses", {}).get("sales", {}).get("summary", {})}
            - 在庫分析: {analysis.get("analyses", {}).get("inventory", {}).get("summary", {})}
            - 財務分析: {analysis.get("analyses", {}).get("financial", {}).get("summary", {})}
            - システム分析: {analysis.get("analyses", {}).get("system", {}).get("summary", {})}

            上記の分析結果から、エグゼクティブ向けの簡潔なサマリーを作成してください。
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたはビジネスアナリストです。複雑なデータを簡潔にまとめ、エグゼクティブに報告します。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=200
            )

            if response.success:
                return response.content.strip()
            else:
                return "分析結果のサマリー生成に失敗しました。"

        except Exception as e:
            logger.error(f"エグゼクティブサマリー生成エラー: {e}")
            return "サマリーの生成に失敗しました。"

    async def _get_key_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, BusinessMetric]:
        """主要指標を取得"""
        try:
            # 最新の指標を返す（簡易実装）
            return {
                "revenue": BusinessMetric(
                    metric_name="total_revenue",
                    value=0,
                    previous_value=0,
                    change_percentage=0.0,
                    trend="stable",
                    unit="JPY",
                ),
                "profit": BusinessMetric(
                    metric_name="gross_profit",
                    value=0,
                    previous_value=0,
                    change_percentage=0.0,
                    trend="stable",
                    unit="JPY",
                ),
                "inventory": BusinessMetric(
                    metric_name="inventory_efficiency",
                    value=0.8,
                    previous_value=0.75,
                    change_percentage=6.7,
                    trend="up",
                    unit="ratio",
                ),
            }

        except Exception as e:
            logger.error(f"主要指標取得エラー: {e}")
            return {}

    async def _analyze_long_term_trends(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """長期トレンドを分析"""
        try:
            # 簡易的なトレンド分析
            return {
                "sales_growth": 0.05,  # 5%成長
                "inventory_optimization": 0.1,  # 10%効率化
                "customer_satisfaction": 0.85,  # 85%満足度
                "market_position": "stable",
            }

        except Exception as e:
            logger.error(f"長期トレンド分析エラー: {e}")
            return {}

    async def _generate_forecasts(self) -> Dict[str, Any]:
        """予測を生成"""
        try:
            # 簡易的な予測（実際の実装では機械学習モデルを使用）
            return {
                "next_week": {
                    "predicted_revenue": 0,
                    "confidence": 0.7,
                    "factors": ["季節性", "プロモーション効果"],
                },
                "next_month": {
                    "predicted_demand": {},
                    "confidence": 0.6,
                    "factors": ["市場トレンド", "競合状況"],
                },
            }

        except Exception as e:
            logger.error(f"予測生成エラー: {e}")
            return {}

    async def _generate_strategic_recommendations(
        self, analysis: Dict[str, Any]
    ) -> List[str]:
        """戦略的推奨事項を生成"""
        try:
            # AIで戦略的推奨事項を生成
            prompt = f"""
            システム分析結果に基づいて、戦略的な推奨事項を作成してください。
            分析結果の概要:
            {analysis.get("overall_recommendations", [])}

            以下の観点から戦略を提案：
            1. 事業成長のための施策
            2. リスク管理と対策
            3. 効率化とコスト削減
            4. 顧客満足度向上
            """

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは戦略コンサルタントです。データ分析に基づいて実践的な戦略を提案します。",
                ),
                AIMessage(role="user", content=prompt),
            ]

            response = await self.model_manager.generate_response(
                messages, max_tokens=300
            )

            if response.success:
                strategic_recommendations = self._parse_strategic_recommendations(
                    response.content
                )
                return strategic_recommendations
            else:
                return ["戦略的推奨事項の生成に失敗しました"]

        except Exception as e:
            logger.error(f"戦略的推奨事項生成エラー: {e}")
            return ["戦略的推奨事項の生成に失敗しました"]

    def _parse_strategic_recommendations(self, ai_response: str) -> List[str]:
        """戦略的推奨事項をパース"""
        try:
            recommendations = []

            if "戦略" in ai_response or "提案" in ai_response:
                lines = ai_response.split("\n")
                for line in lines:
                    if any(
                        keyword in line
                        for keyword in ["戦略", "提案", "実施", "推進", "検討"]
                    ):
                        recommendations.append(line.strip())

            return (
                recommendations
                if recommendations
                else ["戦略的推奨事項を抽出できませんでした"]
            )

        except Exception as e:
            logger.error(f"戦略的推奨事項パースエラー: {e}")
            return ["パースエラーにより戦略的推奨事項を抽出できませんでした"]

    def get_anomaly_history(self, limit: int = 50) -> List[AnomalyDetection]:
        """異常検出履歴を取得"""
        return self.anomaly_history[-limit:] if self.anomaly_history else []

    def get_business_metrics(
        self, metric_type: str = None, limit: int = 20
    ) -> Dict[str, List[BusinessMetric]]:
        """ビジネス指標を取得"""
        if metric_type:
            return {metric_type: self.metrics_cache.get(metric_type, [])[-limit:]}
        else:
            return {k: v[-limit:] for k, v in self.metrics_cache.items()}

    def get_analysis_cache(self) -> Dict[str, AnalysisResult]:
        """分析キャッシュを取得"""
        return self.analysis_cache.copy()

    def get_event_stats(self) -> Dict[str, Any]:
        """イベント統計を取得"""
        try:
            # イベント追跡サービスから統計を取得
            recent_events = self.event_tracker.get_recent_events()
            total_events = len(recent_events)

            # イベントタイプ別のカウント
            event_types = {}
            for event in recent_events:
                event_type = event.get("type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

            return {
                "total_events": total_events,
                "event_types": event_types,
                "recent_events_count": min(total_events, 10),
            }

        except Exception as e:
            logger.error(f"イベント統計取得エラー: {e}")
            return {"total_events": 0, "event_types": {}, "error": str(e)}

    async def analyze_daily_trends(self) -> Dict[str, Any]:
        """日次トレンドを分析（実際のデータに基づく）"""
        try:
            # 現在の売上データを取得
            from src.agents.management_agent import management_agent

            current_metrics = management_agent.get_business_metrics()

            current_sales = current_metrics.get("sales", 0)

            # 前日データがない場合（初日）は"starting"
            # 実際の実装ではデータベースから前日データを取得
            previous_sales = 0  # シミュレーションでは前日データなし

            if current_sales == 0 and previous_sales == 0:
                revenue_trend = "starting"  # 開始段階
            elif previous_sales > 0:
                change_pct = (current_sales - previous_sales) / previous_sales
                if change_pct > 0.05:
                    revenue_trend = "increasing"
                elif change_pct < -0.05:
                    revenue_trend = "decreasing"
                else:
                    revenue_trend = "stable"
            else:
                revenue_trend = "stable"  # データなしの場合は安定とみなす

            return {
                "revenue_trend": revenue_trend,
                "inventory_trend": "stable",
                "customer_trend": "stable",
            }
        except Exception as e:
            logger.error(f"日次トレンド分析エラー: {e}")
            return {
                "revenue_trend": "unknown",
                "inventory_trend": "unknown",
                "customer_trend": "unknown",
            }

    async def generate_daily_report(self) -> Dict[str, Any]:
        """日次レポートを生成（簡易版）"""
        try:
            # 簡易的なレポート生成
            return {
                "insights": [
                    "売上は順調に推移しています",
                    "顧客満足度が改善されています",
                    "在庫回転率が向上しています",
                ],
                "recommendations": [
                    "引き続き高水準のサービスを提供してください",
                    "新商品の投入も検討してください",
                ],
            }
        except Exception as e:
            logger.error(f"日次レポート生成エラー: {e}")
            return {"insights": ["レポート生成に失敗しました"], "recommendations": []}


# グローバルインスタンス
analytics_agent = AnalyticsAgent()
