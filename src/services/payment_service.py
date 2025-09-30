import asyncio
import random
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.models.transaction import PaymentMethod, PaymentDetails, Transaction
from src.config.settings import settings

logger = logging.getLogger(__name__)

class PaymentStatus(str, Enum):
    """決済ステータス"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDING = "refunding"
    REFUNDED = "refunded"

class PaymentError(str, Enum):
    """決済エラー種別"""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_DECLINED = "card_declined"
    NETWORK_ERROR = "network_error"
    INVALID_CARD = "invalid_card"
    EXPIRED_CARD = "expired_card"
    SYSTEM_ERROR = "system_error"

@dataclass
class PaymentResult:
    """決済結果"""
    success: bool
    status: PaymentStatus
    payment_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None
    fee: float = 0.0

@dataclass
class RefundResult:
    """返金結果"""
    success: bool
    refund_id: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

class PaymentSimulator:
    """決済シミュレーター"""

    def __init__(self):
        self.success_rate = 0.95  # 決済成功率（95%）
        self.processing_delay = (1.0, 3.0)  # 処理時間範囲（秒）
        self.error_patterns = {
            PaymentError.INSUFFICIENT_FUNDS: 0.3,
            PaymentError.CARD_DECLINED: 0.25,
            PaymentError.NETWORK_ERROR: 0.2,
            PaymentError.INVALID_CARD: 0.15,
            PaymentError.EXPIRED_CARD: 0.1
        }

    def _generate_payment_id(self) -> str:
        """決済IDを生成"""
        return f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

    def _get_random_error(self) -> PaymentError:
        """ランダムなエラーを取得"""
        errors = list(self.error_patterns.keys())
        probabilities = list(self.error_patterns.values())
        return random.choices(errors, weights=probabilities)[0]

    def _simulate_processing_delay(self) -> float:
        """処理遅延をシミュレート"""
        min_delay, max_delay = self.processing_delay
        return random.uniform(min_delay, max_delay)

    async def simulate_payment(self, amount: float, method: PaymentMethod) -> PaymentResult:
        """決済をシミュレート"""
        logger.info(f"決済シミュレーション開始: 金額={amount}, 方法={method}")

        # 処理遅延をシミュレート
        delay = self._simulate_processing_delay()
        await asyncio.sleep(delay)

        # 決済成功/失敗を決定
        success = random.random() < self.success_rate

        if success:
            payment_id = self._generate_payment_id()
            result = PaymentResult(
                success=True,
                status=PaymentStatus.COMPLETED,
                payment_id=payment_id,
                processed_at=datetime.now(),
                fee=amount * 0.029 + 30  # Stripe手数料をシミュレート（2.9% + 30円）
            )
            logger.info(f"決済成功: {payment_id}")
        else:
            error = self._get_random_error()
            result = PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_code=error.value,
                error_message=self._get_error_message(error, method),
                processed_at=datetime.now()
            )
            logger.warning(f"決済失敗: {error.value} - {result.error_message}")

        return result

    def _get_error_message(self, error: PaymentError, method: PaymentMethod) -> str:
        """エラーメッセージを取得"""
        messages = {
            PaymentError.INSUFFICIENT_FUNDS: "残高が不足しています",
            PaymentError.CARD_DECLINED: "カードが拒否されました",
            PaymentError.NETWORK_ERROR: "ネットワークエラーが発生しました",
            PaymentError.INVALID_CARD: "カード情報が無効です",
            PaymentError.EXPIRED_CARD: "カードの有効期限が切れています",
            PaymentError.SYSTEM_ERROR: "システムエラーが発生しました"
        }
        return messages.get(error, "不明なエラー")

class SalesSimulationModel:
    """販売シミュレーションモデル"""

    def __init__(self):
        # 時間帯別需要パターン（24時間）
        self.time_demand_pattern = self._generate_time_demand_pattern()

        # 商品別人気度（0-1のスケール）
        self.product_popularity = {
            "drink_cola": 0.8,
            "drink_tea": 0.6,
            "snack_chips": 0.7,
            "snack_chocolate": 0.5
        }

        # 曜日別需要変動係数
        self.day_demand_multiplier = {
            0: 0.7,  # 月曜日
            1: 0.8,  # 火曜日
            2: 0.9,  # 水曜日
            3: 1.0,  # 木曜日
            4: 1.2,  # 金曜日
            5: 1.3,  # 土曜日
            6: 1.1   # 日曜日
        }

        # 季節・イベント要因
        self.seasonal_factors = {
            "spring": 1.0,
            "summer": 1.2,
            "autumn": 0.9,
            "winter": 0.8
        }

    def _generate_time_demand_pattern(self) -> List[float]:
        """時間帯別需要パターンを生成"""
        pattern = []

        for hour in range(24):
            if 6 <= hour < 9:  # 朝の通勤・通学時間
                demand = 0.3 + random.uniform(0.2, 0.4)
            elif 9 <= hour < 12:  # 午前中
                demand = 0.6 + random.uniform(0.1, 0.3)
            elif 12 <= hour < 15:  # お昼時間
                demand = 1.0 + random.uniform(0.2, 0.5)
            elif 15 <= hour < 18:  # 午後
                demand = 0.7 + random.uniform(0.1, 0.3)
            elif 18 <= hour < 22:  # 夕方・夜
                demand = 0.9 + random.uniform(0.2, 0.4)
            else:  # 深夜・早朝
                demand = 0.1 + random.uniform(0.0, 0.2)

            pattern.append(demand)

        return pattern

    def predict_demand(self, product_id: str, current_hour: int, current_weekday: int) -> float:
        """商品の需要を予測"""
        base_demand = self.product_popularity.get(product_id, 0.5)

        # 時間帯要因
        time_factor = self.time_demand_pattern[current_hour]

        # 曜日要因
        day_factor = self.day_demand_multiplier.get(current_weekday, 1.0)

        # ランダム要因（実際の予測誤差をシミュレート）
        random_factor = random.uniform(0.8, 1.2)

        predicted_demand = base_demand * time_factor * day_factor * random_factor

        return min(predicted_demand, 2.0)  # 上限を2.0に制限

    def simulate_customer_behavior(self, product_id: str) -> Dict[str, Any]:
        """顧客行動をシミュレート"""
        # 購入確率の計算
        popularity = self.product_popularity.get(product_id, 0.5)
        purchase_probability = popularity * random.uniform(0.3, 0.8)

        # 顧客満足度のシミュレーション
        satisfaction_base = 0.7 + (popularity * 0.2)
        satisfaction = min(satisfaction_base + random.uniform(-0.1, 0.1), 1.0)

        # リピート購入確率
        repeat_probability = satisfaction * 0.6 + random.uniform(0.1, 0.3)

        return {
            "purchase_probability": purchase_probability,
            "satisfaction_score": satisfaction,
            "repeat_purchase_probability": repeat_probability,
            "price_sensitivity": random.uniform(0.3, 0.8),  # 価格感度
            "promotion_responsiveness": random.uniform(0.4, 0.9)  # プロモーション反応度
        }

    def calculate_optimal_price(self, product_id: str, cost_price: float) -> float:
        """最適価格を計算"""
        popularity = self.product_popularity.get(product_id, 0.5)

        # 需要に基づく価格設定
        demand_multiplier = 1.0 + (popularity * 0.5)

        # コストベースの価格設定
        base_price = cost_price * 1.3  # 30%マージン

        # 人気度による調整
        popularity_adjustment = 1.0 + (popularity * 0.3)

        optimal_price = base_price * demand_multiplier * popularity_adjustment

        # 価格を現実的な範囲に調整
        return round(optimal_price / 10) * 10  # 10円単位で丸め

    def simulate_market_trends(self) -> Dict[str, float]:
        """市場トレンドをシミュレート"""
        trends = {}

        for product_id in self.product_popularity.keys():
            # トレンドの変動（-0.1〜0.1の範囲）
            trend_change = random.uniform(-0.1, 0.1)

            # 季節要因の影響
            seasonal_impact = random.uniform(-0.05, 0.05)

            total_change = trend_change + seasonal_impact
            trends[product_id] = max(0.1, min(1.0, self.product_popularity[product_id] + total_change))

        # 人気度の更新
        for product_id, new_popularity in trends.items():
            self.product_popularity[product_id] = new_popularity

        return trends

    def get_sales_forecast(self, product_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """販売予測を取得"""
        forecast = []
        current_time = datetime.now()

        for day in range(days):
            forecast_date = current_time + timedelta(days=day)
            hour = forecast_date.hour
            weekday = forecast_date.weekday()

            predicted_demand = self.predict_demand(product_id, hour, weekday)
            customer_behavior = self.simulate_customer_behavior(product_id)

            forecast.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "hour": hour,
                "predicted_demand": predicted_demand,
                "purchase_probability": customer_behavior["purchase_probability"],
                "expected_sales": predicted_demand * customer_behavior["purchase_probability"]
            })

        return forecast

    def simulate_bulk_purchase_discount(self, base_price: float, quantity: int) -> float:
        """まとめ買い割引をシミュレート"""
        if quantity <= 1:
            return base_price

        # 数量に応じた割引率
        if quantity >= 10:
            discount_rate = 0.15  # 15%割引
        elif quantity >= 5:
            discount_rate = 0.10  # 10%割引
        elif quantity >= 3:
            discount_rate = 0.05  # 5%割引
        else:
            discount_rate = 0.0

        discounted_price = base_price * (1 - discount_rate)
        return round(discounted_price / 10) * 10

    def calculate_inventory_turnover(self, product_id: str, current_stock: int, avg_daily_sales: float) -> float:
        """在庫回転率を計算"""
        if current_stock <= 0:
            return 0.0

        # 在庫回転率 = 年間販売量 / 平均在庫量
        annual_sales = avg_daily_sales * 365
        turnover_rate = annual_sales / current_stock

        return turnover_rate

    def simulate_seasonal_demand(self, product_id: str, current_month: int) -> float:
        """季節需要をシミュレート"""
        # 月別の季節要因
        seasonal_multipliers = {
            1: 0.8,   # 1月（冬）
            2: 0.8,   # 2月（冬）
            3: 0.9,   # 3月（春）
            4: 1.0,   # 4月（春）
            5: 1.1,   # 5月（春）
            6: 1.2,   # 6月（夏）
            7: 1.3,   # 7月（夏）
            8: 1.3,   # 8月（夏）
            9: 1.1,   # 9月（秋）
            10: 1.0,  # 10月（秋）
            11: 0.9,  # 11月（秋）
            12: 0.8   # 12月（冬）
        }

        base_demand = self.product_popularity.get(product_id, 0.5)
        seasonal_multiplier = seasonal_multipliers.get(current_month, 1.0)

        return base_demand * seasonal_multiplier

class PaymentService:
    """決済サービス（シミュレーションモデルベース）"""

    def __init__(self):
        self.simulator = PaymentSimulator()
        self.transaction_log: List[Dict[str, Any]] = []
        self.sales_model = SalesSimulationModel()

    async def process_payment(self, amount: float, method: PaymentMethod, **kwargs) -> PaymentResult:
        """決済を処理"""
        logger.info(f"決済処理開始: 金額={amount}, 方法={method}")

        try:
            # 入力検証
            if amount <= 0:
                raise ValueError("決済金額は0より大きくなければなりません")

            if method not in [PaymentMethod.CARD, PaymentMethod.MOBILE]:
                # 現金とクーポンは即時成功として処理
                if method == PaymentMethod.CASH:
                    return PaymentResult(
                        success=True,
                        status=PaymentStatus.COMPLETED,
                        payment_id=self.simulator._generate_payment_id(),
                        processed_at=datetime.now()
                    )
                elif method == PaymentMethod.COUPON:
                    return await self._process_coupon_payment(amount, **kwargs)

            # カード/モバイル決済のシミュレーション
            result = await self.simulator.simulate_payment(amount, method)

            # ログ記録
            self._log_transaction({
                "type": "payment",
                "amount": amount,
                "method": method,
                "result": result,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            logger.error(f"決済処理エラー: {e}")
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message=str(e),
                processed_at=datetime.now()
            )

    async def _process_coupon_payment(self, amount: float, coupon_code: str = None) -> PaymentResult:
        """クーポン決済を処理"""
        if not coupon_code:
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="クーポンコードが必要です",
                processed_at=datetime.now()
            )

        # クーポン検証をシミュレート（簡易版）
        valid_coupons = {
            "DISCOUNT10": 0.1,
            "DISCOUNT20": 0.2,
            "FREE": 1.0
        }

        if coupon_code not in valid_coupons:
            return PaymentResult(
                success=False,
                status=PaymentStatus.FAILED,
                error_message="無効なクーポンコードです",
                processed_at=datetime.now()
            )

        discount_rate = valid_coupons[coupon_code]
        discount_amount = amount * discount_rate

        return PaymentResult(
            success=True,
            status=PaymentStatus.COMPLETED,
            payment_id=self.simulator._generate_payment_id(),
            processed_at=datetime.now(),
            fee=0.0  # クーポンは手数料なし
        )

    async def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> RefundResult:
        """決済を返金"""
        logger.info(f"返金処理開始: 取引ID={transaction_id}")

        try:
            # 取引の検証（実際の実装ではデータベースから取得）
            transaction = await self._get_transaction(transaction_id)
            if not transaction:
                return RefundResult(
                    success=False,
                    error_message="取引が見つかりません",
                    processed_at=datetime.now()
                )

            # 返金金額の決定（指定がない場合は全額）
            refund_amount = amount if amount is not None else transaction.total_amount

            if refund_amount > transaction.total_amount:
                return RefundResult(
                    success=False,
                    error_message="返金金額が取引金額を超えています",
                    processed_at=datetime.now()
                )

            # 返金処理のシミュレーション
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)

            # 返金成功をシミュレート（高い成功率）
            success = random.random() < 0.98

            if success:
                refund_id = f"refund_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
                result = RefundResult(
                    success=True,
                    refund_id=refund_id,
                    processed_at=datetime.now()
                )
                logger.info(f"返金成功: {refund_id}")
            else:
                result = RefundResult(
                    success=False,
                    error_message="返金処理に失敗しました",
                    processed_at=datetime.now()
                )
                logger.warning(f"返金失敗: {transaction_id}")

            # ログ記録
            self._log_transaction({
                "type": "refund",
                "transaction_id": transaction_id,
                "refund_amount": refund_amount,
                "result": result,
                "timestamp": datetime.now()
            })

            return result

        except Exception as e:
            logger.error(f"返金処理エラー: {e}")
            return RefundResult(
                success=False,
                error_message=str(e),
                processed_at=datetime.now()
            )

    async def _get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """取引を取得（実際の実装ではデータベースから）"""
        # 簡易的な実装：ログから検索
        for log_entry in reversed(self.transaction_log):
            if (log_entry["type"] == "payment" and
                log_entry.get("payment_id") == transaction_id):
                return Transaction(
                    transaction_id=transaction_id,
                    machine_id="VM001",
                    items=[],  # 簡易版のため空
                    subtotal=0,
                    total_amount=log_entry["amount"]
                )
        return None

    def _log_transaction(self, log_entry: Dict[str, Any]):
        """取引ログを記録"""
        self.transaction_log.append(log_entry)

        # ログ数の制限（最新100件のみ保持）
        if len(self.transaction_log) > 100:
            self.transaction_log = self.transaction_log[-100:]

        logger.debug(f"取引ログ記録: {log_entry}")

    def get_payment_stats(self) -> Dict[str, Any]:
        """決済統計を取得"""
        if not self.transaction_log:
            return {"total_transactions": 0, "success_rate": 0.0}

        payment_logs = [log for log in self.transaction_log if log["type"] == "payment"]

        if not payment_logs:
            return {"total_transactions": 0, "success_rate": 0.0}

        successful = sum(1 for log in payment_logs if log["result"].success)
        total = len(payment_logs)

        return {
            "total_transactions": total,
            "successful_transactions": successful,
            "failed_transactions": total - successful,
            "success_rate": successful / total,
            "total_amount": sum(log["amount"] for log in payment_logs if log["result"].success),
            "total_fees": sum(log["result"].fee for log in payment_logs if log["result"].success)
        }

    def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の取引を取得"""
        return self.transaction_log[-limit:] if self.transaction_log else []

    def simulate_realistic_sale(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """現実的な販売をシミュレート"""
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        current_month = current_time.month

        # 需要予測
        predicted_demand = self.sales_model.predict_demand(product_id, current_hour, current_weekday)

        # 顧客行動シミュレーション
        customer_behavior = self.sales_model.simulate_customer_behavior(product_id)

        # 実際の販売数量を決定（予測需要と顧客行動に基づく）
        random_factor = random.uniform(0.8, 1.2)
        actual_quantity = max(0, int(predicted_demand * customer_behavior["purchase_probability"] * random_factor))

        # 価格設定
        base_price = 150.0  # 基準価格（実際には商品マスタから取得）
        optimal_price = self.sales_model.calculate_optimal_price(product_id, base_price * 0.7)

        # まとめ買い割引の適用
        if quantity > 1:
            unit_price = self.sales_model.simulate_bulk_purchase_discount(optimal_price, quantity)
        else:
            unit_price = optimal_price

        total_amount = unit_price * actual_quantity

        # 季節需要の考慮
        seasonal_demand = self.sales_model.simulate_seasonal_demand(product_id, current_month)
        seasonal_adjustment = total_amount * (seasonal_demand / self.sales_model.product_popularity.get(product_id, 0.5))

        final_amount = total_amount * (1 + (seasonal_adjustment - total_amount) / total_amount * 0.3)

        return {
            "product_id": product_id,
            "predicted_demand": predicted_demand,
            "actual_quantity": actual_quantity,
            "unit_price": round(unit_price, 2),
            "total_amount": round(final_amount, 2),
            "customer_satisfaction": customer_behavior["satisfaction_score"],
            "repeat_purchase_probability": customer_behavior["repeat_purchase_probability"],
            "seasonal_impact": seasonal_demand,
            "timestamp": current_time.isoformat(),
            "simulation_model": "advanced_v1.0"
        }

    def get_demand_forecast(self, product_id: str, days: int = 7) -> Dict[str, Any]:
        """需要予測を取得"""
        forecast = self.sales_model.get_sales_forecast(product_id, days)

        # 統計情報の計算
        total_predicted_demand = sum(item["predicted_demand"] for item in forecast)
        total_expected_sales = sum(item["expected_sales"] for item in forecast)
        avg_purchase_probability = sum(item["purchase_probability"] for item in forecast) / len(forecast)

        return {
            "product_id": product_id,
            "forecast_period_days": days,
            "forecast_data": forecast,
            "summary": {
                "total_predicted_demand": round(total_predicted_demand, 2),
                "total_expected_sales": round(total_expected_sales, 2),
                "average_purchase_probability": round(avg_purchase_probability, 3),
                "peak_demand_hour": max(forecast, key=lambda x: x["predicted_demand"])["hour"],
                "lowest_demand_hour": min(forecast, key=lambda x: x["predicted_demand"])["hour"]
            }
        }

    def simulate_market_scenario(self, scenario: str = "normal") -> Dict[str, Any]:
        """市場シナリオをシミュレート"""
        scenarios = {
            "normal": {"demand_change": 0.0, "price_sensitivity": 0.5},
            "economic_boom": {"demand_change": 0.2, "price_sensitivity": 0.3},
            "recession": {"demand_change": -0.3, "price_sensitivity": 0.8},
            "competitor_entry": {"demand_change": -0.2, "price_sensitivity": 0.7},
            "viral_marketing": {"demand_change": 0.5, "price_sensitivity": 0.2},
            "supply_shortage": {"demand_change": 0.1, "price_sensitivity": 0.6}
        }

        if scenario not in scenarios:
            scenario = "normal"

        scenario_params = scenarios[scenario]

        # 市場トレンドのシミュレーション
        market_trends = self.sales_model.simulate_market_trends()

        # シナリオに基づく影響の計算
        scenario_impact = {}
        for product_id, trend in market_trends.items():
            base_demand = self.sales_model.product_popularity[product_id]
            demand_impact = scenario_params["demand_change"]
            price_impact = scenario_params["price_sensitivity"]

            scenario_impact[product_id] = {
                "base_demand": base_demand,
                "trend_demand": trend,
                "scenario_demand": trend * (1 + demand_impact),
                "price_sensitivity": price_impact,
                "recommended_action": self._get_recommended_action(trend, demand_impact, price_impact)
            }

        return {
            "scenario": scenario,
            "scenario_parameters": scenario_params,
            "market_trends": market_trends,
            "scenario_impact": scenario_impact,
            "recommendations": self._generate_scenario_recommendations(scenario_impact)
        }

    def _get_recommended_action(self, trend_demand: float, demand_impact: float, price_sensitivity: float) -> str:
        """推奨アクションを取得"""
        final_demand = trend_demand * (1 + demand_impact)

        if final_demand > 0.8:
            if price_sensitivity < 0.5:
                return "価格を10-15%値上げし、在庫を増強"
            else:
                return "プロモーションを強化し、シェア拡大を狙う"
        elif final_demand < 0.4:
            if price_sensitivity > 0.6:
                return "価格を10-20%値下げし、需要刺激"
            else:
                return "在庫を減らし、廃盤検討"
        else:
            return "現状維持し、継続観察"

    def _generate_scenario_recommendations(self, scenario_impact: Dict[str, Any]) -> List[str]:
        """シナリオベースの推奨事項を生成"""
        recommendations = []

        high_demand_products = [
            product_id for product_id, impact in scenario_impact.items()
            if impact["scenario_demand"] > 0.7
        ]

        low_demand_products = [
            product_id for product_id, impact in scenario_impact.items()
            if impact["scenario_demand"] < 0.4
        ]

        if high_demand_products:
            recommendations.append(f"高需要商品の在庫増強を優先: {', '.join(high_demand_products)}")

        if low_demand_products:
            recommendations.append(f"低需要商品の価格戦略見直しを検討: {', '.join(low_demand_products)}")

        if not recommendations:
            recommendations.append("現在の戦略を維持し、市場動向を継続観察")

        return recommendations

    def get_advanced_analytics(self) -> Dict[str, Any]:
        """高度な分析データを取得"""
        if not self.transaction_log:
            return {"error": "取引データがありません"}

        # 時間帯別分析
        hourly_stats = {}
        for hour in range(24):
            hour_transactions = [
                log for log in self.transaction_log
                if log["timestamp"].hour == hour and log["type"] == "payment"
            ]

            if hour_transactions:
                total_amount = sum(log["amount"] for log in hour_transactions)
                avg_amount = total_amount / len(hour_transactions)
                hourly_stats[hour] = {
                    "transaction_count": len(hour_transactions),
                    "total_amount": total_amount,
                    "average_amount": avg_amount
                }

        # 商品別分析（シミュレーションモデルベース）
        product_analytics = {}
        for product_id in self.sales_model.product_popularity.keys():
            customer_behavior = self.sales_model.simulate_customer_behavior(product_id)
            forecast = self.sales_model.get_sales_forecast(product_id, 1)[0]

            product_analytics[product_id] = {
                "popularity_score": self.sales_model.product_popularity[product_id],
                "customer_satisfaction": customer_behavior["satisfaction_score"],
                "repeat_purchase_probability": customer_behavior["repeat_purchase_probability"],
                "next_hour_forecast": forecast["expected_sales"],
                "optimal_price": self.sales_model.calculate_optimal_price(product_id, 100.0)
            }

        return {
            "hourly_statistics": hourly_stats,
            "product_analytics": product_analytics,
            "market_trends": self.sales_model.simulate_market_trends(),
            "inventory_efficiency": self._calculate_inventory_efficiency(),
            "sales_model_version": "advanced_v1.0"
        }

    def _calculate_inventory_efficiency(self) -> Dict[str, float]:
        """在庫効率性を計算"""
        efficiency_metrics = {}

        for product_id in self.sales_model.product_popularity.keys():
            # シミュレーションによる在庫回転率計算
            current_stock = random.randint(10, 100)  # シミュレーション用
            avg_daily_sales = self.sales_model.product_popularity[product_id] * random.uniform(5, 20)

            turnover_rate = self.sales_model.calculate_inventory_turnover(product_id, current_stock, avg_daily_sales)

            efficiency_metrics[product_id] = {
                "turnover_rate": turnover_rate,
                "days_of_stock": current_stock / max(avg_daily_sales, 0.1),
                "efficiency_rating": self._rate_inventory_efficiency(turnover_rate)
            }

        return efficiency_metrics

    def _rate_inventory_efficiency(self, turnover_rate: float) -> str:
        """在庫効率性を評価"""
        if turnover_rate >= 12:
            return "非常に効率的"
        elif turnover_rate >= 8:
            return "効率的"
        elif turnover_rate >= 4:
            return "標準的"
        elif turnover_rate >= 2:
            return "非効率的"
        else:
            return "非常に非効率的"

# グローバルインスタンス
payment_service = PaymentService()
