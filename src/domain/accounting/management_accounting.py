import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum

from src.models.product import Product
from src.models.transaction import Transaction
from src.accounting.journal_entry import JournalEntryProcessor, journal_processor
from src.config.settings import settings

logger = logging.getLogger(__name__)

class ProfitabilityRating(str, Enum):
    """収益性評価"""
    EXCELLENT = "excellent"  # 粗利率 30%以上
    GOOD = "good"           # 粗利率 20-30%
    AVERAGE = "average"     # 粗利率 10-20%
    POOR = "poor"          # 粗利率 5-10%
    UNPROFITABLE = "unprofitable"  # 粗利率 5%未満

class EfficiencyRating(str, Enum):
    """効率性評価"""
    EXCELLENT = "excellent"  # 回転率 12回以上/年
    GOOD = "good"           # 回転率 8-12回/年
    AVERAGE = "average"     # 回転率 4-8回/年
    POOR = "poor"          # 回転率 2-4回/年
    INEFFICIENT = "inefficient"  # 回転率 2回未満/年

@dataclass
class ProductProfitability:
    """商品別収益性分析結果"""
    product_id: str
    product_name: str
    period: Dict[str, str]
    sales_revenue: float
    cost_of_goods: float
    gross_profit: float
    gross_margin: float
    profitability_rating: ProfitabilityRating
    sales_volume: int
    average_price: float
    total_cost: float

@dataclass
class InventoryEfficiency:
    """在庫効率性分析結果"""
    product_id: str
    product_name: str
    inventory_turnover_ratio: float
    inventory_turnover_days: float
    efficiency_rating: EfficiencyRating
    average_inventory_value: float
    total_inventory_cost: float
    period_days: int

class ManagementAccountingAnalyzer:
    """管理会計分析クラス"""

    def __init__(self):
        self.journal_processor = journal_processor

    def analyze_product_profitability(self, product_id: str, period_days: int = 30) -> ProductProfitability:
        """商品別収益性分析"""
        logger.info(f"商品別収益性分析開始: {product_id}, 期間={period_days}日")

        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=period_days)

            # 商品別売上を取得
            product_sales = self._get_product_sales(product_id, start_date, end_date)

            # 商品別仕入原価を取得
            product_costs = self._get_product_costs(product_id, start_date, end_date)

            # 粗利計算
            gross_profit = product_sales - product_costs
            gross_margin = gross_profit / product_sales if product_sales > 0 else 0

            # 収益性評価
            profitability_rating = self._rate_profitability(gross_margin)

            # 追加情報（簡易版）
            sales_volume = int(product_sales / 150) if product_sales > 0 else 0  # 仮の平均単価で計算
            average_price = 150.0  # 仮の平均単価

            return ProductProfitability(
                product_id=product_id,
                product_name=f"商品_{product_id}",  # 実際には商品情報から取得
                period={"start": start_date.isoformat(), "end": end_date.isoformat()},
                sales_revenue=product_sales,
                cost_of_goods=product_costs,
                gross_profit=gross_profit,
                gross_margin=gross_margin,
                profitability_rating=profitability_rating,
                sales_volume=sales_volume,
                average_price=average_price,
                total_cost=product_costs
            )

        except Exception as e:
            logger.error(f"商品別収益性分析エラー: {e}")
            # エラー時はデフォルト値を返す
            return ProductProfitability(
                product_id=product_id,
                product_name=f"商品_{product_id}",
                period={"start": start_date.isoformat(), "end": end_date.isoformat()},
                sales_revenue=0.0,
                cost_of_goods=0.0,
                gross_profit=0.0,
                gross_margin=0.0,
                profitability_rating=ProfitabilityRating.UNPROFITABLE,
                sales_volume=0,
                average_price=0.0,
                total_cost=0.0
            )

    def _get_product_sales(self, product_id: str, start_date: date, end_date: date) -> float:
        """商品別売上を取得"""
        # 簡易的な実装：売上高から推定
        sales_revenue = self.journal_processor.get_account_balance("4001", start_date, end_date)

        # 商品別配分（実際にはより詳細な計算が必要）
        # ここでは全売上の20%をこの商品の売上とする（サンプル値）
        return abs(sales_revenue) * 0.2

    def _get_product_costs(self, product_id: str, start_date: date, end_date: date) -> float:
        """商品別原価を取得"""
        # 簡易的な実装：仕入高から推定
        cost_of_goods = self.journal_processor.get_account_balance("5001", start_date, end_date)

        # 商品別配分（実際にはより詳細な計算が必要）
        return abs(cost_of_goods) * 0.2

    def _rate_profitability(self, gross_margin: float) -> ProfitabilityRating:
        """収益性を評価"""
        if gross_margin >= 0.30:
            return ProfitabilityRating.EXCELLENT
        elif gross_margin >= 0.20:
            return ProfitabilityRating.GOOD
        elif gross_margin >= 0.10:
            return ProfitabilityRating.AVERAGE
        elif gross_margin >= 0.05:
            return ProfitabilityRating.POOR
        else:
            return ProfitabilityRating.UNPROFITABLE

    def calculate_inventory_turnover(self, product_id: str) -> InventoryEfficiency:
        """在庫回転率計算"""
        logger.info(f"在庫回転率計算開始: {product_id}")

        try:
            # 平均在庫金額を取得
            avg_inventory_value = self._get_average_inventory_value(product_id)

            # 年間売上原価を取得
            annual_cogs = self._get_annual_cogs(product_id)

            # 在庫回転率計算
            turnover_ratio = annual_cogs / avg_inventory_value if avg_inventory_value > 0 else 0

            # 在庫回転日数計算
            turnover_days = 365 / turnover_ratio if turnover_ratio > 0 else 0

            # 効率性評価
            efficiency_rating = self._rate_inventory_efficiency(turnover_ratio)

            return InventoryEfficiency(
                product_id=product_id,
                product_name=f"商品_{product_id}",
                inventory_turnover_ratio=turnover_ratio,
                inventory_turnover_days=turnover_days,
                efficiency_rating=efficiency_rating,
                average_inventory_value=avg_inventory_value,
                total_inventory_cost=avg_inventory_value * 12,  # 年間総在庫コスト
                period_days=365
            )

        except Exception as e:
            logger.error(f"在庫回転率計算エラー: {e}")
            return InventoryEfficiency(
                product_id=product_id,
                product_name=f"商品_{product_id}",
                inventory_turnover_ratio=0.0,
                inventory_turnover_days=0.0,
                efficiency_rating=EfficiencyRating.INEFFICIENT,
                average_inventory_value=0.0,
                total_inventory_cost=0.0,
                period_days=365
            )

    def _get_average_inventory_value(self, product_id: str) -> float:
        """平均在庫金額を取得"""
        # 簡易的な実装：在庫残高を取得
        inventory_balance = self.journal_processor.get_account_balance("1101")
        return abs(inventory_balance) * 0.2  # 商品別配分

    def _get_annual_cogs(self, product_id: str) -> float:
        """年間売上原価を取得"""
        # 年間仕入高を取得
        annual_cost = self.journal_processor.get_account_balance("5001", None, date.today())
        return abs(annual_cost) * 0.2  # 商品別配分

    def _rate_inventory_efficiency(self, turnover_ratio: float) -> EfficiencyRating:
        """在庫効率性を評価"""
        if turnover_ratio >= 12:
            return EfficiencyRating.EXCELLENT
        elif turnover_ratio >= 8:
            return EfficiencyRating.GOOD
        elif turnover_ratio >= 4:
            return EfficiencyRating.AVERAGE
        elif turnover_ratio >= 2:
            return EfficiencyRating.POOR
        else:
            return EfficiencyRating.INEFFICIENT

    def analyze_period_profitability(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """期間別収益性分析"""
        logger.info(f"期間別収益性分析開始: {start_date} - {end_date}")

        try:
            # 売上高を取得
            sales_revenue = abs(self.journal_processor.get_account_balance("4001", start_date, end_date))

            # 売上原価を取得
            cost_of_goods = abs(self.journal_processor.get_account_balance("5001", start_date, end_date))

            # 粗利益計算
            gross_profit = sales_revenue - cost_of_goods
            gross_margin = gross_profit / sales_revenue if sales_revenue > 0 else 0

            # 販管費を取得
            operating_expenses = abs(self.journal_processor.get_account_balance("6001", start_date, end_date))

            # 営業利益計算
            operating_profit = gross_profit - operating_expenses
            operating_margin = operating_profit / sales_revenue if sales_revenue > 0 else 0

            return {
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "sales_revenue": sales_revenue,
                "cost_of_goods_sold": cost_of_goods,
                "gross_profit": gross_profit,
                "gross_margin": gross_margin,
                "operating_expenses": operating_expenses,
                "operating_profit": operating_profit,
                "operating_margin": operating_margin,
                "analysis": {
                    "profitability": "良好" if operating_profit > 0 else "赤字",
                    "efficiency": "効率的" if gross_margin > 0.2 else "改善必要",
                    "cost_control": "適切" if operating_expenses < sales_revenue * 0.3 else "見直し必要"
                }
            }

        except Exception as e:
            logger.error(f"期間別収益性分析エラー: {e}")
            return {
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "error": str(e)
            }

    def generate_profitability_report(self, product_ids: Optional[List[str]] = None, period_days: int = 30) -> Dict[str, Any]:
        """収益性レポートを生成"""
        logger.info("収益性レポート生成開始")

        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=period_days)

            report = {
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "products": [],
                "summary": {},
                "generated_at": datetime.now().isoformat()
            }

            # 商品別分析
            if product_ids:
                for product_id in product_ids:
                    profitability = self.analyze_product_profitability(product_id, period_days)
                    report["products"].append(profitability.__dict__)
            else:
                # 全商品の分析（簡易版：サンプル商品のみ）
                for i in range(1, 4):  # サンプルとして3商品
                    profitability = self.analyze_product_profitability(f"product_{i}", period_days)
                    report["products"].append(profitability.__dict__)

            # サマリ計算
            if report["products"]:
                total_revenue = sum(p["sales_revenue"] for p in report["products"])
                total_cost = sum(p["cost_of_goods"] for p in report["products"])
                total_profit = sum(p["gross_profit"] for p in report["products"])

                report["summary"] = {
                    "total_products": len(report["products"]),
                    "total_revenue": total_revenue,
                    "total_cost": total_cost,
                    "total_gross_profit": total_profit,
                    "overall_gross_margin": total_profit / total_revenue if total_revenue > 0 else 0,
                    "profitable_products": sum(1 for p in report["products"] if p["gross_profit"] > 0),
                    "unprofitable_products": sum(1 for p in report["products"] if p["gross_profit"] <= 0)
                }

            return report

        except Exception as e:
            logger.error(f"収益性レポート生成エラー: {e}")
            return {"error": str(e)}

    def analyze_trend(self, periods: int = 6) -> Dict[str, Any]:
        """トレンド分析"""
        logger.info(f"トレンド分析開始: {periods}期間")

        try:
            trends = []
            current_date = date.today()

            for i in range(periods):
                # 各期間の開始日と終了日を計算
                period_end = current_date - timedelta(days=i*30)
                period_start = period_end - timedelta(days=30)

                # 期間別収益性分析
                period_analysis = self.analyze_period_profitability(period_start, period_end)
                trends.append({
                    "period": f"{period_start}〜{period_end}",
                    "revenue": period_analysis.get("sales_revenue", 0),
                    "profit": period_analysis.get("operating_profit", 0),
                    "margin": period_analysis.get("operating_margin", 0)
                })

            # トレンド方向の計算
            if len(trends) >= 2:
                revenue_trend = self._calculate_trend_direction([t["revenue"] for t in trends])
                profit_trend = self._calculate_trend_direction([t["profit"] for t in trends])
                margin_trend = self._calculate_trend_direction([t["margin"] for t in trends])
            else:
                revenue_trend = profit_trend = margin_trend = "insufficient_data"

            return {
                "periods": trends,
                "trends": {
                    "revenue": revenue_trend,
                    "profit": profit_trend,
                    "margin": margin_trend
                },
                "analysis": {
                    "revenue_trend": "上昇中" if revenue_trend == "increasing" else "下降中" if revenue_trend == "decreasing" else "安定",
                    "profit_trend": "改善中" if profit_trend == "increasing" else "悪化中" if profit_trend == "decreasing" else "安定",
                    "margin_trend": "向上中" if margin_trend == "increasing" else "低下中" if margin_trend == "decreasing" else "安定"
                }
            }

        except Exception as e:
            logger.error(f"トレンド分析エラー: {e}")
            return {"error": str(e)}

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """トレンド方向を計算"""
        if len(values) < 2:
            return "insufficient_data"

        # 単純な線形トレンド（最新値と過去値の比較）
        recent_avg = sum(values[:2]) / 2  # 最新2期間の平均
        older_avg = sum(values[-2:]) / 2   # 最古2期間の平均

        if recent_avg > older_avg * 1.05:  # 5%以上の増加
            return "increasing"
        elif recent_avg < older_avg * 0.95:  # 5%以上の減少
            return "decreasing"
        else:
            return "stable"

    def generate_management_dashboard_data(self) -> Dict[str, Any]:
        """管理会計ダッシュボードデータを生成"""
        logger.info("管理会計ダッシュボードデータ生成開始")

        try:
            # 現在の収益性分析
            current_profitability = self.analyze_period_profitability(
                date.today().replace(day=1), date.today()
            )

            # 商品別収益性レポート
            profitability_report = self.generate_profitability_report(period_days=30)

            # トレンド分析
            trend_analysis = self.analyze_trend(periods=6)

            # 在庫効率性分析（サンプル商品）
            inventory_efficiency = []
            for i in range(1, 4):
                efficiency = self.calculate_inventory_turnover(f"product_{i}")
                inventory_efficiency.append(efficiency.__dict__)

            return {
                "current_period": current_profitability,
                "product_profitability": profitability_report,
                "trend_analysis": trend_analysis,
                "inventory_efficiency": inventory_efficiency,
                "kpi_summary": {
                    "gross_margin": current_profitability.get("gross_margin", 0),
                    "operating_margin": current_profitability.get("operating_margin", 0),
                    "inventory_turnover": sum(e["inventory_turnover_ratio"] for e in inventory_efficiency) / len(inventory_efficiency) if inventory_efficiency else 0,
                    "profitable_products_ratio": profitability_report.get("summary", {}).get("profitable_products", 0) / max(profitability_report.get("summary", {}).get("total_products", 1), 1)
                },
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"管理会計ダッシュボードデータ生成エラー: {e}")
            return {"error": str(e)}

# グローバルインスタンス
management_analyzer = ManagementAccountingAnalyzer()
