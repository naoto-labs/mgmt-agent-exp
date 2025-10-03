"""
performance_monitor.py - パフォーマンス監視ツール

売上・業務KPIの継続監視・トレンド分析Tool
"""

import logging
from datetime import datetime, timedelta

from src.agents.management_agent.management_tools.get_business_metrics import (
    get_business_metrics,
)
from src.agents.shared_tools.tools.data_retrieval.check_inventory_status import (
    check_inventory_status,
)

logger = logging.getLogger(__name__)


def monitor_business_performance() -> dict:
    """事業パフォーマンスの監視"""
    logger.info("Monitoring business performance with KPIs")

    try:
        # 現在のビジネスメトリクス取得
        current_metrics = get_business_metrics()

        # 過去7日間のトレンド計算 (簡易)
        # 実際の運用では時系列DBから取得
        performance_trends = {
            "sales_trend": "stable",  # 実際: DBから計算
            "inventory_trend": "adequate",
            "customer_satisfaction_trend": "improving",
        }

        # KPIダッシュボード生成
        kpi_dashboard = {
            "timestamp": datetime.now().isoformat(),
            "current_kpis": {
                "sales": {
                    "value": current_metrics["sales"],
                    "target": 1000000,  # 月間目標
                    "achievement_rate": min(current_metrics["sales"] / 1000000, 1.0),
                    "status": "on_track"
                    if current_metrics["sales"] >= 500000
                    else "attention_needed",
                },
                "profit_margin": {
                    "value": current_metrics["profit_margin"] * 100,
                    "target": 35.0,  # 目標35%
                    "status": "good"
                    if current_metrics["profit_margin"] >= 0.3
                    else "fair",
                },
                "customer_satisfaction": {
                    "value": current_metrics["customer_satisfaction"],
                    "target": 4.5,  # 目標4.5/5.0
                    "status": "excellent"
                    if current_metrics["customer_satisfaction"] >= 4.5
                    else "acceptable",
                },
                "inventory_health": {
                    "total_slots": current_metrics["inventory_status"]["total_slots"],
                    "low_stock_count": current_metrics["inventory_status"][
                        "low_stock_count"
                    ],
                    "out_of_stock_count": current_metrics["inventory_status"][
                        "out_of_stock_count"
                    ],
                    "health_score": (
                        1
                        - (
                            current_metrics["inventory_status"]["low_stock_count"]
                            / max(current_metrics["inventory_status"]["total_slots"], 1)
                        )
                    )
                    * 100,
                    "status": "healthy"
                    if current_metrics["inventory_status"]["out_of_stock_count"] == 0
                    else "critical",
                },
            },
            "trends": performance_trends,
            "alerts": _generate_performance_alerts(current_metrics),
        }

        logger.info(
            f"Performance dashboard generated: {len(kpi_dashboard['alerts'])} alerts"
        )
        return kpi_dashboard

    except Exception as e:
        logger.error(f"パフォーマンス監視エラー: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "monitoring_failed",
        }


def _generate_performance_alerts(metrics) -> list:
    """パフォーマンスアラート生成"""
    alerts = []

    if metrics["sales"] < 500000:  # 月間目標の半分未満
        alerts.append(
            {
                "type": "sales_warning",
                "severity": "high",
                "message": f"売上実績が目標比{metrics['sales'] / 1000000:.1%}と低調",
                "action_required": "営業戦略の見直しを検討",
            }
        )

    if metrics["profit_margin"] < 0.25:  # 25%未満
        alerts.append(
            {
                "type": "profit_warning",
                "severity": "medium",
                "message": f"利益率が{metrics['profit_margin']:.1%}と目標を下回っています",
                "action_required": "コスト構造の見直し",
            }
        )

    if metrics["customer_satisfaction"] < 3.5:
        alerts.append(
            {
                "type": "satisfaction_warning",
                "severity": "high",
                "message": f"顧客満足度が{metrics['customer_satisfaction']:.1f}/5.0と低下",
                "action_required": "品質改善と顧客対応強化",
            }
        )

    if metrics["inventory_status"]["out_of_stock_count"] > 0:
        alerts.append(
            {
                "type": "inventory_critical",
                "severity": "critical",
                "message": f"品切れ商品が{metrics['inventory_status']['out_of_stock_count']}件発生",
                "action_required": "緊急在庫補充実施",
            }
        )

    return alerts
