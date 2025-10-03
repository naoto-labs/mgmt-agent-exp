"""
anomaly_detector.py - 異常検知ツール

売上急変・在庫異常・顧客苦情急増などの異常を統計的手法で検出Tool
"""

import logging
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any, Dict, List

from src.agents.management_agent.management_tools.get_business_metrics import (
    get_business_metrics,
)

logger = logging.getLogger(__name__)


def detect_business_anomalies() -> Dict[str, Any]:
    """ビジネス異常検知"""
    logger.info("Detecting business anomalies using statistical analysis")

    try:
        # 現在のメトリクス取得
        current_metrics = get_business_metrics()

        # 異常検知の結果
        anomalies = []

        # 売上異常検知 (簡易標準偏差検知)
        # 本来は複数日分のデータが必要だが、シミュレーション
        sales_anomaly = _detect_sales_anomaly(current_metrics["sales"])
        if sales_anomaly:
            anomalies.append(sales_anomaly)

        # 在庫異常検知
        inventory_anomaly = _detect_inventory_anomaly(
            current_metrics["inventory_level"]
        )
        if inventory_anomaly:
            anomalies.append(inventory_anomaly)

        # 顧客満足度異常検知
        satisfaction_anomaly = _detect_satisfaction_anomaly(
            current_metrics["customer_satisfaction"]
        )
        if satisfaction_anomaly:
            anomalies.append(satisfaction_anomaly)

        detection_result = {
            "timestamp": datetime.now().isoformat(),
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "status": "anomalies_found" if anomalies else "normal",
            "recommendations": _generate_anomaly_recommendations(anomalies),
        }

        logger.info(f"Anomaly detection completed: {len(anomalies)} anomalies found")
        return detection_result

    except Exception as e:
        logger.error(f"異常検知エラー: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "detection_failed",
        }


def _detect_sales_anomaly(current_sales: float) -> Dict[str, Any]:
    """売上異常検知"""
    # 簡易的な閾値ベース検知
    # 本来は過去データとの比較
    threshold_low = 80000  # 月間基準値 (シミュレーション)
    threshold_high = 150000  # 月間基準値 (シミュレーション)

    if current_sales < threshold_low:
        return {
            "type": "sales_drop_anomaly",
            "severity": "high",
            "value": current_sales,
            "threshold": threshold_low,
            "deviation": (threshold_low - current_sales) / threshold_low * 100,
            "description": f"売上実績が基準値({threshold_low:,}円)を下回っています",
            "impact": "revenue_risk",
        }
    elif current_sales > threshold_high:
        return {
            "type": "sales_spike_anomaly",
            "severity": "medium",
            "value": current_sales,
            "threshold": threshold_high,
            "deviation": (current_sales - threshold_high) / threshold_high * 100,
            "description": f"売上が基準値({threshold_high:,}円)を大幅に上回っています",
            "impact": "positive_growth",
        }

    return None


def _detect_inventory_anomaly(inventory_level: Dict[str, int]) -> Dict[str, Any]:
    """在庫異常検知"""
    total_inventory = sum(inventory_level.values())

    # 在庫総量が異常な場合
    if total_inventory < 30:
        return {
            "type": "inventory_depletion_anomaly",
            "severity": "critical",
            "value": total_inventory,
            "threshold": 30,
            "description": "全在庫量が極端に低くなっています",
            "impact": "supply_chain_risk",
            "details": inventory_level,
        }

    # 特定商品の在庫異常
    for product, stock in inventory_level.items():
        if stock == 0:
            return {
                "type": "out_of_stock_anomaly",
                "severity": "high",
                "product": product,
                "value": stock,
                "description": f"{product}が品切れ状態です",
                "impact": "customer_impact",
            }

    return None


def _detect_satisfaction_anomaly(current_satisfaction: float) -> Dict[str, Any]:
    """顧客満足度異常検知"""
    threshold = 3.0  # 最低許容値

    if current_satisfaction < threshold:
        return {
            "type": "satisfaction_drop_anomaly",
            "severity": "high",
            "value": current_satisfaction,
            "threshold": threshold,
            "deviation": (threshold - current_satisfaction) * 20,  # /5.0スケール
            "description": f"顧客満足度が{current_satisfaction:.1f}/5.0で基準値を下回っています",
            "impact": "customer_retention_risk",
        }

    return None


def _generate_anomaly_recommendations(anomalies: List[Dict]) -> List[str]:
    """異常に対する推奨事項生成"""
    recommendations = []

    anomaly_types = {anomaly["type"] for anomaly in anomalies}

    if "sales_drop_anomaly" in anomaly_types:
        recommendations.extend(
            [
                "売上向上策の検討（プロモーション、価格戦略の見直し）",
                "市場動向の分析と競合比較",
            ]
        )

    if "inventory_depletion_anomaly" in anomaly_types:
        recommendations.extend(
            [
                "緊急入荷ルートの確保",
                "在庫管理プロセスの見直し",
            ]
        )

    if "out_of_stock_anomaly" in anomaly_types:
        recommendations.extend(
            [
                "代替商品の提案強化",
                "需給予測精度の向上",
            ]
        )

    if "satisfaction_drop_anomaly" in anomaly_types:
        recommendations.extend(
            [
                "顧客フィードバックの詳細分析",
                "品質改善プロジェクトの実施",
            ]
        )

    if not recommendations:
        recommendations.append("現在のビジネス状態は安定しています")

    return recommendations
