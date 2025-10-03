"""
plan_agent_operations.py - Agent運営計画作成ツール

日次・週次業務計画の立案・実行手順策定Tool
"""

import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, List

from src.agents.management_agent.management_tools.get_business_metrics import (
    get_business_metrics,
)

logger = logging.getLogger(__name__)


def plan_agent_operations(plan_period: str = "daily") -> Dict[str, Any]:
    """AI Agent自身の運営計画を立案 (日次/週次業務計画)"""
    logger.info(f"Planning agent operations for {plan_period} period")

    try:
        current_time = datetime.now()

        if plan_period == "daily":
            plan = _create_daily_operations_plan(current_time)
        elif plan_period == "weekly":
            plan = _create_weekly_operations_plan(current_time)
        else:
            raise ValueError(f"Unsupported plan period: {plan_period}")

        return {
            "timestamp": current_time.isoformat(),
            "plan_period": plan_period,
            "plan": plan,
            "recommendations": _generate_plan_recommendations(plan),
        }

    except Exception as e:
        logger.error(f"Agent運営計画作成エラー: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "planning_failed",
        }


def _create_daily_operations_plan(current_time: datetime) -> Dict[str, Any]:
    """日次運営計画を作成"""
    current_hour = current_time.hour

    # 基本業務テンプレート
    base_activities = {
        "morning": [
            {
                "type": "health_check",
                "description": "システム健康状態確認",
                "priority": "high",
                "duration": 15,
            },
            {
                "type": "metric_review",
                "description": "朝のKPI確認",
                "priority": "high",
                "duration": 20,
            },
            {
                "type": "plan_review",
                "description": "日次計画立案",
                "priority": "medium",
                "duration": 30,
            },
        ],
        "midday": [
            {
                "type": "performance_check",
                "description": "午前実績確認",
                "priority": "high",
                "duration": 15,
            },
            {
                "type": "adjustment_decision",
                "description": "午後調整判断",
                "priority": "high",
                "duration": 25,
            },
            {
                "type": "maintenance_check",
                "description": "定期メンテナンス",
                "priority": "low",
                "duration": 10,
            },
        ],
        "evening": [
            {
                "type": "daily_summary",
                "description": "1日実績まとめ",
                "priority": "high",
                "duration": 30,
            },
            {
                "type": "improvement_plan",
                "description": "改善策立案",
                "priority": "medium",
                "duration": 20,
            },
            {
                "type": "next_day_prep",
                "description": "翌日準備",
                "priority": "medium",
                "duration": 15,
            },
        ],
        "night": [
            {
                "type": "off_peak_monitoring",
                "description": "深夜監視",
                "priority": "low",
                "duration": 10,
            },
            {
                "type": "data_backup",
                "description": "バックアップ処理",
                "priority": "low",
                "duration": 5,
            },
        ],
    }

    # 現在の時間枠に応じた活動選択
    if 6 <= current_hour < 12:
        current_phase = "morning"
        remaining_hours = 12 - current_hour
    elif 12 <= current_hour < 18:
        current_phase = "midday"
        remaining_hours = 18 - current_hour
    elif 18 <= current_hour < 24:
        current_phase = "evening"
        remaining_hours = 24 - current_hour
    else:
        current_phase = "night"
        remaining_hours = 6 - current_hour

    # 優先度・リソース状況を考慮した計画立案
    prioritized_activities = _prioritize_daily_activities(
        base_activities[current_phase], remaining_hours
    )

    # 実行タイムライン作成
    execution_timeline = _create_execution_timeline(
        prioritized_activities, current_time, remaining_hours
    )

    return {
        "phase": current_phase,
        "remaining_hours": remaining_hours,
        "prioritized_activities": prioritized_activities,
        "execution_timeline": execution_timeline,
        "resource_status": _assess_resource_status(),
        "risk_assessment": _assess_planning_risks(execution_timeline),
    }


def _create_weekly_operations_plan(current_time: datetime) -> Dict[str, Any]:
    """週次運営計画を作成"""
    # 週次パターン分析と計画
    business_metrics = get_business_metrics()

    # 売上・利用パターンを週次トレンドに変換
    weekly_patterns = {
        "busy_days": ["monday", "friday"],  # 繁忙日
        "maintenance_days": ["sunday"],  # メンテナンス集中日
        "analysis_days": ["saturday"],  # 分析・計画日
    }

    # リソース配分計画
    resource_allocation = {
        "daily_monitoring": {"allocation": 0.4, "description": "日常監視業務"},
        "strategic_planning": {"allocation": 0.3, "description": "戦略的計画立案"},
        "system_maintenance": {
            "allocation": 0.2,
            "description": "システムメンテナンス",
        },
        "emergency_response": {"allocation": 0.1, "description": "緊急対応準備"},
    }

    # 週次目標設定
    weekly_goals = {
        "performance_targets": {
            "response_time_target": "2秒以内維持",
            "decision_accuracy_target": "85%以上",
        },
        "system_targets": {
            "uptime_target": "99.5%以上",
            "error_rate_target": "2%以内",
        },
        "learning_targets": {
            "pattern_learning_target": "5件/週",
            "improvement_implementation": "2件/週",
        },
    }

    return {
        "start_date": current_time.date(),
        "end_date": (current_time + timedelta(days=7)).date(),
        "weekly_patterns": weekly_patterns,
        "resource_allocation": resource_allocation,
        "weekly_goals": weekly_goals,
        "special_operations": _identify_special_weekly_operations(business_metrics),
    }


def _prioritize_daily_activities(
    activities: List[Dict], remaining_hours: float
) -> List[Dict]:
    """日次活動の優先順位付け"""
    # 優先度重み付け
    priority_weights = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
    }

    # 緊急度評価
    current_metrics = get_business_metrics()

    emergency_score = 0
    if current_metrics["inventory_status"]["out_of_stock_count"] > 0:
        emergency_score += 2  # 品切れ緊急度
    if current_metrics["sales"] < 50000:  # 売上低調
        emergency_score += 1
    if current_metrics["customer_satisfaction"] < 3.5:  # 満足度低下
        emergency_score += 1

    # 優先度再計算
    for activity in activities:
        base_priority = priority_weights.get(activity["priority"], 1.0)
        # 緊急時調整
        if emergency_score > 0 and activity["type"] in [
            "health_check",
            "performance_check",
        ]:
            base_priority += emergency_score * 0.5

        activity["calculated_priority"] = base_priority

    # 時間制約内でソート
    sorted_activities = sorted(
        activities, key=lambda x: x["calculated_priority"], reverse=True
    )

    # 実行可能活動のみ選択
    total_time = 0
    selectable_activities = []

    for activity in sorted_activities:
        if total_time + activity["duration"] <= remaining_hours * 60:  # 分変換
            selectable_activities.append(activity)
            total_time += activity["duration"]

    return selectable_activities


def _create_execution_timeline(
    activities: List[Dict[str, Any]], current_time: datetime, remaining_hours: float
) -> List[Dict[str, Any]]:
    """実行タイムライン作成"""
    timeline = []
    current_slot = current_time

    for activity in activities:
        start_time = current_slot
        duration_minutes = activity["duration"]
        end_time = start_time + timedelta(minutes=duration_minutes)

        timeline_entry = {
            "activity": activity["description"],
            "type": activity["type"],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "priority": activity["priority"],
        }

        timeline.append(timeline_entry)
        current_slot = end_time + timedelta(minutes=5)  # 5分休憩

    return timeline


def _assess_resource_status() -> Dict[str, Any]:
    """資源状況評価"""
    # システム資源の現在の状況 (簡易評価)
    return {
        "cpu_status": "normal",  # 実際はシステム監視
        "memory_status": "adequate",
        "network_status": "stable",
        "data_access_status": "available",
        "ai_model_status": "healthy",
        "estimated_capacity": "80%",  # 使用可能な処理能力
    }


def _assess_planning_risks(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """計画実行リスク評価"""
    risk_factors = []

    # 時間制約リスク
    total_planned_time = sum(activity["duration_minutes"] for activity in timeline)
    available_minutes = 4 * 60  # 仮定4時間

    if total_planned_time > available_minutes:
        risk_factors.append(
            {
                "type": "time_constraint",
                "level": "high",
                "description": f"計画時間({total_planned_time}分)が利用可能時間({available_minutes}分)を超過",
            }
        )

    # 依存関係リスク
    dependent_activities = []
    for activity in timeline:
        if activity["type"] in ["metric_review", "adjustment_decision"]:
            dependent_activities.append(activity["activity"])

    if (
        dependent_activities
        and timeline.index(timeline[-1]) != len(dependent_activities) - 1
    ):
        risk_factors.append(
            {
                "type": "dependency_risk",
                "level": "medium",
                "description": "重要な活動の実行順序に依存関係あり",
            }
        )

    return {
        "overall_risk_level": "high"
        if any(r["level"] == "high" for r in risk_factors)
        else "low",
        "risk_factors": risk_factors,
        "mitigation_suggestions": _suggest_risk_mitigations(risk_factors),
    }


def _generate_plan_recommendations(plan: Dict[str, Any]) -> List[str]:
    """計画に関する全般的推奨事項"""
    recommendations = []

    # リソース状況に応じた推奨
    resource_status = plan.get("resource_status", {})
    if resource_status.get("estimated_capacity", "100%") < "70%":
        recommendations.append("処理能力に余裕がないため、複雑な分析を延期考慮")

    # リスク状況に応じた推奨
    risk_assessment = plan.get("risk_assessment", {})
    if risk_assessment.get("overall_risk_level") == "high":
        recommendations.append("高リスク計画のため、バックアップ計画の準備を推奨")

    # 活動実行の推奨
    activities = plan.get("prioritized_activities", [])
    if len(activities) == 0:
        recommendations.append("実行可能な活動がないため、資源状況の見直し要")
    elif len(activities) < 3:
        recommendations.append("計画実施数が少ないため、業務効率化の検討を推奨")

    # デフォルト推奨
    if not recommendations:
        recommendations.append("計画実行可能、安定した運用が見込まれます")

    return recommendations


def _identify_special_weekly_operations(
    metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """週次特別運営業務特定"""
    special_ops = []

    # 繁忙期対応
    if metrics["sales"] > 100000:  # 月間売上基準
        special_ops.append(
            {
                "operation": "period_peak_support",
                "days": ["mon", "tue", "wed", "thu", "fri"],
                "description": "売上好調期の追加リソース配分",
                "resource_allocation": "enhanced_monitoring",
            }
        )

    # 品質監査週間
    if metrics["customer_satisfaction"] < 4.0:
        special_ops.append(
            {
                "operation": "quality_focus_week",
                "description": "顧客満足度向上のための品質監査強化週",
                "focus_areas": ["response_quality", "decision_accuracy"],
            }
        )

    return special_ops


def _suggest_risk_mitigations(risk_factors: List[Dict[str, Any]]) -> List[str]:
    """リスク軽減策提案"""
    mitigation_suggestions = []

    for risk in risk_factors:
        if risk["type"] == "time_constraint":
            mitigation_suggestions.append("活動の優先度再設定または時間配分の見直し")
            mitigation_suggestions.append("自動化可能な業務を優先的に実行")

        elif risk["type"] == "dependency_risk":
            mitigation_suggestions.append("依存関係の深い活動をグループ化して連続実行")
            mitigation_suggestions.append("中間成果物の確認ポイントを設ける")

    if not mitigation_suggestions:
        mitigation_suggestions.append("リスク要因なし、標準運用継続")

    return mitigation_suggestions
