"""
orchestrator.py - Analytics Agent オーケストレーター

Analytics Agent全分析機能を統合・ワークフロー管理
"""

import logging
from datetime import datetime, time
from typing import Any, Dict, List, Optional

from src.agents.analytics_agent.ai_governance.decision_quality_monitor import (
    monitor_decision_quality,
)
from src.agents.analytics_agent.ai_governance.performance_tracker import (
    track_ai_performance,
)
from src.agents.analytics_agent.ai_governance.safety_compliance_checker import (
    check_ai_safety_compliance,
)
from src.agents.analytics_agent.analysis.efficiency_analyzer import (
    analyze_business_efficiency,
)
from src.agents.analytics_agent.business_monitoring.anomaly_detector import (
    detect_business_anomalies,
)
from src.agents.analytics_agent.business_monitoring.compliance_checker import (
    check_compliance,
)
from src.agents.analytics_agent.business_monitoring.performance_monitor import (
    monitor_business_performance,
)

logger = logging.getLogger(__name__)


class AnalyticsAgentOrchestrator:
    """Analytics Agentオーケストレーター"""

    def __init__(self):
        """Analytics Agent初期化"""
        self.active_analysis_session: Optional[Dict[str, Any]] = None
        logger.info("AnalyticsAgentOrchestrator initialized")

    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """総合的なビジネス分析実行"""
        logger.info("Starting comprehensive business analysis")

        try:
            # Business Monitoring分析
            performance_analysis = monitor_business_performance()
            anomaly_analysis = detect_business_anomalies()
            compliance_analysis = check_compliance()

            # Efficiency Analysis
            efficiency_analysis = analyze_business_efficiency()

            # AI Governance分析
            ai_performance = await track_ai_performance()
            decision_quality = monitor_decision_quality()
            safety_compliance = check_ai_safety_compliance()

            # 総合結果統合
            comprehensive_result = self._integrate_analysis_results(
                {
                    "performance_monitoring": {
                        "performance": performance_analysis,
                        "anomalies": anomaly_analysis,
                        "compliance": compliance_analysis,
                    },
                    "business_efficiency": efficiency_analysis,
                    "ai_governance": {
                        "performance": ai_performance,
                        "decision_quality": decision_quality,
                        "safety": safety_compliance,
                    },
                }
            )

            # 重要アラート抽出
            critical_alerts = self._extract_critical_alerts(comprehensive_result)

            final_result = {
                "timestamp": datetime.now().isoformat(),
                "session_type": "comprehensive_analysis",
                "analysis_results": comprehensive_result,
                "critical_alerts": critical_alerts,
                "recommendations_summary": self._generate_executive_summary(
                    comprehensive_result
                ),
            }

            logger.info("Comprehensive analysis completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "analysis_failed",
                "error": str(e),
            }

    async def run_focused_analysis(self, focus_areas: List[str]) -> Dict[str, Any]:
        """特定分野に焦点を当てた分析実行"""
        logger.info(f"Running focused analysis on: {focus_areas}")

        focused_results = {}

        if "performance" in focus_areas:
            focused_results["performance"] = monitor_business_performance()
            focused_results["anomalies"] = detect_business_anomalies()
            focused_results["compliance"] = check_compliance()

        if "efficiency" in focus_areas:
            focused_results["efficiency"] = analyze_business_efficiency()

        if "ai_governance" in focus_areas:
            focused_results["ai_performance"] = await track_ai_performance()
            focused_results["decision_quality"] = monitor_decision_quality()
            focused_results["safety_compliance"] = check_ai_safety_compliance()

        return {
            "timestamp": datetime.now().isoformat(),
            "focus_areas": focus_areas,
            "results": focused_results,
        }

    async def run_real_time_monitoring(self) -> Dict[str, Any]:
        """リアルタイム監視実行"""
        logger.info("Running real-time monitoring cycle")

        # 軽量監視: 主要指標のみ
        performance_health = monitor_business_performance()
        safety_status = check_ai_safety_compliance()

        monitoring_result = {
            "timestamp": datetime.now().isoformat(),
            "performance_health": performance_health["current_kpis"],
            "safety_status": safety_status["safety_status"],
            "alerts": performance_health["alerts"]
            + safety_status.get("violations", []),
            "status": "monitoring_active",
        }

        # 異常検知が真の異常の場合は詳細分析をトリガー
        if performance_health["alerts"] or safety_status["violations_count"] > 0:
            logger.warning("Anomalies detected, triggering detailed analysis")
            # 必要に応じて詳細分析を呼び出し

        return monitoring_result

    def _integrate_analysis_results(
        self, analysis_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析結果の統合"""
        integrated_results = {}

        # Business Monitoring統合
        bm = analysis_components["performance_monitoring"]
        integrated_results["business_monitoring"] = {
            "overall_health_score": self._calculate_business_health_score(bm),
            "performance": bm["performance"],
            "anomalies": bm["anomalies"],
            "compliance": bm["compliance"],
        }

        # Efficiency統合
        efficiency = analysis_components["business_efficiency"]
        integrated_results["business_efficiency"] = {
            "efficiency_score": efficiency["overall_efficiency_score"],
            "analysis": efficiency["efficiency_analysis"],
            "improvement_opportunities": efficiency["improvement_opportunities"],
            "priority_actions": efficiency["priority_actions"],
        }

        # AI Governance統合
        governance = analysis_components["ai_governance"]
        integrated_results["ai_governance"] = {
            "ai_health_score": self._calculate_ai_health_score(governance),
            "performance": governance["performance"],
            "decision_quality": governance["decision_quality"],
            "safety": governance["safety"],
        }

        # クロスドメイン分析
        integrated_results["cross_domain_insights"] = (
            self._generate_cross_domain_insights(integrated_results)
        )

        return integrated_results

    def _calculate_business_health_score(
        self, business_monitoring: Dict[str, Any]
    ) -> float:
        """ビジネス健康スコア計算 (0-1スケール)"""
        performance_score = (
            business_monitoring["performance"]["current_kpis"]["inventory_health"][
                "health_score"
            ]
            / 100
        )

        anomaly_score = 1.0 - (
            business_monitoring["anomalies"]["anomalies_detected"] * 0.1
        )  # 異常1件あたり-10%

        compliance_score = business_monitoring["compliance"]["compliance_rate"] / 100

        return round((performance_score + anomaly_score + compliance_score) / 3, 3)

    def _calculate_ai_health_score(self, ai_governance: Dict[str, Any]) -> float:
        """AI健康スコア計算"""
        # 各拠点のスコアを統合
        scores = []

        if "overall_performance_score" in ai_governance["performance"]:
            scores.append(ai_governance["performance"]["overall_performance_score"])

        if "quality_score" in ai_governance["decision_quality"]:
            scores.append(ai_governance["decision_quality"]["quality_score"])

        compliance_rate = ai_governance["safety"]["compliance_rate"]
        if compliance_rate:
            scores.append(compliance_rate / 100)

        return round(sum(scores) / len(scores), 3) if scores else 0.5

    def _generate_cross_domain_insights(
        self, integrated_results: Dict[str, Any]
    ) -> List[str]:
        """クロスドメインインサイト生成"""
        insights = []

        # 効率性とパフォーマンスの相関分析
        efficiency_score = integrated_results["business_efficiency"]["efficiency_score"]
        health_score = integrated_results["business_monitoring"]["overall_health_score"]

        if efficiency_score > 0.8 and health_score < 0.7:
            insights.append(
                "高効率を実現しているものの、全体的なパフォーマンスが追いついていない"
            )
        elif efficiency_score < 0.7 and health_score > 0.8:
            insights.append("運用的に安定しているが、効率化の余地が大きい")

        # AI健康とビジネスパフォーマンスの関連
        ai_health = integrated_results["ai_governance"]["ai_health_score"]
        business_health = integrated_results["business_monitoring"][
            "overall_health_score"
        ]

        if ai_health < business_health:
            insights.append("AIシステムの健康状態がビジネス運用に比べて低い")
        elif ai_health > business_health:
            insights.append(
                "AIシステムがビジネスパフォーマンスを上回る高い信頼性を持っている"
            )

        return insights

    def _extract_critical_alerts(
        self, analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """重要アラート抽出"""
        critical_alerts = []

        # Business Monitoringからのクリティカルアラート
        bm_alerts = analysis_results["business_monitoring"]["performance"]["alerts"]
        for alert in bm_alerts:
            if alert["level"] == "red":
                critical_alerts.append(
                    {
                        "source": "business_performance",
                        "severity": "critical",
                        "alert": alert["message"],
                        "action_required": alert["action_required"],
                    }
                )

        # Efficiencyからの高優先アクション
        efficiency_actions = analysis_results["business_efficiency"]["priority_actions"]
        for action in efficiency_actions:
            if action["priority_level"] == "高":
                critical_alerts.append(
                    {
                        "source": "business_efficiency",
                        "severity": "high",
                        "alert": f"緊急改善が必要: {action['action']}",
                        "action_required": f"実装努力: {action['implementation_effort']}",
                    }
                )

        # AI Governanceからの重大violations
        safety_violations = analysis_results["ai_governance"]["safety"].get(
            "violations", []
        )
        for violation in safety_violations:
            if violation["severity"] == "critical":
                critical_alerts.append(
                    {
                        "source": "ai_governance",
                        "severity": "critical",
                        "alert": f"AI安全違反: {violation['description']}",
                        "action_required": violation["action_required"],
                    }
                )

        return critical_alerts

    def _generate_executive_summary(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """エグゼクティブサマリー生成"""
        business_health = analysis_results["business_monitoring"][
            "overall_health_score"
        ]
        efficiency = analysis_results["business_efficiency"]["efficiency_score"]
        ai_health = analysis_results["ai_governance"]["ai_health_score"]

        # 全体評価
        average_score = (business_health + efficiency + ai_health) / 3

        summary_level = (
            "excellent"
            if average_score > 0.85
            else "good"
            if average_score > 0.7
            else "fair"
            if average_score > 0.5
            else "needs_attention"
        )

        # キー指標
        key_metrics = {
            "business_health": round(business_health, 3),
            "operational_efficiency": round(efficiency, 3),
            "ai_system_health": round(ai_health, 3),
        }

        # 主要レコメンデーション
        recommendations = []

        if business_health < 0.8:
            recommendations.append("ビジネス運用改善を優先")

        if efficiency < 0.8:
            recommendations.append("業務効率化施策実施")

        if ai_health < 0.8:
            recommendations.append("AIシステム品質向上")

        return {
            "summary_level": summary_level,
            "overall_score": round(average_score, 3),
            "key_metrics": key_metrics,
            "top_recommendations": recommendations,
        }


# グローバルインスタンス
analytics_orchestrator = AnalyticsAgentOrchestrator()
