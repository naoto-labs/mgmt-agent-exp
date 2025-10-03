"""
safety_compliance_checker.py - AI安全性コンプライアンスチェックツール

AIガードレールの遵守状況を確認・安全基準逸脱を検知Tool
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from src.shared import settings

logger = logging.getLogger(__name__)


def check_ai_safety_compliance() -> Dict[str, Any]:
    """AI安全性のコンプライアンスチェック"""
    logger.info("Checking AI safety compliance")

    try:
        # ガードレール設定の読み取り
        safety_config = {
            "enable_guardrails": settings.enable_guardrails,
            "ai_safety_threshold": settings.ai_safety_threshold,
            "allowed_actions": settings.allowed_actions,
            "forbidden_patterns": settings.forbidden_patterns,
        }

        # 安全性チェック結果
        safety_violations = []
        warning_count = 0

        # ガードレール有効性チェック
        if not safety_config["enable_guardrails"]:
            warning_count += 1
            safety_violations.append(
                {
                    "type": "guardrail_disabled",
                    "severity": "high",
                    "description": "AIガードレールが無効化されています",
                    "action_required": "ガードレールを有効化してください",
                }
            )

        # 安全閾値チェック
        threshold_check = _check_safety_threshold(safety_config["ai_safety_threshold"])
        if not threshold_check["compliant"]:
            warning_count += 1
            safety_violations.append(threshold_check["violation"])

        # 許可アクション遵守チェック
        action_violations = _check_action_compliance(safety_config["allowed_actions"])
        safety_violations.extend(action_violations)

        # 禁止パターン検知チェック
        pattern_violations = _check_forbidden_patterns(
            safety_config["forbidden_patterns"]
        )
        safety_violations.extend(pattern_violations)

        # 緊急停止判定
        emergency_shutdown = _evaluate_emergency_shutdown(safety_violations)

        compliance_result = {
            "timestamp": datetime.now().isoformat(),
            "safety_status": "safe" if not safety_violations else "violations_detected",
            "violations_count": len(safety_violations),
            "violations": safety_violations,
            "emergency_shutdown_required": emergency_shutdown,
            "compliance_rate": (1 - (len(safety_violations) / 10)) * 100,  # 簡易計算
            "next_check_recommended": _calculate_next_safety_check(safety_violations),
        }

        logger.info(
            f"AI safety compliance check completed: {len(safety_violations)} violations"
        )
        if emergency_shutdown:
            logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")

        return compliance_result

    except Exception as e:
        logger.error(f"AI安全性チェックエラー: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "safety_check_failed",
            "emergency_shutdown_required": True,  # エラー時は安全側
        }


def _check_safety_threshold(threshold: float) -> Dict[str, Any]:
    """安全閾値の妥当性チェック"""
    # 一般的な安全閾値範囲: 0.7-0.95
    min_acceptable = 0.7
    max_acceptable = 0.95

    if threshold < min_acceptable:
        return {
            "compliant": False,
            "violation": {
                "type": "invalid_safety_threshold",
                "severity": "high",
                "description": f"安全閾値が低すぎます (現在: {threshold}, 推奨最小: {min_acceptable})",
                "action_required": "安全閾値を適切に設定してください",
            },
        }
    elif threshold > max_acceptable:
        return {
            "compliant": False,
            "violation": {
                "type": "invalid_safety_threshold",
                "severity": "medium",
                "description": f"安全閾値が高すぎます (現在: {threshold}, 推奨最大: {max_acceptable})",
                "action_required": "安全閾値を現実的に調整してください",
            },
        }

    return {"compliant": True}


def _check_action_compliance(allowed_actions: List[str]) -> List[Dict[str, Any]]:
    """許可アクション遵守チェック"""
    violations = []

    # 必須許可アクションのチェック
    required_actions = ["select_product", "process_payment", "dispense_product"]
    missing_actions = []

    for action in required_actions:
        if action not in allowed_actions:
            missing_actions.append(action)

    if missing_actions:
        violations.append(
            {
                "type": "missing_required_actions",
                "severity": "high",
                "description": f"必須許可アクションが欠落: {', '.join(missing_actions)}",
                "action_required": "必須アクションを許可リストに追加",
            }
        )

    # 危険なアクションが含まれていないかチェック
    dangerous_actions = ["unlimited_dispense", "override_payment", "access_admin"]
    found_dangerous = []

    for action in dangerous_actions:
        if action in allowed_actions:
            found_dangerous.append(action)

    if found_dangerous:
        violations.append(
            {
                "type": "dangerous_actions_allowed",
                "severity": "critical",
                "description": f"危険な許可アクション検知: {', '.join(found_dangerous)}",
                "action_required": "即時危険アクションを禁止リスト移動",
            }
        )

    return violations


def _check_forbidden_patterns(forbidden_patterns: List[str]) -> List[Dict[str, Any]]:
    """禁止パターンの妥当性チェック"""
    violations = []

    # 必須禁止パターンのチェック
    required_patterns = ["override_safety", "bypass_payment", "unlimited_access"]
    missing_patterns = []

    for pattern in required_patterns:
        if pattern not in forbidden_patterns:
            missing_patterns.append(pattern)

    if missing_patterns:
        violations.append(
            {
                "type": "missing_forbidden_patterns",
                "severity": "high",
                "description": f"必須禁止パターンが欠落: {', '.join(missing_patterns)}",
                "action_required": "必須パターンを禁止リストに追加",
            }
        )

    # 禁止パターンのテスト実行 (簡易)
    test_inputs = [
        "override_safety now",
        "bypass_payment completely",
        "unlimited_access granted",
        "select_product normally",  # 許可されるべき
    ]

    false_positives = []
    for test_input in test_inputs:
        blocked_incorrectly = any(
            re.search(pattern, test_input) for pattern in forbidden_patterns
        )
        if blocked_incorrectly and test_input == "select_product normally":
            false_positives.append(test_input)

    if false_positives:
        violations.append(
            {
                "type": "false_positive_detected",
                "severity": "medium",
                "description": f"正常指令を誤検知: {', '.join(false_positives)}",
                "action_required": "禁止パターンを精密化",
            }
        )

    return violations


def _evaluate_emergency_shutdown(violations: List[Dict[str, Any]]) -> bool:
    """緊急停止判定"""
    critical_severities = [v for v in violations if v["severity"] == "critical"]

    # 重要な基準:
    # 1. ガードレール無効化
    # 2. 安全閾値の重大逸脱
    # 3. 危険アクションの許可

    for violation in violations:
        violation_type = violation["type"]
        if violation_type in ["guardrail_disabled", "dangerous_actions_allowed"]:
            return True

    # 複数のクリティカル違反
    if len(critical_severities) >= 2:
        return True

    return False


def _calculate_next_safety_check(violations: List[Dict[str, Any]]) -> str:
    """次回安全性チェック推奨日"""
    if any(v["severity"] == "critical" for v in violations):
        return "即時再チェックを推奨"
    elif any(v["severity"] == "high" for v in violations):
        return "1時間以内再チェック"
    elif violations:
        return "24時間以内再チェック"
    else:
        return "次回定期チェックまで維持"
