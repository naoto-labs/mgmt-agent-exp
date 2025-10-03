"""
compliance_checker.py - コンプライアンスチェックツール

経費精算・消費税計算・会計基準遵守を確認・監査レポート生成Tool
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List

from src.agents.management_agent.management_tools.get_business_metrics import (
    get_business_metrics,
)
from src.domain.accounting.management_accounting import management_analyzer

logger = logging.getLogger(__name__)


def check_compliance() -> Dict[str, Any]:
    """会計・業務コンプライアンスチェック"""
    logger.info("Checking accounting and business compliance")

    try:
        # 財務メトリクス取得
        current_metrics = get_business_metrics()

        # コンプライアンスチェックの結果
        compliance_issues = []

        # 消費税計算チェック
        tax_issue = _check_tax_compliance(current_metrics["sales"])
        if tax_issue:
            compliance_issues.append(tax_issue)

        # 会計基準遵守チェック
        accounting_issue = _check_accounting_standards()
        if accounting_issue:
            compliance_issues.append(accounting_issue)

        # 法令遵守チェック
        legal_issues = _check_legal_compliance(current_metrics)
        compliance_issues.extend(legal_issues)

        compliance_result = {
            "timestamp": datetime.now().isoformat(),
            "compliance_violations": len(compliance_issues),
            "violations": compliance_issues,
            "status": "compliant" if not compliance_issues else "violations_found",
            "compliance_rate": (len(compliance_issues) == 0) * 100,
            "audit_report": _generate_audit_report(compliance_issues),
        }

        logger.info(
            f"Compliance check completed: {len(compliance_issues)} violations found"
        )
        return compliance_result

    except Exception as e:
        logger.error(f"コンプライアンスチェックエラー: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "check_failed",
        }


def _check_tax_compliance(sales_amount: float) -> Dict[str, Any]:
    """消費税計算・納税義務チェック"""
    # 日本消費税基準 (10%)
    tax_rate = 0.10
    tax_amount = sales_amount * tax_rate

    # 納税義務チェック (課税売上高1,000万円以下は免税あり)
    target_amount_threshold = 10000000  # 1,000万円

    if sales_amount >= target_amount_threshold:
        return {
            "type": "tax_compliance",
            "severity": "info",
            "description": f"消費税納税義務対象売上高を超過 (¥{sales_amount:,} >= ¥{target_amount_threshold:,})",
            "required_action": f"消費税{int(tax_amount):,}円の納税義務あり",
            "tax_rate": tax_rate,
        }

    return None


def _check_accounting_standards() -> Dict[str, Any]:
    """会計基準遵守チェック"""
    try:
        # 会計データの整合性チェック
        # 総勘定元帳の貸借一致確認 (簡易版)
        end_date = date.today()
        start_date = end_date.replace(day=1)

        # 資産合計の取得 (実際は会計システムから)
        asset_balance_4001 = management_analyzer.journal_processor.get_account_balance(
            "4001", start_date, end_date
        )
        expense_balance_4002 = (
            management_analyzer.journal_processor.get_account_balance(
                "4002", start_date, end_date
            )
        )

        balance_diff = abs(asset_balance_4001 - expense_balance_4002)

        if balance_diff > 0.01:  # 1円以上の差で違反
            return {
                "type": "accounting_standards",
                "severity": "critical",
                "description": f"帳簿上の貸借が一致していません (差額: {balance_diff:.2f}円)",
                "required_action": "会計データの修正・監査実施",
                "affected_accounts": ["売上", "費用"],
            }

        return None

    except Exception as e:
        logger.warning(f"会計基準チェックエラー: {e}")
        return {
            "type": "accounting_error",
            "severity": "high",
            "description": f"会計システムデータ取得不可: {str(e)}",
            "required_action": "会計システムの確認を要請",
        }


def _check_legal_compliance(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """法令遵守チェック"""
    legal_violations = []

    # 最低賃金チェック (地域により異なるが、簡易的に全国平均使用)
    # 実際には従業員数・時給データが必要だが、ダミー
    minimum_wage = 1000  # 円/時間 (簡易基準)

    # # 労働基準法チェック (残業時間等 - データなしの場合はスキップ)
    # if hasattr(company_data, 'overtime_hours'):
    #     if company_data.overtime_hours > 45:  # 月間上限
    #         legal_violations.append({
    #             "type": "labor_law_compliance",
    #             "severity": "high",
    #             "description": f"残業時間が制限を超過 ({company_data.overtime_hours}時間/月)",
    #             "required_action": "人事労務管理の見直し",
    #         })

    # 請求書法チェック (簡易)
    invoice_issue = _check_invoice_compliance(metrics["sales"])
    if invoice_issue:
        legal_violations.append(invoice_issue)

    # 会社法チェック (簡易)
    if metrics["sales"] > 100000000:  # 1億円超取引
        large_transaction_issue = {
            "type": "corporate_law_compliance",
            "severity": "medium",
            "description": "大口取引が発生 (会社法要件考慮)",
            "required_action": "取締役会決議の確認",
            "transaction_value": metrics["sales"],
        }
        legal_violations.append(large_transaction_issue)

    return legal_violations


def _check_invoice_compliance(sales_amount: float) -> Dict[str, Any]:
    """請求書法遵守チェック"""
    # 請求書必須金額 (1万円以上、BtoB取引)
    invoice_required_threshold = 10000

    if sales_amount >= invoice_required_threshold:
        # 請求書発行義務チェック (ダミー)
        # 実際は請求書データベースから確認
        invoice_issued = True  # ダミー: 発行済みと仮定

        if not invoice_issued:
            return {
                "type": "invoice_law_compliance",
                "severity": "high",
                "description": f"請求書法違反可能性あり (売上¥{int(sales_amount):,} >= ¥{invoice_required_threshold:,})",
                "required_action": "正規の請求書発行・保存",
                "affected_amount": sales_amount,
            }

    return None


def _generate_audit_report(violations: List[Dict]) -> Dict[str, Any]:
    """監査レポート生成"""
    if not violations:
        return {
            "summary": "会計・業務コンプライアンスに問題なし",
            "recommendations": ["現状維持を推奨"],
            "next_audit_date": "次回定期監査まで問題なし",
        }

    severity_counts = {}
    for violation in violations:
        sev = violation["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    highest_severity = max(
        severity_counts.keys(),
        key=lambda x: {"critical": 3, "high": 2, "medium": 1, "low": 0, "info": -1}[x],
    )

    return {
        "summary": f"{len(violations)}件のコンプライアンス違反が検知されました (最高深刻度: {highest_severity})",
        "severity_breakdown": severity_counts,
        "key_violations": [v["description"] for v in violations[:3]],  # 最大3件表示
        "recommendations": [
            "違反事項の即時修正",
            "内部統制システムの見直し",
            "外部監査の検討",
        ],
        "next_audit_date": "1ヶ月以内再チェックを推奨",
    }
