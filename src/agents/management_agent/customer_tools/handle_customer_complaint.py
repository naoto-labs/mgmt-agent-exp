"""
handle_customer_complaint.py - 顧客苦情処理ツール

クレーム内容解決策を提案、補償措置を実施Tool
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def handle_customer_complaint(customer_id: str, complaint: str) -> Dict[str, Any]:
    """顧客苦情をLLMで分析・解決策提案・補償措置決定

    Args:
        customer_id: 顧客ID
        complaint: 苦情内容

    Returns:
        Dict: 顧客ID、苦情内容、LLM分析結果、解決案、補償措置、対応ステータスを含む結果データ
    """
    logger.info(
        f"LLMベースで顧客 {customer_id} の苦情を分析・解決: {complaint[:50]}..."
    )

    try:
        # 管理エージェントからのLLMマネージャーをインポート
        from src.agents.management_agent import management_agent

        # 苦情内容を分析し、解決策と補償を決定するプロンプト
        complaint_analysis_prompt = f"""
あなたは自動販売機事業のクレーム対応担当者です。以下の苦情内容を分析し、適切な解決策と補償措置を決定してください。

【苦情内容】
{complaint}

【分析事項】
1. 苦情の深刻度評価 (critical/high/medium/low)
2. 原因特定 (商品品質/機械故障/サービス対応/価格/その他)
3. 顧客の感情・影響度 (非常に不満/不満/軽い不満)
4. 事業への影響評価 (顧客離反リスク/口コミ悪影響/即時対応必要度)
5. 推奨解決期間 (即時/1日以内/数日以内/継続フォロー)

【解決方針指針】
- 深刻な苦情には積極的な補償で救済
- 顧客満足度向上のための再発防止策提案
- 事業イメージ回復のための丁寧な対応
- 苦情を改善機会として活用

【自動販売機事業の補償オプション】
- 返金: 同一金額または2倍額
- クーポン券: 500円/1000円/2000円分
- 無料商品: 次の購入時
- 現金補償: 小額対応
- サービス改善: 優先サポート
- 謝罪状: メールまたは手紙

【出力形式】
JSON形式で以下の構造で回答してください：
{{
    "complaint_analysis": {{
        "severity": "critical/high/medium/low",
        "root_cause": "原因特定",
        "customer_impact": "顧客への影響度",
        "business_impact": "事業への影響度",
        "resolution_timeline": "解決目標期間"
    }},
    "resolution_strategy": {{
        "immediate_actions": ["即時実行アクション"],
        "resolution_approach": "解決アプローチの方針",
        "prevention_measures": ["再発防止対策"],
        "follow_up_plan": "フォローアップ計画"
    }},
    "customer_satisfaction": {{
        "compassion_level": "共感表明のレベル (1-5)",
        "solution_explanation": "解決策の明確な説明",
        "customer_response": "顧客向け最終回答文（共感・解決策・謝罪・対応方法含む）"
    }},
    "compensation": {{
        "compensation_type": "補償の種類",
        "compensation_amount": "補償額または内容",
        "additional_benefits": ["追加的なサービスや対応"],
        "compensation_reason": "補償決定の根拠"
    }},
    "escalation_required": "true/false",
    "stakeholders_to_involve": ["必要に応じて関与する関係者"],
    "next_steps": ["後続対応項目"]
}}
"""

        # LLMに苦情分析を依頼
        messages = [
            management_agent.llm_manager.create_ai_message(
                role="system",
                content="あなたは自動販売機事業のクレーム対応専門家です。顧客の苦情を真摯に受け止め、適切な解決策を提案してください。",
            ),
            management_agent.llm_manager.create_ai_message(
                role="user", content=complaint_analysis_prompt
            ),
        ]

        response = await management_agent.llm_manager.generate_response(
            messages, max_tokens=1200
        )

        if response.success:
            # LLMレスポンスをJSONとしてパース
            import json

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            analysis_result = json.loads(content)

            # デフォルト値の設定
            complaint_analysis = analysis_result.get("complaint_analysis", {})
            complaint_analysis.setdefault("severity", "medium")
            complaint_analysis.setdefault("root_cause", "to_be_investigated")
            complaint_analysis.setdefault("customer_impact", "moderate")
            complaint_analysis.setdefault("business_impact", "low")
            complaint_analysis.setdefault("resolution_timeline", "within_24_hours")

            resolution_strategy = analysis_result.get("resolution_strategy", {})
            resolution_strategy.setdefault("immediate_actions", [])
            resolution_strategy.setdefault(
                "resolution_approach", "apologize_and_resolve"
            )
            resolution_strategy.setdefault("prevention_measures", [])
            resolution_strategy.setdefault("follow_up_plan", "follow_up_in_24h")

            customer_satisfaction = analysis_result.get("customer_satisfaction", {})
            customer_satisfaction.setdefault("compassion_level", "3")
            customer_satisfaction.setdefault(
                "solution_explanation", "問題を解決いたします"
            )
            customer_satisfaction.setdefault(
                "customer_response",
                "お手数をおかけし申し訳ございません。対応いたします。",
            )

            compensation = analysis_result.get("compensation", {})
            compensation.setdefault("compensation_type", "coupon")
            compensation.setdefault("compensation_amount", "500")
            compensation.setdefault("additional_benefits", [])
            compensation.setdefault("compensation_reason", "customer_satisfaction")

            escalation_required = analysis_result.get("escalation_required", False)
            stakeholders_to_involve = analysis_result.get("stakeholders_to_involve", [])
            next_steps = analysis_result.get("next_steps", [])

            logger.info(
                f"LLM苦情分析完了: 深刻度={complaint_analysis['severity']}, 補償={compensation['compensation_type']}"
            )

            return {
                "customer_id": customer_id,
                "complaint": complaint,
                "complaint_analysis": complaint_analysis,
                "resolution_strategy": resolution_strategy,
                "customer_response": customer_satisfaction["customer_response"],
                "compensation": compensation,
                "escalation_required": escalation_required,
                "stakeholders_to_involve": stakeholders_to_involve,
                "next_steps": next_steps,
                "status": "analyzed_and_resolved",
                "llm_used": True,
                "resolution_confidence": "high",
            }

        else:
            logger.warning(f"LLM苦情分析失敗: {response.error_message}")
            # LLM失敗時のフォールバック
            return {
                "customer_id": customer_id,
                "complaint": complaint,
                "resolution": "商品の返金処理を行い、次回使用可能なクーポンを発行しました。",
                "compensation": "500円クーポン",
                "status": "resolved_with_fallback",
                "llm_used": False,
                "next_steps": ["手動補償処理"],
            }

    except Exception as e:
        logger.error(f"顧客苦情LLM分析エラー: {e}")
        # 完全フォールバック
        return {
            "customer_id": customer_id,
            "complaint": complaint,
            "resolution": "お手数をおかけし大変申し訳ございません。現在確認中のため、24時間以内に折り返しご連絡いたします。",
            "compensation": "要確認",
            "status": "escalated_to_manual",
            "llm_used": False,
            "next_steps": ["即時エスカレーション", "管理者対応"],
        }
