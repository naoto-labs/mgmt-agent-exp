"""
respond_to_customer_inquiry.py - 顧客問い合わせ対応ツール

顧客問い合わせ内容を分析、適切な回答を自動生成Tool
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def respond_to_customer_inquiry(customer_id: str, inquiry: str) -> Dict[str, Any]:
    """顧客問い合わせをLLMで分析・適切な回答を生成

    Args:
        customer_id: 顧客ID
        inquiry: 問い合わせ内容

    Returns:
        Dict: 顧客ID、問い合わせ内容、LLM生成回答、対応ステータスを含む回答データ
    """
    logger.info(f"LLMベースで顧客 {customer_id} の問い合わせに対応: {inquiry[:50]}...")

    try:
        # 管理エージェントからのLLMマネージャーをインポート
        from src.agents.management_agent import management_agent

        # 問い合わせ内容を分析し、適切な回答を生成するプロンプト
        inquiry_analysis_prompt = f"""
あなたは自動販売機事業の顧客サポート担当者です。以下の問い合わせ内容を分析し、適切な回答を生成してください。

【問い合わせ内容】
{inquiry}

【分析事項】
1. 問い合わせの種類を分類 (製品情報/営業時間/商品価格/苦情/提案/その他)
2. 緊急度評価 (緊急/通常/要確認)
3. 顧客の感情状態 (満足/不満/疑問/提案)
4. 必要となる対応内容

【回答指針】
- 自動販売機事業の情報提供役として、正確な情報提供
- 苦情・不満の場合は共感を示し、解決案を提示
- 商品やサービスに関する質問には詳細回答
- 提案やアイデアには感謝を示し、積極的な姿勢を示す
- オペレーションに関する問い合わせには適切な担当者対応を案内
- 回答は丁寧で、ブランド価値を維持するもの

【自動販売機事業情報】
- 営業時間: 24時間 (メンテナンス時を除く)
- 商品: ソフトドリンク、缶コーヒー、お菓子、サンドイッチなど
- 他店舗: 不動産紹介可能
- メンテナンス: 定期的な商品補充と機械清掃を実施

【出力形式】
JSON形式で以下の構造で回答してください：
{{
    "inquiry_analysis": {{
        "category": "問い合わせ分類",
        "urgency": "emergency/normal/review",
        "sentiment": "satisfied/dissatisfied/question/suggestion",
        "key_topics": ["話題の主要キーワードリスト"]
    }},
    "response_strategy": {{
        "approach": "対応方針",
        "information_needed": ["追加で必要な情報リスト"],
        "recommended_actions": ["推奨対応アクションリスト"]
    }},
    "customer_response": "顧客への最終回答文（丁寧でわかりやすく、100文字以上）",
    "follow_up_required": "true/false",
    "next_steps": ["後続対応が必要な項目"]
}}
"""

        # LLMに問い合わせ分析を依頼
        messages = [
            management_agent.llm_manager.create_ai_message(
                role="system",
                content="あなたは自動販売機事業の顧客サポート担当者です。顧客の問い合わせに適切に対応してください。",
            ),
            management_agent.llm_manager.create_ai_message(
                role="user", content=inquiry_analysis_prompt
            ),
        ]

        response = await management_agent.llm_manager.generate_response(
            messages, max_tokens=1000
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
            inquiry_analysis = analysis_result.get("inquiry_analysis", {})
            inquiry_analysis.setdefault("category", "general_inquiry")
            inquiry_analysis.setdefault("urgency", "normal")
            inquiry_analysis.setdefault("sentiment", "neutral")
            inquiry_analysis.setdefault("key_topics", [])

            response_strategy = analysis_result.get("response_strategy", {})
            response_strategy.setdefault("approach", "standard_response")
            response_strategy.setdefault("information_needed", [])
            response_strategy.setdefault("recommended_actions", [])

            customer_response = analysis_result.get(
                "customer_response",
                "お問い合わせいただきありがとうございます。当店の詳細について調査いたします。",
            )
            follow_up_required = analysis_result.get("follow_up_required", False)
            next_steps = analysis_result.get("next_steps", [])

            logger.info(
                f"LLM問い合わせ分析完了: カテゴリ={inquiry_analysis['category']}, 緊急度={inquiry_analysis['urgency']}"
            )

            return {
                "customer_id": customer_id,
                "inquiry": inquiry,
                "inquiry_analysis": inquiry_analysis,
                "response": customer_response,
                "follow_up_required": follow_up_required,
                "next_steps": next_steps,
                "status": "analyzed_and_responded",
                "llm_used": True,
                "analysis_confidence": "high",
            }

        else:
            logger.warning(f"LLM問い合わせ分析失敗: {response.error_message}")
            # LLM失敗時のフォールバックレスポンス
            return {
                "customer_id": customer_id,
                "inquiry": inquiry,
                "response": "お問い合わせいただきありがとうございます。担当者が確認して折り返しご連絡いたします。",
                "status": "responded_with_fallback",
                "llm_used": False,
                "follow_up_required": True,
                "next_steps": ["手動対応が必要"],
            }

    except Exception as e:
        logger.error(f"顧客問い合わせLLM分析エラー: {e}")
        # 完全フォールバック
        return {
            "customer_id": customer_id,
            "inquiry": inquiry,
            "response": "お問い合わせいただきありがとうございます。システムメンテナンス中のため、折り返しご連絡いたします。",
            "status": "error_fallback",
            "llm_used": False,
            "follow_up_required": True,
            "next_steps": ["システムエラー対応", "管理者確認"],
        }
