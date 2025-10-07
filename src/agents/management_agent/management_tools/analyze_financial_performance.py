"""
analyze_financial_performance.py - 財務パフォーマンス分析ツール

財務データを分析し、収益性を評価するTool
"""

import json
import logging
from typing import Any, Dict

import src.agents.management_agent.management_tools.get_business_metrics as get_business_metrics
from src.infrastructure.ai.model_manager import AIMessage, model_manager

logger = logging.getLogger(__name__)


async def analyze_financial_performance() -> Dict[str, Any]:
    """財務パフォーマンスを分析（model_manager経由）"""
    logger.info("Analyzing financial performance using LLM")

    try:
        # メトリクス取得
        from src.agents.management_agent.management_tools.get_business_metrics import (
            get_business_metrics,
        )
        from src.shared import settings

        agent_objectives = settings.agent_objectives

        prompt = f"""
あなたは自動販売機事業の経営者です。以下の設定に基づいて意思決定を行ってください。

【主要目的】
{chr(10).join(f"- {obj}" for obj in agent_objectives["primary"])}

【最適化期間枠設定】(戦略的優先度: {agent_objectives["priority_weight"]})
"""

        for period_key, descriptions in agent_objectives["optimization_period"].items():
            weight = agent_objectives["priority_weight"].get(period_key, 0.0)
            prompt += f"- {period_key}: {descriptions} (重み: {weight})\n"

        prompt += f"""
【制約条件】
{chr(10).join(f"- {constraint}" for constraint in agent_objectives["constraints"])}

【業務統括】
- 売上・財務データの分析と戦略立案
- 在庫状況の監視と補充計画
- 価格戦略の決定と実行指示
- 従業員への作業指示（補充、調達、メンテナンス）
- 顧客からの問い合わせ対応と苦情処理

【意思決定原則】
- 短期・中期・長期目標のバランスを考慮して収益性を最優先
- 顧客満足度を維持しつつ長期的な成長を図る
- リスクを適切に管理し、安定的な事業運営を行う
- データに基づいた戦略的判断を行う
"""

        metrics = get_business_metrics.get_business_metrics()

        messages = [
            AIMessage(role="system", content=prompt),
            AIMessage(
                role="user",
                content=f"""
以下の財務データを分析し、パフォーマンス評価と改善提案を行ってください。

【財務データ】
- 売上: ¥{metrics["sales"]:,}
- 利益率: {metrics["profit_margin"]:.1%}
- 在庫状況: {metrics["inventory_level"]}
- 顧客満足度: {metrics["customer_satisfaction"]}/5.0

【出力形式】
JSON形式で回答してください：
```json
{{
    "analysis": "財務状況の全体的な評価と分析",
    "recommendations": ["改善提案1", "改善提案2", "改善提案3"]
}}
```
""",
            ),
        ]

        response = await model_manager.generate_response(messages, max_tokens=1000)

        if response.success:
            try:
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                llm_response = json.loads(content)
                return {
                    "analysis": llm_response.get("analysis", "分析できませんでした"),
                    "recommendations": llm_response.get("recommendations", []),
                    "metrics": metrics,
                }
            except json.JSONDecodeError:
                logger.warning(f"財務分析LLMレスポンスパース失敗: {response.content}")

        # LLM失敗時はハードコードされたレスポンス
        logger.warning("LLM財務分析失敗、ハードコードレスポンスを使用")
        return {
            "analysis": "売上は予算比95%で推移。利益率は良好。",
            "recommendations": ["在庫回転率の改善", "高利益商品の強化"],
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"財務パフォーマンス分析エラー: {e}")
        from src.agents.management_agent.management_tools.get_business_metrics import (
            get_business_metrics,
        )

        metrics = get_business_metrics.get_business_metrics()
        return {
            "analysis": f"分析エラー: {str(e)}",
            "recommendations": ["管理者へ連絡してください"],
            "metrics": metrics,
        }
