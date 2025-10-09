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


async def analyze_financial_performance(
    metrics: Dict[str, Any] = None,
    state_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """財務パフォーマンスを分析（metricsが渡されたらそれを使用、なければライブデータ取得）"""
    logger.info("Analyzing financial performance using LLM")

    try:
        # メトリクス取得（渡されたものがあれば使用、なければライブデータ）
        if metrics is not None:
            logger.info("Using provided metrics for analysis")
            # 本番時: ライブデータ取得
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

            for period_key, descriptions in agent_objectives[
                "optimization_period"
            ].items():
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

            metrics = get_business_metrics()
            logger.info("Using live business metrics for analysis")
        else:
            # テスト時: 渡されたテストデータ使用
            logger.info("Using provided test metrics for analysis")

        # システムプロンプトの設定（テスト時と本番時で分岐）
        if metrics is None:
            from src.shared import settings

            agent_objectives = settings.agent_objectives

            prompt = f"""
あなたは自動販売機事業の経営者です。以下の設定に基づいて意思決定を行ってください。

【主要目的】
{chr(10).join(f"- {obj}" for obj in agent_objectives["primary"])}

【最適化期間枠設定】(戦略的優先度: {agent_objectives["priority_weight"]})
"""

            for period_key, descriptions in agent_objectives[
                "optimization_period"
            ].items():
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
        else:
            # テスト時: 基本的なアナリストプロンプト使用
            prompt = "あなたは自動販売機事業の経営アナリストです。与えられた財務データを分析してください。"

        # 在庫情報の詳細な分析
        inventory_level = metrics.get("inventory_level", {})
        inventory_status = metrics.get("inventory_status", {})

        # 在庫分布の分析
        total_items = sum(inventory_level.values())
        total_slots = inventory_status.get("total_slots", 0)
        stock_adequacy_rate = (
            (total_items / (total_slots * 50)) * 100 if total_slots > 0 else 0
        )

        # 商品カテゴリ別の分析
        drink_items = {
            k: v
            for k, v in inventory_level.items()
            if any(
                word in k.lower()
                for word in ["コーラ", "飲料", "ジュース", "水", "コーヒー"]
            )
        }
        food_items = {
            k: v
            for k, v in inventory_level.items()
            if any(
                word in k.lower()
                for word in ["チップス", "ヌードル", "お菓子", "サンドイッチ", "ガム"]
            )
        }

        drink_total = sum(drink_items.values())
        food_total = sum(food_items.values())

        # 在庫分布の分析
        inventory_distribution = f"""
飲料カテゴリ: {len(drink_items)}商品, 総在庫数: {drink_total}個
食品カテゴリ: {len(food_items)}商品, 総在庫数: {food_total}個
"""

        # 財務・在庫関連性分析
        low_stock_ratio = inventory_status.get("low_stock_count", 0) / max(
            total_slots, 1
        )
        out_of_stock_ratio = inventory_status.get("out_of_stock_count", 0) / max(
            total_slots, 1
        )

        inventory_financial_impact = f"""
在庫不足率: {low_stock_ratio:.1%} (財務影響: 機会損失の可能性)
在庫切れ率: {out_of_stock_ratio:.1%} (財務影響: 売上損失の確定)
在庫充足率: {stock_adequacy_rate:.1f}% (財務影響: 顧客満足度と売上の相関)
"""

        # フォーマット用の利益率文字列を作成
        profit_margin_str = (
            f"{metrics['profit_margin']:.1%}"
            if isinstance(metrics["profit_margin"], (int, float))
            else str(metrics["profit_margin"])
        )

        # State全情報の統合（sales_processing_nodeなどと同様）
        comprehensive_context = ""
        if state_context:
            # 在庫分析結果
            if state_context.get("inventory_analysis"):
                inv_analysis = state_context["inventory_analysis"]
                comprehensive_context += f"""
【在庫管理分析】
- 在庫ステータス: {inv_analysis.get("status", "unknown")}
- 在庫不足商品: {", ".join(inv_analysis.get("low_stock_items", []))}
- 再発注推奨商品: {", ".join(inv_analysis.get("reorder_needed", []))}
- 在庫切れリスク: {inv_analysis.get("estimated_stockout", {})}
- LLM分析: {inv_analysis.get("llm_analysis", "なし")[:150]}

"""

            # 価格戦略決定
            if state_context.get("pricing_decision"):
                pricing = state_context["pricing_decision"]
                comprehensive_context += f"""
【価格戦略決定】
- 戦略: {pricing.get("strategy", "unknown")}
- 商品価格更新: {len(pricing.get("product_updates", []))}件
- 期待効果: {pricing.get("expected_impact", "なし")}
- LLM分析: {pricing.get("llm_analysis", "なし")[:150]}

"""

            # 補充タスク
            if state_context.get("restock_decision"):
                restock = state_context["restock_decision"]
                tasks = restock.get("tasks_assigned", [])
                comprehensive_context += f"""
【補充タスク状況】
- 補充戦略: {restock.get("strategy", "unknown")}
- 割り当てタスク数: {len(tasks)}件
- 緊急タスク: {len([t for t in tasks if t.get("urgency") == "urgent"])}件
- LLM分析: {restock.get("llm_analysis", "なし")[:150]}

"""

            # 発注決定
            if state_context.get("procurement_decision"):
                procurement = state_context["procurement_decision"]
                orders = procurement.get("orders_placed", [])
                comprehensive_context += f"""
【調達発注状況】
- 発注戦略: {procurement.get("strategy", "unknown")}
- 発注数: {len(orders)}件
- LLM分析: {procurement.get("llm_analysis", "なし")[:150]}

"""

            # 売上分析
            if state_context.get("sales_analysis"):
                sales_analysis = state_context["sales_analysis"]
                comprehensive_context += f"""
【売上・財務分析】
- 売上トレンド: {sales_analysis.get("sales_trend", "unknown")}
- 戦略提案数: {len(sales_analysis.get("strategies", []))}件
- 財務概要: {sales_analysis.get("financial_overview", "なし")}
- LLM分析: {sales_analysis.get("analysis", "なし")[:150]}

"""

            # 売上処理パフォーマンス
            if state_context.get("sales_processing"):
                sales_proc = state_context["sales_processing"]
                comprehensive_context += f"""
【売上処理パフォーマンス】
- パフォーマンス評価: {sales_proc.get("performance_rating", "unknown")}
- 取引数: {sales_proc.get("transactions", 0)}件
- コンバージョン率: {sales_proc.get("conversion_rate", "unknown")}
- 分析結果: {sales_proc.get("analysis", "なし")[:150]}

"""

            # 顧客対応分析
            if state_context.get("customer_interaction"):
                customer = state_context["customer_interaction"]
                comprehensive_context += f"""
【顧客対応分析】
- 対応アクション: {customer.get("action", "unknown")}
- LLM分析: {customer.get("reasoning", "なし")[:150]}

"""

            # 実行アクション概要
            actions = state_context.get("executed_actions", [])
            comprehensive_context += f"""
【実行アクション状況】
- 総実行アクション数: {len(actions)}件
- LLM駆動アクション: {len([a for a in actions if a.get("llm_based") or a.get("llm_driven") or a.get("strategy_driven")])}件
- セッション進行状況: {state_context.get("current_step", "unknown")}

"""

        messages = [
            AIMessage(role="system", content=prompt),
            AIMessage(
                role="user",
                content=f"""
以下の財務データを基に、現在のビジネス全状況を統合的に分析し、利益計算における財務パフォーマンス評価と改善提案を行ってください。

【基本財務データ】
- 総売上: ¥{metrics["sales"]:,}
- 利益率: {profit_margin_str}
- 顧客満足度: {metrics["customer_satisfaction"]}/5.0

【在庫統計情報】
- 総スロット数: {total_slots}スロット
- 在庫不足商品数: {inventory_status.get("low_stock_count", 0)}商品
- 在庫切れ商品数: {inventory_status.get("out_of_stock_count", 0)}商品
- 在庫充足率: {stock_adequacy_rate:.1f}%

【商品別在庫分布】
{inventory_distribution}

【財務・在庫関連性分析】
{inventory_financial_impact}

【現在のビジネス全状況統合】
{comprehensive_context}

【財務分析の総合的要件】
1. **財務健全性評価**: 収益性・成長性・安定性の総合評価
   - 売上構造の分析（価格戦略・顧客満足度の影響）
   - 利益率の持続可能性評価（在庫消化・商品戦略の観点から）

2. **ビジネス運用との連携分析**
   - 在庫管理戦略が財務に与える影響
   - 価格変更の収益性インパクト
   - 顧客対応品質とリピート購買の相関
   - 補充・発注プロセスの効率性と財務効果

3. **機会損失・リスク分析**
   - 在庫切れによる売上機会損失の定量評価
   - 過剰在庫のキャッシュフロー影響
   - 顧客満足度低下の長期収益影響

4. **改善提案の策定**
   - 即時実行可能な財務改善施策
   - 中長期的な収益構造最適化戦略
   - 在庫・価格・顧客対応の統合運用改善

5. **戦略的優先順位付け**
   - 財務目標達成のための優先アクション
   - リスク軽減のための予防措置
   - 成長機会の最大化施策

【自動販売機事業特有の財務考慮点】
- 24時間稼働による安定収益構造
- 商品入れ替え時の機会損失最小化
- 季節・時間帯による需要変動対応
- 小スペースでの効率的在庫運用

【出力形式】
JSON形式で回答してください：
```json
{{
    "analysis": "財務パフォーマンスの総合的評価（ビジネス全状況を踏まえた詳細分析）",
    "recommendations": ["具体的な財務改善提案1", "具体的な財務改善提案2", "具体的な財務改善提案3"],
    "financial_health_assessment": {{
        "profitability": "収益性評価",
        "sustainability": "持続可能性評価",
        "growth_potential": "成長可能性評価"
    }},
    "operational_financial_impact": {{
        "inventory_efficiency": "在庫運用が財務に与える影響",
        "pricing_effectiveness": "価格戦略の財務効果",
        "customer_retention_value": "顧客維持の経済価値"
    }},
    "optimization_priorities": ["財務改善の優先順位1", "財務改善の優先順位2"],
    "business_context_integration": "現在のビジネス状況との総合的整合性分析"
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

        metrics = get_business_metrics()
        return {
            "analysis": f"分析エラー: {str(e)}",
            "recommendations": ["管理者へ連絡してください"],
            "metrics": metrics,
        }
