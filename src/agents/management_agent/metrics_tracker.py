"""
VendingBench Metrics Tracker
Agentの意思決定時に現在のメトリクス状況を提供するクラス
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.agents.management_agent.agent import ManagementState

from src.agents.management_agent.evaluation_metrics import (
    calculate_current_metrics_for_agent,
    format_metrics_for_llm_prompt,
)
from src.shared.config.vending_bench_metrics import VENDING_BENCH_METRICS

# ロガーの取得
logger = logging.getLogger(__name__)


class VendingBenchMetricsTracker:
    """
    VendingBenchメトリクスを追跡し、Agentの意思決定に活用するためのクラス

    主な機能:
    - 各node実行前のメトリクス状態計算
    - LLMプロンプト用メトリクス情報整形
    - メトリクス改善度の追跡
    """

    def __init__(self, difficulty: str = "normal"):
        """
        Args:
            difficulty: 目標値の難易度 ("easy", "normal", "hard")
        """
        self.difficulty = difficulty
        self.last_metrics: Optional[Dict[str, Any]] = None

    def calculate_current_state(self, state: "ManagementState") -> Dict[str, Any]:
        """
        現在の全メトリクス状態を計算

        Args:
            state: 現在のManagementState

        Returns:
            Agent用メトリクス状態辞書
        """
        try:
            current_metrics = calculate_current_metrics_for_agent(state)
            self.last_metrics = current_metrics
            return current_metrics
        except Exception as e:
            # エラー時はデフォルト値
            print(f"Metrics calculation error: {e}")
            return self._get_default_metrics()

    def format_for_llm_prompt(self, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        LLMプロンプト用にメトリクス情報を整形

        Args:
            metrics: メトリクス状態辞書（Noneの場合はlast_metricsを使用）

        Returns:
            LLMプロンプト用整形文字列
        """
        if metrics is None:
            metrics = self.last_metrics

        if metrics is None:
            return "メトリクス情報: 計算待ち"

        try:
            return format_metrics_for_llm_prompt(metrics)
        except Exception as e:
            print(f"Metrics formatting error: {e}")
            return "メトリクス情報: フォーマットエラー"

    def get_metrics_summary(
        self, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        メトリクスの概要情報を取得

        Args:
            metrics: メトリクス状態辞書

        Returns:
            概要情報辞書
        """
        if metrics is None:
            metrics = self.last_metrics

        if metrics is None:
            return {
                "total_metrics": 0,
                "pass_count": 0,
                "fail_count": 0,
                "critical_count": 0,
            }

        pass_count = sum(1 for m in metrics.values() if m.get("status") == "PASS")
        fail_count = sum(1 for m in metrics.values() if m.get("status") == "FAIL")
        critical_count = len(
            [m for m in metrics.values() if m.get("gap", 0) < -0.5]
        )  # かなり悪いもの

        return {
            "total_metrics": len(metrics),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "critical_count": critical_count,
            "pass_ratio": pass_count / len(metrics) if metrics else 0,
            "requires_attention": fail_count > 0,
        }

    def should_prioritize_metric(
        self, metric_name: str, metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        指定メトリクスの優先度が高いかを判定

        Args:
            metric_name: メトリクス名
            metrics: メトリクス状態辞書

        Returns:
            優先度が高い場合True
        """
        if metrics is None:
            metrics = self.last_metrics

        if metrics is None or metric_name not in metrics:
            return False

        metric_data = metrics[metric_name]

        # FAIL状態の場合は常に優先
        if metric_data.get("status") == "FAIL":
            return True

        # ギャップが大きい場合は優先
        gap = abs(metric_data.get("gap", 0))
        threshold = self._get_priority_threshold(metric_name)
        return gap > threshold

    def _get_priority_threshold(self, metric_name: str) -> float:
        """
        メトリクスごとの優先度判定閾値を取得

        Args:
            metric_name: メトリクス名

        Returns:
            優先度判定閾値
        """
        thresholds = {
            "profit": 50000,  # ¥50,000以上のギャップ
            "stockout_rate": 0.05,  # 5%以上のギャップ
            "pricing_accuracy": 0.1,  # 10%以上のギャップ
            "action_correctness": 0.2,  # 20%以上のギャップ
            "customer_satisfaction": 0.5,  # 0.5ポイント以上のギャップ
            "long_term_consistency": 0.1,  # 10%以上のギャップ
        }
        return thresholds.get(metric_name, 0.1)

    def generate_strategy_guidance(
        self, metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        メトリクス状況に基づいた戦略指針を生成

        Args:
            metrics: メトリクス状態辞書

        Returns:
            戦略指針テキスト
        """
        if metrics is None:
            metrics = self.last_metrics

        if metrics is None:
            return "戦略指針: メトリクス状況把握待ち"

        summary = self.get_metrics_summary(metrics)

        guidance_parts = []

        # 全体状況評価
        pass_ratio = summary["pass_ratio"]
        if pass_ratio >= 0.8:
            guidance_parts.append("全体状況良好 - 既存戦略の最適化と継続に注力")
        elif pass_ratio >= 0.6:
            guidance_parts.append("一部改善要 - FAIL項目の優先対応と全体バランスの維持")
        else:
            guidance_parts.append("総合改善要 - 複数項目の同時改善と抜本的戦略見直し")

        # FAIL項目の特定と指針
        fail_metrics = [
            name for name, data in metrics.items() if data.get("status") == "FAIL"
        ]
        if fail_metrics:
            priority_items = []
            for metric in fail_metrics:
                if metric == "profit":
                    priority_items.append("収益改善（価格戦略・販促施策）")
                elif metric == "stockout_rate":
                    priority_items.append("在庫安定化（補充計画・需給予測）")
                elif metric == "pricing_accuracy":
                    priority_items.append("価格最適化（市場分析・競争戦略）")
                elif metric == "action_correctness":
                    priority_items.append("業務効率化（プロセス改善・自動化）")
                elif metric == "customer_satisfaction":
                    priority_items.append(
                        "顧客体験向上（サービス改善・エンゲージメント）"
                    )
                elif metric == "long_term_consistency":
                    priority_items.append("一貫性確保（戦略継続・フィードバック活用）")

            if priority_items:
                guidance_parts.append(f"優先課題: {', '.join(priority_items[:3])}")

        # 戦略的アドバイス
        if summary["critical_count"] > 2:
            guidance_parts.append("緊急性高 - 短期対応を優先しつつ長期視点の戦略立案")
        else:
            guidance_parts.append(
                "バランス重視 - Primary/Secondary Metricsの調和的改善"
            )

        return " | ".join(guidance_parts)

    def _get_default_metrics(self) -> Dict[str, Any]:
        """
        エラー時のデフォルトメトリクス情報を返す

        Returns:
            デフォルトメトリクス辞書
        """
        return {
            metric_name: {
                "current": 0,
                "target": config["target"],
                "gap": -config["target"],
                "status": "UNKNOWN",
                "direction": config["direction"],
                "description": config["description"],
            }
            for metric_name, config in {
                **VENDING_BENCH_METRICS["primary_metrics"],
                **VENDING_BENCH_METRICS["secondary_metrics"],
            }.items()
        }

    def reset(self):
        """メトリクス履歴をリセット"""
        self.last_metrics = None

    def update_step_metrics(
        self, run_id: str, step: int, metrics_result: Dict[str, Any]
    ):
        """
        ステップ単位メトリクスの更新を実行
        VendingBench eval_step_metrics() の結果をトラッカーへ反映

        Args:
            run_id: 実行ID
            step: ステップ番号
            metrics_result: eval_step_metrics() の戻り値
        """
        try:
            # エラー時でもさえログ出力のみ行い、正常処理継続
            # stateはNoneで渡されても問題ないため、エラーは抑止
            if metrics_result.get("status") == "success":
                logger.info(
                    f"Metrics tracker updated with step metrics: run_id={run_id}, step={step}, status=success"
                )
                # successの場合は特に何もしない（last_metricsはそのまま）
            else:
                logger.warning(
                    f"Metrics tracker received failed step metrics: run_id={run_id}, step={step}, error={metrics_result.get('error', 'unknown')}"
                )
                # エラー情報はログに保存するのみ

        except Exception as e:
            logger.error(f"Failed to update step metrics in tracker: {e}")
            # エラーが発生しても処理を継続（throwしない）

    def step_metrics_evaluation(
        self, db, run_id: str, step: int, state: "ManagementState"
    ) -> Dict[str, Any]:
        """
        VendingBench準拠のステップ単位メトリクス評価を実行
        各node実行後に呼び出し、リアルタイムmetrics更新を行う

        Args:
            db: データベース接続
            run_id: 実行ID
            step: ステップ番号
            state: 現在のManagementState

        Returns:
            評価結果metrics_dict
        """
        from src.agents.management_agent.evaluation_metrics import eval_step_metrics

        try:
            # eval_step_metrics実行（VendingBench準拠）
            metrics_dict = eval_step_metrics(db, run_id, step, state)

            # 現在のmetrics状態も更新（LLMプロンプト用）
            updated_metrics = self.calculate_current_state(state)
            self.last_metrics = updated_metrics

            return metrics_dict

        except Exception as e:
            print(f"VendingBench step metrics evaluation error: {e}")
            return {
                "run_id": run_id,
                "step": step,
                "status": "error",
                "error": str(e),
                "evaluation_timestamp": "None",
            }
