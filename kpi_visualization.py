#!/usr/bin/env python3
"""
VendingBench KPI Time Series Visualization Tool

Usage:
python kpi_visualization.py

Features:
- KPI history retrieval from database
- Time series visualization of cumulative KPIs
- Multi-KPI comparison chart generation
- KPI achievement analysis report
"""

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# English font settings (removed Japanese fonts for better compatibility)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# seaborn style settings
sns.set_style("whitegrid")
sns.set_palette("husl")


class KPI_Visualizer:
    """VendingBench KPI時系列可視化クラス"""

    def __init__(
        self, db_path: str = "data/vending_bench.db", output_dir: str = "visualizations"
    ):
        """
        Args:
            db_path: データベースファイルパス
            output_dir: 出力ディレクトリ
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # VendingBench KPI目標値
        self.targets = {
            "profit": 50000,  # 月間目標
            "stockout_rate": 0.10,  # 10%以内
            "pricing_accuracy": 0.80,  # 80%以上
            "action_correctness": 0.70,  # 70%以上
            "customer_satisfaction": 3.5,  # 3.5以上
        }

    def get_continuous_simulation_run_ids(self) -> List[str]:
        """10日間シミュレーションの全run_idを取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT run_id FROM benchmarks WHERE run_id LIKE 'continuous_simulation_%' ORDER BY run_id ASC"
            )
            results = cursor.fetchall()
            conn.close()

            if results:
                return [result[0] for result in results]
            else:
                return []
        except Exception as e:
            print(f"continuous_simulation run_id取得エラー: {e}")
            return []

    def get_kpi_history_from_db(self) -> pd.DataFrame:
        """データベースからKPI履歴を取得（10日間シミュレーションの全データ）"""
        try:
            # 10日間シミュレーションの全run_idを取得
            run_ids = self.get_continuous_simulation_run_ids()
            if not run_ids:
                print("10日間シミュレーションのrun_idが見つかりません")
                return pd.DataFrame()

            print(f"10日間シミュレーションのrun_idを使用: {len(run_ids)}件")

            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT
                run_id,
                step,
                profit_actual as profit,
                CASE
                    WHEN total_demand > 0 THEN CAST(stockout_count AS FLOAT) / total_demand
                    ELSE 0.0
                END as stockout_rate,
                pricing_accuracy,
                action_correctness,
                customer_satisfaction
            FROM benchmarks
            WHERE run_id LIKE 'continuous_simulation_%'
            ORDER BY run_id ASC, step ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("10日間シミュレーションのデータが見つかりません")
                return pd.DataFrame()

            print(f"全実行データ取得完了: {len(df)}件")

            # データポイントを間引く（最大100ポイントに制限）
            max_points = 100
            if len(df) > max_points:
                # 均等にサンプリング
                indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
                df = df.iloc[indices].reset_index(drop=True)
                print(f"データポイントを{len(df)}件に間引き")

            # 実行順序に基づいて時系列を生成
            df["execution_order"] = range(1, len(df) + 1)

            # 時系列表示用のタイムスタンプを生成（見やすく間隔を調整）
            base_time = pd.Timestamp.now()
            # データポイントを適切な間隔で配置（1ポイントあたり30分間隔）
            df["timestamp"] = [
                base_time + pd.Timedelta(minutes=i * 30) for i in range(len(df))
            ]

            # 数値変換
            numeric_cols = [
                "profit",
                "stockout_rate",
                "pricing_accuracy",
                "action_correctness",
                "customer_satisfaction",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            print(f"データベースからのKPI履歴取得エラー: {e}")
            return pd.DataFrame()

    def get_cumulative_kpi_data(self) -> Dict[str, Any]:
        """10日間シミュレーションの累積KPIデータを取得"""
        try:
            # 10日間シミュレーションの全run_idを取得
            run_ids = self.get_continuous_simulation_run_ids()
            if not run_ids:
                print("10日間シミュレーションのrun_idが見つかりません")
                return {
                    "total_profit": 0.0,
                    "average_stockout_rate": 0.0,
                    "customer_satisfaction_trend": [],
                    "action_accuracy_history": [],
                    "days": 0,
                }

            conn = sqlite3.connect(self.db_path)

            # 総利益の計算
            total_profit_query = """
            SELECT SUM(profit_actual) as total_profit
            FROM benchmarks
            WHERE run_id LIKE 'continuous_simulation_%'
            """
            total_profit_df = pd.read_sql_query(total_profit_query, conn)
            total_profit = (
                total_profit_df.iloc[0]["total_profit"]
                if not total_profit_df.empty
                else 0.0
            )

            # 平均在庫切れ率の計算
            stockout_query = """
            SELECT AVG(CASE
                WHEN total_demand > 0 THEN CAST(stockout_count AS FLOAT) / total_demand
                ELSE 0.0
            END) as avg_stockout_rate
            FROM benchmarks
            WHERE run_id LIKE 'continuous_simulation_%'
            """
            stockout_df = pd.read_sql_query(stockout_query, conn)
            avg_stockout_rate = (
                stockout_df.iloc[0]["avg_stockout_rate"]
                if not stockout_df.empty
                else 0.0
            )

            # 顧客満足度のトレンド（日ごとの平均）
            satisfaction_query = """
            SELECT customer_satisfaction
            FROM benchmarks
            WHERE run_id LIKE 'continuous_simulation_%'
            AND customer_satisfaction IS NOT NULL
            ORDER BY run_id, step
            """
            satisfaction_df = pd.read_sql_query(satisfaction_query, conn)
            satisfaction_trend = (
                satisfaction_df["customer_satisfaction"].tolist()
                if not satisfaction_df.empty
                else []
            )

            # 行動精度の履歴（日ごとの平均）
            accuracy_query = """
            SELECT action_correctness
            FROM benchmarks
            WHERE run_id LIKE 'continuous_simulation_%'
            AND action_correctness IS NOT NULL
            ORDER BY run_id, step
            """
            accuracy_df = pd.read_sql_query(accuracy_query, conn)
            accuracy_history = (
                accuracy_df["action_correctness"].tolist()
                if not accuracy_df.empty
                else []
            )

            conn.close()

            return {
                "total_profit": total_profit,
                "average_stockout_rate": avg_stockout_rate,
                "customer_satisfaction_trend": satisfaction_trend[-10:]
                if len(satisfaction_trend) > 10
                else satisfaction_trend,
                "action_accuracy_history": accuracy_history[-10:]
                if len(accuracy_history) > 10
                else accuracy_history,
                "days": len(run_ids),
            }

        except Exception as e:
            print(f"累積KPIデータ取得エラー: {e}")
            return {
                "total_profit": 0.0,
                "average_stockout_rate": 0.0,
                "customer_satisfaction_trend": [],
                "action_accuracy_history": [],
                "days": 0,
            }

    def create_time_series_plots(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """時系列KPIチャートを作成"""
        figures = {}

        if df.empty:
            print("KPIデータがありません")
            return figures

        # 図のサイズ
        figsize = (12, 8)

        # 1. 全KPI統合チャート
        fig1, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()

        # KPI columns in English
        kpi_cols = {
            "profit": ("Profit", "¥{:.0f}", "blue"),
            "stockout_rate": ("Stockout Rate", "{:.1%}", "red"),
            "pricing_accuracy": ("Pricing Accuracy", "{:.1%}", "green"),
            "action_correctness": ("Action Correctness", "{:.1%}", "orange"),
            "customer_satisfaction": ("Customer Satisfaction", "{:.1f}", "purple"),
        }

        # X軸ラベル用に時系列を生成
        df_plot = df.copy()

        # より明確な時系列表示のために、実行順序を時間軸に変換
        # データポイントを適切に分散させて時系列的に表示
        if len(df_plot) > 0:
            # データポイントを1時間間隔で分散（見やすくするため）
            start_time = pd.Timestamp.now() - pd.Timedelta(hours=len(df_plot))
            df_plot["timestamp"] = [
                start_time + pd.Timedelta(hours=i) for i in range(len(df_plot))
            ]
        else:
            df_plot["timestamp"] = pd.to_datetime([])

        for i, (col, (label, format_str, color)) in enumerate(kpi_cols.items()):
            if col in df_plot.columns and i < len(axes):
                ax = axes[i]

                # データプロット
                if "run_id" in df_plot.columns:
                    # 複数run_idがある場合は分けてプロット
                    for run_id in df_plot["run_id"].unique():
                        run_data = df_plot[df_plot["run_id"] == run_id].copy()
                        ax.plot(
                            run_data["timestamp"],
                            run_data[col],
                            marker="o",
                            label=f"Run {run_id}",
                            color=color,
                            alpha=0.7,
                        )
                else:
                    ax.plot(
                        df_plot["timestamp"],
                        df_plot[col],
                        marker="o",
                        color=color,
                        alpha=0.7,
                    )

                # 目標線（該当する場合）
                if col == "stockout_rate":
                    ax.axhline(
                        y=self.targets["stockout_rate"],
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Target",
                    )
                elif col in [
                    "pricing_accuracy",
                    "action_correctness",
                    "customer_satisfaction",
                ]:
                    target_key = col.replace("_", "_")
                    if target_key in self.targets:
                        ax.axhline(
                            y=self.targets[target_key],
                            color="green",
                            linestyle="--",
                            alpha=0.5,
                            label="Target",
                        )

                ax.set_title(f"{label} Time Series", fontsize=12, fontweight="bold")
                ax.set_xlabel("Execution Order")
                ax.set_ylabel(label)

                # グリッドと凡例
                ax.grid(True, alpha=0.3)
                if "run_id" in df_plot.columns:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                # X軸を時系列としてフォーマット
                if not df_plot.empty:
                    # X軸の目盛りを設定（実行順序で表示）
                    x_ticks = df_plot["timestamp"]
                    ax.set_xticks(x_ticks)

                    # X axis labels in English: "Step 1, Step 2, ..."
                    labels = [f"Step {i + 1}" for i in range(len(df_plot))]
                    ax.set_xticklabels(labels, rotation=45, ha="right")

                    # X軸の範囲を調整
                    if len(x_ticks) > 0:
                        ax.set_xlim(x_ticks.iloc[0], x_ticks.iloc[-1])

        # 余分なサブプロットを非表示
        for i in range(len(kpi_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        figures["all_kpis"] = fig1

        # 2. 収益・満足度重点チャート
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 収益推移
        if "profit" in df.columns:
            ax1.plot(
                df_plot["timestamp"],
                df_plot["profit"],
                "b-o",
                linewidth=2,
                markersize=6,
            )
            ax1.fill_between(
                df_plot["timestamp"], df_plot["profit"], alpha=0.3, color="blue"
            )
            ax1.set_title("Cumulative Profit Trend", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Execution Order")
            ax1.set_ylabel("Profit (¥)")
            ax1.grid(True, alpha=0.3)

            # X軸を時系列としてフォーマット
            if not df_plot.empty:
                x_ticks = df_plot["timestamp"]
                ax1.set_xticks(x_ticks)
                labels = [f"Step {i + 1}" for i in range(len(df_plot))]
                ax1.set_xticklabels(labels, rotation=45, ha="right")
                if len(x_ticks) > 0:
                    ax1.set_xlim(x_ticks.iloc[0], x_ticks.iloc[-1])

        # 顧客満足度推移
        if "customer_satisfaction" in df.columns:
            ax2.plot(
                df_plot["timestamp"],
                df_plot["customer_satisfaction"],
                "g-o",
                linewidth=2,
                markersize=6,
            )
            ax2.fill_between(
                df_plot["timestamp"],
                df_plot["customer_satisfaction"],
                alpha=0.3,
                color="green",
            )
            ax2.axhline(
                y=self.targets["customer_satisfaction"],
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Target",
            )
            ax2.set_title("Customer Satisfaction Trend", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Execution Order")
            ax2.set_ylabel("Satisfaction (out of 5)")
            ax2.set_ylim(0, 5)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # X軸を時系列としてフォーマット
            if not df_plot.empty:
                x_ticks = df_plot["timestamp"]
                ax2.set_xticks(x_ticks)
                labels = [f"Step {i + 1}" for i in range(len(df_plot))]
                ax2.set_xticklabels(labels, rotation=45, ha="right")
                if len(x_ticks) > 0:
                    ax2.set_xlim(x_ticks.iloc[0], x_ticks.iloc[-1])

        plt.tight_layout()
        figures["revenue_satisfaction"] = fig2

        # 3. パフォーマンス指標ダッシュボード
        fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 在庫切れ率推移
        if "stockout_rate" in df.columns:
            ax1.plot(
                df_plot["timestamp"],
                df_plot["stockout_rate"] * 100,
                "r-s",
                linewidth=2,
                markersize=6,
            )
            ax1.axhline(
                y=self.targets["stockout_rate"] * 100,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Target",
            )
            ax1.set_title("Stockout Rate Trend", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Execution Order")
            ax1.set_ylabel("Stockout Rate (%)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # X軸を時系列としてフォーマット
            if not df_plot.empty:
                x_ticks = df_plot["timestamp"]
                ax1.set_xticks(x_ticks)
                labels = [f"Step {i + 1}" for i in range(len(df_plot))]
                ax1.set_xticklabels(labels, rotation=45, ha="right")
                if len(x_ticks) > 0:
                    ax1.set_xlim(x_ticks.iloc[0], x_ticks.iloc[-1])

        # 価格精度・行動正しさ比較
        metrics_to_plot = [
            ("pricing_accuracy", "価格精度", "blue"),
            ("action_correctness", "行動正確性", "orange"),
        ]

        for i, (col, label, color) in enumerate(metrics_to_plot):
            if col in df.columns:
                ax = [ax2, ax3][i]
                ax.plot(
                    df_plot["timestamp"],
                    df_plot[col] * 100,
                    color=color,
                    marker="^",
                    linestyle="-",
                    linewidth=2,
                    markersize=6,
                    label=label,
                )

                target_key = col.replace("_", "_")
                if target_key in self.targets:
                    ax.axhline(
                        y=self.targets[target_key] * 100,
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{label} Target",
                    )

                ax.set_title(f"{label} Trend", fontsize=12, fontweight="bold")
                ax.set_xlabel("Execution Order")
                ax.set_ylabel(f"{label} (%)")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # X軸を時系列としてフォーマット
                if not df_plot.empty:
                    x_ticks = df_plot["timestamp"]
                    ax.set_xticks(x_ticks)
                    labels = [f"Step {i + 1}" for i in range(len(df_plot))]
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    if len(x_ticks) > 0:
                        ax.set_xlim(x_ticks.iloc[0], x_ticks.iloc[-1])

        # 統合パフォーマンス指標（4番目のサブプロット）
        performance_cols = [
            "stockout_rate",
            "pricing_accuracy",
            "action_correctness",
            "customer_satisfaction",
        ]
        performance_data = []
        performance_labels = []

        for col in performance_cols:
            if col in df.columns:
                latest_val = df[col].iloc[-1] if not df.empty else 0

                if col == "stockout_rate":
                    # 在庫切れ率は低い方が良い
                    score = max(0, 100 - (latest_val * 100))
                    performance_labels.append("在庫管理効率")
                else:
                    score = latest_val * 100 if latest_val <= 1 else latest_val
                    label_map = {
                        "pricing_accuracy": "価格管理効率",
                        "action_correctness": "業務正確性",
                        "customer_satisfaction": "顧客満足度",
                    }
                    performance_labels.append(label_map.get(col, col))

                performance_data.append(score)

        if performance_data:
            bars = ax4.bar(
                performance_labels,
                performance_data,
                color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"],
            )
            ax4.set_title("Performance Summary", fontsize=12, fontweight="bold")
            ax4.set_ylabel("Score (%)")

            # バー上に値表示
            for bar, value in zip(bars, performance_data):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        figures["performance_dashboard"] = fig3

        return figures

    def create_cumulative_kpi_chart(
        self, cumulative_data: Dict[str, Any]
    ) -> plt.Figure:
        """累積KPIデータを可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Total Profit
        axes[0, 0].bar(
            ["Total Profit"],
            [cumulative_data.get("total_profit", 0)],
            color="blue",
            alpha=0.7,
        )
        axes[0, 0].set_title("Total Profit", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("Profit (¥)")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # Average Stockout Rate
        avg_stockout = cumulative_data.get("average_stockout_rate", 0) * 100
        axes[0, 1].bar(["Avg Stockout Rate"], [avg_stockout], color="red", alpha=0.7)
        axes[0, 1].axhline(
            y=self.targets["stockout_rate"] * 100,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Target",
        )
        axes[0, 1].set_title("Average Stockout Rate", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("Stockout Rate (%)")
        axes[0, 1].grid(True, alpha=0.3, axis="y")
        axes[0, 1].legend()

        # Customer Satisfaction Trend
        satisfaction_trend = cumulative_data.get("customer_satisfaction_trend", [])
        if satisfaction_trend:
            days = range(1, len(satisfaction_trend) + 1)
            axes[1, 0].plot(days, satisfaction_trend, "g-o", linewidth=2, markersize=8)
            axes[1, 0].fill_between(days, satisfaction_trend, alpha=0.3, color="green")
            axes[1, 0].axhline(
                y=self.targets["customer_satisfaction"],
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Target",
            )
            axes[1, 0].set_title(
                "Customer Satisfaction Trend", fontsize=14, fontweight="bold"
            )
            axes[1, 0].set_xlabel("Days")
            axes[1, 0].set_ylabel("Satisfaction (out of 5)")
            axes[1, 0].set_ylim(0, 5)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Action Accuracy History
        accuracy_history = cumulative_data.get("action_accuracy_history", [])
        if accuracy_history:
            days = range(1, len(accuracy_history) + 1)
            axes[1, 1].plot(
                days, np.array(accuracy_history) * 100, "b-s", linewidth=2, markersize=8
            )
            axes[1, 1].fill_between(
                days, np.array(accuracy_history) * 100, alpha=0.3, color="blue"
            )
            axes[1, 1].axhline(
                y=self.targets["action_correctness"] * 100,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Target",
            )
            axes[1, 1].set_title(
                "Action Accuracy History", fontsize=14, fontweight="bold"
            )
            axes[1, 1].set_xlabel("Days")
            axes[1, 1].set_ylabel("Accuracy (%)")
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def generate_kpi_report(
        self, df: pd.DataFrame, cumulative_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """KPI分析レポート生成"""
        if df.empty:
            return {"error": "KPIデータがありません"}

        # 最新のKPI値を取得
        latest_kpi = {}
        for col in [
            "profit",
            "stockout_rate",
            "pricing_accuracy",
            "action_correctness",
            "customer_satisfaction",
        ]:
            if col in df.columns and not df[col].empty:
                latest_kpi[col] = df[col].iloc[-1]

        # KPI達成度分析
        achievement_analysis = {}
        for kpi, value in latest_kpi.items():
            target = self.targets.get(kpi)
            if target is not None:
                if kpi == "stockout_rate":
                    # 在庫切れ率は低い方が良い
                    achievement = (target - value) / target * 100
                else:
                    achievement = value / target * 100
                achievement_analysis[kpi] = {
                    "value": value,
                    "target": target,
                    "achievement_rate": max(0, min(100, achievement)),
                    "status": "達成" if achievement >= 80 else "要改善",
                }

        # トレンド分析
        trend_analysis = {}
        if len(df) >= 2:
            for col in ["profit", "customer_satisfaction"]:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) >= 2:
                        first_val = values.iloc[0]
                        last_val = values.iloc[-1]
                        if first_val != 0:
                            trend = (last_val - first_val) / abs(first_val) * 100
                            trend_analysis[col] = {
                                "direction": "上昇"
                                if trend > 0
                                else "下降"
                                if trend < 0
                                else "横ばい",
                                "change_rate": trend,
                                "first_value": first_val,
                                "last_value": last_val,
                            }

        return {
            "summary": {
                "total_days": cumulative_data.get("days", 0),
                "total_runs": len(df["run_id"].unique())
                if "run_id" in df.columns
                else 0,
                "latest_metrics": latest_kpi,
                "cumulative_totals": {
                    "profit": cumulative_data.get("total_profit", 0),
                    "avg_stockout_rate": cumulative_data.get("average_stockout_rate", 0)
                    * 100,
                },
            },
            "achievement_analysis": achievement_analysis,
            "trend_analysis": trend_analysis,
            "recommendations": self.generate_recommendations(
                achievement_analysis, trend_analysis
            ),
        }

    def generate_recommendations(
        self, achievement_analysis: Dict, trend_analysis: Dict
    ) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []

        # Recommendations based on achievement analysis
        for kpi, analysis in achievement_analysis.items():
            if analysis["status"] == "要改善":
                if kpi == "profit":
                    recommendations.append(
                        "Consider reviewing pricing strategy or introducing new products to improve profitability"
                    )
                elif kpi == "stockout_rate":
                    recommendations.append(
                        "Optimize procurement cycles and improve inventory forecasting to reduce stockouts"
                    )
                elif kpi == "pricing_accuracy":
                    recommendations.append(
                        "Enhance pricing accuracy through competitive analysis and customer feedback collection"
                    )
                elif kpi == "action_correctness":
                    recommendations.append(
                        "Improve operational accuracy through employee training and process optimization"
                    )
                elif kpi == "customer_satisfaction":
                    recommendations.append(
                        "Enhance customer satisfaction through service quality improvements and follow-up processes"
                    )

        # Recommendations based on trend analysis
        for kpi, trend in trend_analysis.items():
            if trend["direction"] == "下降":
                if kpi == "profit":
                    recommendations.append(
                        "Implement cost reduction and sales promotion measures to address declining profit trends"
                    )
                elif kpi == "customer_satisfaction":
                    recommendations.append(
                        "Strengthen quality management and customer service to address declining satisfaction trends"
                    )

        if not recommendations:
            recommendations.append(
                "Overall performance is being maintained at a good level"
            )

        return recommendations

    def save_plots(self, figures: Dict[str, plt.Figure], output_dir: Path = None):
        """チャートをファイルに保存"""
        if output_dir is None:
            output_dir = self.output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for chart_name, fig in figures.items():
            filename = f"kpi_{chart_name}_{timestamp}.png"
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✅ チャート保存: {filepath}")

    def export_data(
        self, df: pd.DataFrame, cumulative_data: Dict[str, Any], report: Dict[str, Any]
    ):
        """データを各種形式でエクスポート"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSVエクスポート
        if not df.empty:
            csv_path = self.output_dir / f"kpi_data_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ CSVデータ出力: {csv_path}")

        # JSONレポートエクスポート（Timestampを文字列に変換）
        json_path = self.output_dir / f"kpi_report_{timestamp}.json"

        # DataFrameを辞書形式に変換し、JSONシリアライズ可能な形式にする
        kpi_history = []
        if not df.empty:
            for _, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    val = row[col]
                    # TimestampをISO文字列に変換
                    if hasattr(val, "isoformat"):
                        row_dict[col] = val.isoformat()
                    elif hasattr(val, "timestamp") and hasattr(val, "timetuple"):
                        # date型の場合
                        row_dict[col] = val.isoformat()
                    else:
                        row_dict[col] = val
                kpi_history.append(row_dict)

        export_data = {
            "kpi_history": kpi_history,
            "cumulative_kpis": cumulative_data,
            "analysis_report": report,
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSONレポート出力: {json_path}")

    def run_complete_analysis(
        self, save_plots: bool = True, export_data: bool = True
    ) -> Dict[str, Any]:
        """完全分析実行（データ取得・可視化・レポート生成）"""
        print("🚀 VendingBench KPI時系列分析を開始します...")
        print("=" * 60)

        # データ取得
        print("📊 KPIデータをデータベースから取得中...")
        df = self.get_kpi_history_from_db()
        cumulative_data = self.get_cumulative_kpi_data()

        if df.empty:
            print("⚠️  KPIデータが見つからないか、データベースが空です")
            return {}

        print(f"✅ データ取得完了: {len(df)}件のKPIレコード")

        # レポート生成
        print("📈 KPIレポートを生成中...")
        report = self.generate_kpi_report(df, cumulative_data)

        # 可視化チャート生成
        print("🎨 時系列チャートを生成中...")
        time_series_figures = self.create_time_series_plots(df)
        cumulative_figure = self.create_cumulative_kpi_chart(cumulative_data)

        all_figures = {**time_series_figures, "cumulative_kpis": cumulative_figure}

        # 保存
        if save_plots and all_figures:
            print("💾 チャートを保存中...")
            self.save_plots(all_figures)

        if export_data:
            print("💾 データをエクスポート中...")
            self.export_data(df, cumulative_data, report)

        # レポート表示
        print("\n" + "=" * 60)
        print("📊 VENDING BENCH KPI分析レポート")
        print("=" * 60)

        summary = report.get("summary", {})
        print(f"📅 分析期間: {summary.get('total_days', 0)}日間")
        print(f"🔄 実行回数: {summary.get('total_runs', 0)}回")
        print(
            f"💰 最終収益: ¥{summary.get('latest_metrics', {}).get('profit', 0):,.0f}"
        )
        print(
            f"📦 在庫切れ率: {summary.get('latest_metrics', {}).get('stockout_rate', 0):.1%}"
        )
        print(
            f"🎯 顧客満足度: {summary.get('latest_metrics', {}).get('customer_satisfaction', 0):.1f}/5.0"
        )

        print("\n🏆 KPI Achievement Status:")
        for kpi, analysis in report.get("achievement_analysis", {}).items():
            status_emoji = "✅" if analysis["status"] == "達成" else "⚠️"
            kpi_name_map = {
                "profit": "Profit",
                "stockout_rate": "Stockout Rate",
                "pricing_accuracy": "Pricing Accuracy",
                "action_correctness": "Action Correctness",
                "customer_satisfaction": "Customer Satisfaction",
            }
            kpi_name = kpi_name_map.get(kpi, kpi)
            print(
                f"  {status_emoji} {kpi_name}: {analysis['achievement_rate']:.1f}% ({analysis['status']})"
            )

        print("\n💡 Improvement Recommendations:")
        for i, rec in enumerate(report.get("recommendations", []), 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 60)

        return {
            "data": df,
            "cumulative_data": cumulative_data,
            "report": report,
            "figures": all_figures,
            "charts_saved": save_plots,
            "data_exported": export_data,
        }


def main():
    """メイン実行関数"""
    print("🎯 VendingBench KPI時系列可視化ツール")
    print("-" * 50)

    # 可視化実行
    visualizer = KPI_Visualizer()

    try:
        result = visualizer.run_complete_analysis(save_plots=True, export_data=True)

        if result:
            print(
                "\n🎉 分析完了！ 可視化ファイルを 'visualizations' ディレクトリで確認してください"
            )
            print("📁 生成ファイル:")
            print("  - KPI時系列チャート（PNG）")
            print("  - パフォーマンスダッシュボード（PNG）")
            print("  - KPI履歴データ（CSV）")
            print("  - 分析レポート（JSON）")
        else:
            print("\n❌ 分析でエラーが発生しました")

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
