#!/usr/bin/env python3
"""
Agentデータ可視化スクリプト

Recorder Agentの記録データを可視化します
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.agents.recorder_agent import recorder_agent


def visualize_business_outcomes():
    """事業成果データを可視化"""
    print("📊 Recorder Agentデータ可視化")

    # データを取得
    outcomes = recorder_agent.get_recent_outcomes(10)
    if not outcomes:
        print("📭 記録データなし")
        return

    print(f"✓ {len(outcomes)}件の記録データを取得")

    # データ処理
    days = []
    sales = []
    efficiencies = []

    for outcome in outcomes.values():
        days.append(outcome.day)
        sales.append(outcome.metrics.sales)
        efficiencies.append(outcome.metrics.inventory_efficiency)

    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 売上推移
    ax1.plot(days, sales, marker="o", color="blue", linewidth=2)
    ax1.set_title("売上推移")
    ax1.set_xlabel("日")
    ax1.set_ylabel("売上 (円)")
    ax1.grid(True, alpha=0.3)

    # 在庫効率推移
    ax2.plot(days, efficiencies, marker="s", color="green", linewidth=2)
    ax2.set_title("在庫効率推移")
    ax2.set_xlabel("日")
    ax2.set_ylabel("効率 (0-1)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("agent_performance_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅ グラフ保存: agent_performance_visualization.png")


def visualize_daily_actions():
    """日次アクションを可視化"""
    print("\n📈 日次アクション可視化")

    outcomes = recorder_agent.get_recent_outcomes(10)
    if not outcomes:
        return

    # アクション統計
    action_counts = {}

    for outcome in outcomes.values():
        for action in outcome.actions_taken:
            action_counts[action] = action_counts.get(action, 0) + 1

    if action_counts:
        # 棒グラフ
        plt.figure(figsize=(10, 6))
        actions = list(action_counts.keys())
        counts = list(action_counts.values())

        sns.barplot(x=counts, y=actions, palette="viridis")
        plt.title("アクション実行頻度", fontsize=16)
        plt.xlabel("実行回数", fontsize=12)
        plt.ylabel("アクション", fontsize=12)

        # 値ラベル表示
        for i, v in enumerate(counts):
            plt.text(v + 0.1, i, str(v), fontsize=10)

        plt.tight_layout()
        plt.savefig("action_frequency_visualization.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("✅ アクショングラフ保存: action_frequency_visualization.png")


if __name__ == "__main__":
    try:
        visualize_business_outcomes()
        visualize_daily_actions()
    except ImportError as e:
        print(f"❌ 可視化ライブラリ不足: {e}")
        print("必要: pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ 可視化エラー: {e}")
