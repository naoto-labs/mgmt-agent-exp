"""
VendingBench Metrics 設定
Primary/Secondary Metricsの目標値と定義を管理
"""

VENDING_BENCH_METRICS = {
    "primary_metrics": {
        "profit": {
            "target": 100000,  # 月間目標利益（円）
            "description": "月間利益目標",
            "unit": "円",
            "direction": "maximize",  # 最大化目標
        },
        "stockout_rate": {
            "target": 0.1,  # 在庫切れ率（10%）
            "description": "在庫切れ率",
            "unit": "%",
            "direction": "minimize",  # 最小化目標
        },
        "pricing_accuracy": {
            "target": 0.8,  # 価格設定精度（80%）
            "description": "価格設定精度",
            "unit": "%",
            "direction": "maximize",  # 最大化目標
        },
        "action_correctness": {
            "target": 0.7,  # アクション正しさ（70%）
            "description": "アクション正しさ",
            "unit": "%",
            "direction": "maximize",  # 最大化目標
        },
        "customer_satisfaction": {
            "target": 3.5,  # 顧客満足度（3.5/5.0）
            "description": "顧客満足度",
            "unit": "/5.0",
            "direction": "maximize",  # 最大化目標
        },
    },
    "secondary_metrics": {
        "long_term_consistency": {
            "target": 0.75,  # 長期的一貫性（75%）
            "description": "長期的一貫性",
            "unit": "%",
            "direction": "maximize",  # 最大化目標
        },
    },
}

# 難易度ごとの目標値プリセット
DIFFICULTY_PRESETS = {
    "easy": {  # 緩い目標（学習・実験用）
        "primary_metrics": {
            "profit": {"target": 50000},
            "stockout_rate": {"target": 0.2},
            "pricing_accuracy": {"target": 0.6},
            "action_correctness": {"target": 0.5},
            "customer_satisfaction": {"target": 3.0},
        },
        "secondary_metrics": {
            "long_term_consistency": {"target": 0.5},
        },
    },
    "normal": {  # 標準目標（VendingBench準拠）
        "primary_metrics": {
            "profit": {"target": 1000000},
            "stockout_rate": {"target": 0.1},
            "pricing_accuracy": {"target": 0.8},
            "action_correctness": {"target": 0.7},
            "customer_satisfaction": {"target": 3.5},
        },
        "secondary_metrics": {
            "long_term_consistency": {"target": 0.75},
        },
    },
    "hard": {  # 厳しい目標（上級チャレンジ）
        "primary_metrics": {
            "profit": {"target": 150000},
            "stockout_rate": {"target": 0.05},
            "pricing_accuracy": {"target": 0.9},
            "action_correctness": {"target": 0.8},
            "customer_satisfaction": {"target": 4.0},
        },
        "secondary_metrics": {
            "long_term_consistency": {"target": 0.85},
        },
    },
}


def get_metrics_targets(difficulty: str = "normal") -> dict:
    """
    指定した難易度の目標値を取得

    Args:
        difficulty: "easy" | "normal" | "hard"

    Returns:
        メトリクス目標値辞書
    """
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unknown difficulty: {difficulty}. Choose from {list(DIFFICULTY_PRESETS.keys())}"
        )

    return DIFFICULTY_PRESETS[difficulty]


def format_metric_value(metric_name: str, value: float) -> str:
    """
    メトリクス名に応じた適切なフォーマットで値を文字列化

    Args:
        metric_name: メトリクス名
        value: 値

    Returns:
        フォーマットされた文字列
    """
    if metric_name == "profit":
        return ",.0f"
    elif metric_name in [
        "stockout_rate",
        "pricing_accuracy",
        "action_correctness",
        "long_term_consistency",
    ]:
        return ".1%"
    elif metric_name == "customer_satisfaction":
        return ".1f"
    else:
        return "g"  # 一般形式
