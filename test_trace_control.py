#!/usr/bin/env python3
"""
トレース制御システムのテストスクリプト
LangSmithトレースの設定と動作を確認
"""

import logging
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.shared.config.logging_config import get_trace_config, setup_logging
from src.shared.utils.trace_control import (
    conditional_traceable,
    get_trace_level,
    should_trace_function,
    trace_enabled,
)
from src.shared.utils.trace_control import get_trace_config as get_trace_config_utils


def test_trace_control():
    """トレース制御システムのテスト"""
    print("=== トレース制御システムテスト ===\n")

    # 現在のトレース設定を表示
    print("現在のトレース設定:")
    config = get_trace_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nトレースレベル: {get_trace_level()}")
    print(f"トレース有効: {trace_enabled()}")

    # 関数別トレース判定テスト
    print("\n=== 関数別トレース判定テスト ===")
    test_functions = [
        "make_strategic_decision",
        "inventory_check_node",
        "sales_plan_node",
        "feedback_node",
        "pricing_node",
        "restock_node",
        "unknown_function",
    ]

    for func_name in test_functions:
        should_trace = should_trace_function(func_name)
        print(
            f"  {func_name}: {'✅ トレース対象' if should_trace else '❌ トレース対象外'}"
        )

    # ログ設定のテスト
    print("\n=== ログ設定テスト ===")
    try:
        setup_logging()
        logger = logging.getLogger("test_logger")
        logger.info("ログ設定テストメッセージ")
        print("✅ ログ設定が正常に初期化されました")
    except Exception as e:
        print(f"❌ ログ設定エラー: {e}")

    print("\n=== テスト完了 ===")


@conditional_traceable(name="test_function")
def test_traced_function():
    """テスト用のトレース対象関数"""
    print("テスト用のトレース対象関数が実行されました")
    return "テスト結果"


def test_conditional_traceable():
    """条件付きトレースデコレーターのテスト"""
    print("\n=== 条件付きトレースデコレーター試験 ===")

    try:
        result = test_traced_function()
        print(f"関数実行結果: {result}")
        print("✅ 条件付きトレースデコレーターが正常に動作しました")
    except Exception as e:
        print(f"❌ 条件付きトレースデコレーターエラー: {e}")


if __name__ == "__main__":
    test_trace_control()
    test_conditional_traceable()
