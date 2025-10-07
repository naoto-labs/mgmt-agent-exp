"""
トレース制御ユーティリティ
環境変数に基づいて選択的にトレースを適用
"""

import functools
import os
from typing import Any, Callable, Optional

from langsmith import traceable


def get_trace_level() -> str:
    """環境変数からトレースレベルを取得"""
    return os.getenv("TRACE_LEVEL", "minimal").lower()


def should_trace_function(func_name: str, trace_level: Optional[str] = None) -> bool:
    """
    関数がトレース対象かどうかを判定

    Args:
        func_name: 関数名
        trace_level: トレースレベル（指定がない場合は環境変数から取得）

    Returns:
        bool: トレース対象の場合True
    """
    if trace_level is None:
        trace_level = get_trace_level()

    # トレースが無効の場合
    if trace_level == "off":
        return False

    # 最小トレースの場合、重要な関数のみトレース
    if trace_level == "minimal":
        critical_functions = {
            "make_strategic_decision",
            "inventory_check_node",
            "sales_plan_node",
            "feedback_node",
        }
        # LLM呼び出し関数もトレース対象に含める
        llm_functions = {
            "analyze_financial_performance",
            "check_inventory_status",
            "generate_response",  # model_managerのLLM呼び出し
        }
        return func_name in critical_functions or func_name in llm_functions

    # 詳細トレースの場合、全関数をトレース
    if trace_level == "detailed":
        return True

    # デフォルトは最小トレース
    return False


def conditional_traceable(name: Optional[str] = None, **kwargs):
    """
    条件付きでトレースを適用するデコレーター

    Args:
        name: トレース名（指定がない場合は関数名を使用）
        **kwargs: traceableデコレーターに渡す追加パラメータ

    Returns:
        デコレーター関数
    """

    def decorator(func: Callable) -> Callable:
        # トレースレベルを確認
        trace_level = get_trace_level()
        func_name = func.__name__

        # トレースが無効の場合、何もせずに関数を返す
        if trace_level == "off":
            return func

        # 関数がトレース対象かどうかを判定
        if not should_trace_function(func_name, trace_level):
            return func

        # トレース対象の場合、@traceableデコレーターを適用
        trace_name = name if name is not None else func_name
        return traceable(name=trace_name, **kwargs)(func)

    return decorator


def trace_enabled() -> bool:
    """トレースが有効かどうかを確認"""
    return get_trace_level() not in ["off", "false"]


def get_trace_config() -> dict:
    """現在のトレース設定を取得"""
    return {
        "trace_level": get_trace_level(),
        "trace_enabled": trace_enabled(),
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
        == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "langsmith_log_level": os.getenv("LANGSMITH_LOG_LEVEL", "WARNING"),
    }
