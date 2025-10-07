"""
ログ設定モジュール
LangSmithトレースとアプリケーションログの設定を一元管理
"""

import logging
import logging.config
import os
import sys
from typing import Any, Dict


def get_trace_level() -> str:
    """環境変数からトレースレベルを取得"""
    return os.getenv("TRACE_LEVEL", "minimal").lower()


def get_log_level() -> str:
    """環境変数からログレベルを取得"""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def get_langsmith_log_level() -> str:
    """環境変数からLangSmithログレベルを取得"""
    return os.getenv("LANGSMITH_LOG_LEVEL", "WARNING").upper()


def setup_logging():
    """ログ設定を初期化"""
    trace_level = get_trace_level()
    log_level = get_log_level()
    langsmith_log_level = get_langsmith_log_level()

    # ログ設定辞書
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # LangSmith関連ログの設定
            "langsmith": {
                "level": langsmith_log_level,
                "handlers": ["console"] if trace_level == "detailed" else [],
                "propagate": False,
            },
            # LangChain関連ログの設定
            "langchain": {
                "level": langsmith_log_level,
                "handlers": ["console"] if trace_level == "detailed" else [],
                "propagate": False,
            },
            # HTTPクライアントログの抑制
            "httpx": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            "httpcore": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            "urllib3": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            # ビジネスロジックログの設定
            "src.agents.management_agent": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "src.infrastructure.ai.model_manager": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # その他のアプリケーションログ
            "src": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file"],
        },
    }

    # トレースレベルに応じてログ設定を調整
    if trace_level == "minimal":
        # 最小トレースの場合、LangSmithログをファイルのみに出力
        log_config["handlers"]["langsmith_file"] = {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "logs/langsmith_trace.json",
            "encoding": "utf-8",
        }
        log_config["loggers"]["langsmith"]["handlers"] = ["langsmith_file"]
        log_config["loggers"]["langchain"]["handlers"] = ["langsmith_file"]

    elif trace_level == "detailed":
        # 詳細トレースの場合、コンソールにも出力
        log_config["handlers"]["langsmith_console"] = {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": sys.stdout,
        }
        log_config["loggers"]["langsmith"]["handlers"] = ["langsmith_console"]
        log_config["loggers"]["langchain"]["handlers"] = ["langsmith_console"]

    # ログディレクトリの作成
    os.makedirs("logs", exist_ok=True)

    # ログ設定を適用
    logging.config.dictConfig(log_config)

    # 初期化メッセージ
    logger = logging.getLogger(__name__)
    logger.info(
        f"ログ設定を初期化しました (trace_level={trace_level}, log_level={log_level})"
    )


def get_logger(name: str) -> logging.Logger:
    """名前付きロガーを取得"""
    return logging.getLogger(name)


def configure_langsmith_tracing():
    """LangSmithトレースの設定"""
    trace_level = get_trace_level()

    # トレースレベルの設定
    if trace_level == "off":
        # トレースを完全に無効化
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    elif trace_level == "minimal":
        # 最小トレース（重要な関数のみ）
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # トレースする関数を制限するための設定を追加予定
    elif trace_level == "detailed":
        # 詳細トレース（すべての関数）
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    logger = logging.getLogger(__name__)
    logger.info(f"LangSmithトレースを設定しました (level={trace_level})")
