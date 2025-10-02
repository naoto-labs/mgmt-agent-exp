#!/usr/bin/env python3
"""
テストシナリオ設定

統合テストで使用するシナリオデータを一元管理します。
"""

from typing import Any, Dict, List

# ===== 3日間シミュレーション設定 =====
THREE_DAY_SIMULATION_CONFIG = {
    "days": 3,  # シミュレーション日数
    "provider": "openai",  # デフォルトAIプロバイダ
    "expected_decisions_per_day": 3,  # 1日あたりの意思決定数（朝・昼・夕）
    # 販売イベント倍率（時間帯別）
    "daily_sales_multipliers": {
        "morning": 1.8,  # 朝イベント倍率
        "afternoon": 3.0,  # 昼イベント倍率
        "evening": 2.0,  # 夕イベント倍率
    },
    # テスト期待値
    "min_sales_per_day": 0,  # 1日あたりの最小売上
    "expected_simulation_days": 3,  # 完了すべきシミュレーション日数
}


# ===== マルチエージェント統合テスト設定 =====
MULTI_AGENT_CONFIG = {
    "scenario": "high_sales",  # シナリオタイプ
    "days": 3,  # テスト日数
    # 販売イベント倍率（営業時間別）
    "sales_multipliers": {
        "morning_business": 2.5,  # 午前営業倍率
        "afternoon_business": 3.5,  # 午後営業倍率
    },
    # 必須アクションリスト（各エージェントの主要機能）
    "required_actions": [
        "morning_routine",  # Management Agent: 朝ルーチン
        "trend_analysis",  # Analytics Agent: トレンド分析
        "midday_check",  # Management Agent: 昼間チェック
        "product_search",  # Search Agent: 商品検索
        "customer_inquiry",  # Customer Agent: 顧客問い合わせ対応
        "inventory_check",  # Procurement Agent: 在庫確認
        "evening_summary",  # Management Agent: 夕方総括
        "daily_report",  # Analytics Agent: 日次レポート
        "learning_record",  # Recorder Agent: 学習記録
    ],
    # 各Agentのテスト期待値
    "agent_expectations": {
        "search_results_min_count": 0,  # 検索結果最小件数
        "customer_response_required": True,  # 顧客応答生成必須
        "inventory_check_required": True,  # 在庫確認必須
        "daily_report_insights_required": True,  # 日次レポート必須
        "learning_records_required": True,  # 学習記録必須
    },
    # テスト共通設定
    "min_total_sales": 0,  # テスト期間中の最小総売上
}


# ===== 共通テスト設定 =====
COMMON_TEST_CONFIG = {
    "agents": [
        "management_agent",  # SessionBasedManagementAgent
        "analytics_agent",  # AnalyticsAgent
        "recorder_agent",  # RecorderAgent
        "vending_agent",  # VendingAgent
        "search_agent",  # SearchAgent
        "customer_agent",  # CustomerAgent
        "procurement_agent",  # ProcurementAgent
    ],
    "inventory": {
        "setup_initial_slots": True,  # 初期在庫設定
    },
    "timeouts": {
        "max_simulation_time": 300,  # 最大シミュレーション時間（秒）
        "async_operation_timeout": 30,  # 非同期操作タイムアウト
    },
    "logging": {
        "enable_detailed_logging": True,  # 詳細ログ有効化
        "supress_external_logs": [  # 抑制する外部ライブラリログ
            "httpx",
            "httpcore",
            "urllib3",
        ],
    },
    "cleanup": {
        "reset_inventory_after_test": True,  # テスト後の在庫リセット
    },
}


# ===== シナリオ設定取得関数 =====


def get_three_day_config() -> Dict[str, Any]:
    """3日間シミュレーション設定を取得"""
    return THREE_DAY_SIMULATION_CONFIG.copy()


def get_multi_agent_config() -> Dict[str, Any]:
    """マルチエージェントテスト設定を取得"""
    return MULTI_AGENT_CONFIG.copy()


def get_common_config() -> Dict[str, Any]:
    """共通テスト設定を取得"""
    return COMMON_TEST_CONFIG.copy()


def get_test_agents() -> List[str]:
    """テスト対象Agentリストを取得"""
    return COMMON_TEST_CONFIG["agents"].copy()


def get_required_actions() -> List[str]:
    """マルチエージェントテストの必須アクションを取得"""
    return MULTI_AGENT_CONFIG["required_actions"].copy()


# ===== テスト向けユーティリティ関数 =====


def validate_sales_value(sales: float) -> bool:
    """売上値の妥当性を検証"""
    return sales >= COMMON_TEST_CONFIG["min_total_sales"]


def get_simulation_days() -> int:
    """シミュレーション日数を取得"""
    return THREE_DAY_SIMULATION_CONFIG["days"]


def get_expected_decisions_per_day() -> int:
    """1日あたりの期待意思決定数を取得"""
    return THREE_DAY_SIMULATION_CONFIG["expected_decisions_per_day"]


def get_sales_multiplier(time_slot: str) -> float:
    """時間帯別の販売倍率を取得"""
    return THREE_DAY_SIMULATION_CONFIG["daily_sales_multipliers"].get(time_slot, 1.0)
