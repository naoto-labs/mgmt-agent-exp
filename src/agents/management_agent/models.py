"""
models.py - 経営管理エージェントのデータモデル

管理業務に必要なPydanticモデルクラスを定義
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BusinessMetrics(BaseModel):
    """事業メトリクス"""

    sales: float
    profit_margin: float
    inventory_level: Dict[str, int]
    customer_satisfaction: float
    timestamp: datetime


class ManagementState(BaseModel):
    """Management Agentの完全な状態管理クラス (VendingBench準拠・Multi-day運用対応)"""

    # ===== セッション管理 =====
    session_id: str = Field(description="セッション固有ID")
    session_type: str = Field(
        description="セッションタイプ (management_flow, node_based_managementなど)"
    )

    # ===== 日時・期間管理 =====
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="セッション開始日時 (ログ保存・メモリ用)",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="最終更新日時"
    )
    business_date: date = Field(
        default_factory=date.today, description="事業日 (営業日単位)"
    )
    day_sequence: int = Field(default=1, description="連続稼働日数 (1日目、2日目...)")

    # ===== ビジネスデータ入力 =====
    business_metrics: Optional[BusinessMetrics] = Field(
        default=None, description="売上、利益、在庫、顧客満足度の基本指標"
    )

    # 分析フェーズの出力
    inventory_analysis: Optional[Dict] = Field(
        default=None,
        description="在庫状況の詳細分析結果（ステータス、アラート、再発注推奨）",
    )

    sales_analysis: Optional[Dict] = Field(
        default=None, description="売上・財務パフォーマンス分析（トレンド、戦略推奨）"
    )

    financial_analysis: Optional[Dict] = Field(
        default=None, description="詳細財務分析結果"
    )

    sales_processing: Optional[Dict] = Field(
        default=None, description="売上処理・販売効率分析結果"
    )

    profit_calculation: Optional[Dict] = Field(
        default=None, description="利益計算・財務健全性詳細分析結果"
    )

    # 戦略決定フェーズ
    pricing_decision: Optional[Dict] = Field(
        default=None, description="価格戦略決定（価格変更、新価格、理由）"
    )

    restock_decision: Optional[Dict] = Field(
        default=None, description="補充タスク決定（製品リスト、タスクID、緊急度）"
    )

    procurement_decision: Optional[Dict] = Field(
        default=None, description="調達依頼決定（製品、数量、発注情報）"
    )

    # 顧客対応
    customer_interaction: Optional[Dict] = Field(
        default=None, description="顧客対応結果（フィードバック、新規キャンペーン）"
    )

    # 実行履歴
    executed_actions: List[Dict] = Field(
        default_factory=list, description="実行済みアクションの履歴"
    )

    # 状態管理
    current_step: str = Field(
        default="initialization", description="現在の処理ステップ"
    )

    processing_status: str = Field(
        default="pending",
        description="全体処理ステータス (pending, processing, completed, error)",
    )

    # エラーハンドリング
    errors: List[str] = Field(
        default_factory=list, description="発生したエラーメッセージ一覧"
    )

    # ===== メモリ連携フィールド (ConversationBufferWindowMemory + VectorStore連携) ====
    # TODO VectorStore未作成
    memory_snapshot: Optional[Dict] = Field(
        default=None, description="直近の会話履歴スナップショット（短期メモリ）"
    )
    learned_patterns: Optional[Dict] = Field(
        default=None, description="VectorStoreからの学習パターン（長期メモリ）"
    )
    historical_insights: List[Dict] = Field(
        default_factory=list,
        description="過去データからの洞察（売上傾向、在庫パターン等）",
    )

    # ===== Multi-day運用フィールド =====
    previous_day_carry_over: Optional[Dict] = Field(
        default=None, description="前日のfinal_reportデータ引き継ぎ"
    )
    cumulative_kpis: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_profit": 0,
            "average_stockout_rate": 0.0,
            "customer_satisfaction_trend": [],
            "action_accuracy_history": [],
        },
        description="全稼働期間の累積KPI（VendingBench Secondary Metrics用）",
    )

    # ===== イベント駆動対応フィールド (Case C向けしばらく未使用) =====
    external_events: List[Dict] = Field(
        default_factory=list, description="人間による制約、突発イベントの履歴"
    )
    agent_communications: List[Dict] = Field(
        default_factory=list, description="他のAgentとのメッセージ交換（Queueベース）"
    )
    pending_human_tasks: List[Dict] = Field(
        default_factory=list, description="人間従業員待ちのタスク（補充、調達依頼等）"
    )

    # ===== ベンチマーク評価フィールド =====
    primary_metrics_history: List[Dict] = Field(
        default_factory=list, description="各実行回のProfit, StockoutRate等の履歴"
    )
    consistency_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="長期的一貫性評価データ"
    )

    # 最終出力
    feedback: Optional[Dict] = Field(
        default=None, description="最終フィードバックと要約"
    )
    final_report: Optional[Dict] = Field(default=None, description="最終総合レポート")


class SessionInfo(BaseModel):
    """セッション情報"""

    session_id: str
    session_type: str  # "morning_routine", "midday_check", "evening_summary"
    start_time: datetime
    end_time: Optional[datetime] = None
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    actions_executed: List[Dict[str, Any]] = Field(default_factory=list)
