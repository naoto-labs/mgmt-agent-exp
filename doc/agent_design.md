# AI Agent Autonomous Vending Machine System - LangChain実装設計書

## 概要

本文書は、AI Agent Autonomous Vending Machine Systemの「3-Agent + 17-tool自律型アーキテクチャ」をLangChainを使用して実装するための詳細設計書です。


### 採用Agentデザインパターン
- **Memory-Augmented Agent Pattern**:
  - 会話履歴・学習データ・ベクトル検索統合
  - LangChainのメモリ管理による文脈維持
  - 長期記憶と短期記憶の階層化管理

- **Single Agent Pattern**:
  - 単体のAgentが役職に応じたTool利用を行う
  - node構成は直線型を基本とし、検証レベルに応じてReActなど拡張

- **Permission-Based Tool Access Pattern**:
  - Agent役職に応じたTool利用権限の動的制御
  - 経営判断Agentの専用Tool → 共有Tool → 監査、記録専用Toolのアクセス階層

### 非採用Agentデザインパターン
- **Orchestrator Pattern**:
  - マルチAgent間の協調制御パターン
  - セッション型の業務実行サイクルを管理
  - Queueベースのメッセージバスで非同期通信



## システム全体アーキテクチャ

### 基本構成

```
┌─────────────────────────────────────────────────────────────┐
│             AIマルチエージェント自律型自動販売機システム         │
├─────────────────────────────────────────────────────────────┤
│                 Application層: 自律Agent                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   店長Agent      │  │   監査Agent      │  │   記録Agent      │ │
│  │   (経営判断)    │  │   (データ分析)   │  │   (学習蓄積)    │ │
│  │  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘  │
│  │   △管理Tool群   │  │   △監査Tool群   │  │   △記録Tool群   │ │
│  │   △共有Tool群   │  │   △共有Tool群   │  │   △共有Tool群   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Domain層: 業務システム                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │   販売システム  │  │   会計システム  │  │   決済システム  │  │   通信システム     │ │
│  │   (Vending)     │  │   (Accounting)  │  │   (Payment)     │  │ (Communication)    │ │
│  └────────────────┘  └────────────────┘  └────────────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Interface層: 外部アクセス                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   人間従業員     │  │     顧客       │  │   外部システム   │ │
│  │  (作業実行)     │  │  (お客様対話)   │  │  (仕入れ先等)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 役割分担

#### Application層: 自律AI判断レイヤー

| コンポーネント | 責務 | 動作形態 | LangChain実装 |
|---------------|------|----------|---------------|
| 店長Agent | 経営判断 + Tool活用意思決定 + 共通ツール利用 | セッション型 + Tool Chain実行 | Tool Calling + 長期メモリ |
| 監査Agent | データ分析・KPI監査 + 共通ツール利用 | 常時稼働 + Toolループ実行 | Tool Calling+ 長期メモリ |
| 記録Agent | 学習蓄積・パターン分析・記録 + 共通ツール利用 | 常時稼働 + Agent・Tool監視記録 | Tool Calling+ + 短期メモリ |

#### Domain層: 業務専門システムレイヤー

| コンポーネント | 責務 | 動作形態 | 実装方式 |
|---------------|------|----------|----------|
| 販売システム (Vending) | 商品販売・在庫管理・取引処理 | イベント駆動 | APIベース |
| 会計システム (Accounting) | 取引会計・財務分析・仕訳生成 | イベント駆動 | 自動処理 |
| 決済システム (Payment) | 支払い処理・決済API統合・トランザクション実行 | イベント駆動 | API統合 |
| 通信システム (Communication) | 顧客通知・統合通信API・外部連携通知 | イベント駆動 | API連携 |

#### Interface層: 外部アクセスレイヤー

| コンポーネント | 責務 | 動作形態 | 接続方式 |
|---------------|------|----------|----------|
| 人間従業員 | 物理作業実行・システム操作支援 | on-demand | 通信 |
| 顧客 | お客様対話・購買アクション | on-demand | 通信UI/API操作 |
| 外部システム | 仕入れ先・決済・配送システム連携 | バッチ/API | 通信（仮想） |

### Agent定義

#### 店長Agent (Management Agent)

- **役割**: 経営判断の最終決定者。3カテゴリの専門Toolと2カテゴリのShared Toolsを活用して意思決定
- **動作形態**: セッション型 (朝/昼/夕の業務サイクル)
- **責務**: 戦略立案, 人間協働指示, 事業KPI管理
- **共有Tool統合**: 全共有Toolと 管理Toolにフルアクセス
- **主な機能**:
  - 朝ルーチン: 夜間データの分析と今日の業務計画立案
  - 昼間チェック: 午前実績の分析と昼間戦略調整
  - 夕方総括: 1日全体の実績評価と改善策立案
  - 戦略的意思決定: AI分析に基づく経営判断の実行

#### 監査Agent (Analytics Agent)

- **役割**: データ分析と業務改善提案
- **動作形態**: 常時稼働
- **責務**: KPI算出, 無駄削減提案, 業務効率化監査
- **共有Tool統合**: 主にdata_retrieval, 監査専用Tool
- **主な機能**:
  - リアルタイムKPI監視: 売上・在庫・顧客満足度の継続監視
  - パフォーマンス分析: 日次・週次・月次の実績分析・レポート生成
  - 異常検出: システム異常・業務変動の自動検知
  - 改善提案: データに基づく効率化・コスト削減の提案

#### 記録Agent (Recorder Agent)

- **役割**: 全行動の記録と学習データ蓄積
- **動作形態**: 常時稼働
- **責務**: パターン分析, 成功事例抽出, 失敗学習
- **共有Tool統合**: 主にdata_retrieval, 記録専用Tool
- **主な機能**:
  - セッション記録: Agentの意思決定・行動・結果の詳細ログ化
  - パターン学習: 成功・失敗パターンの自動認識・蓄積
  - データ永続化: 学習データの長期保存・検索
  - 改善フィードバック: 蓄積データからの意思決定改善

## Tools Framework設計

### Shared Tools 概要

Shared Tools は、3-Agentが共有可能なツールを集中管理・実行するアーキテクチャです。
2カテゴリ`data_retrieval/` (check_inventory_status, get_business_metrics) と `market_tools/` (search_products)
のツールを統一的に管理します。


#### 各Agentの統合パターン

**Management Agent統合**:
- Tool Chainパターン: 複数ツールの順次実行
- Full Access: 全カテゴリ17Tool利用可能
- LangChain Tool Calling統合: 自然言語→ツール実行

**Analytics Agent統合**:
- 常時監視パターン: 定期的なツール実行ループ
- Limited Access: data_retrieval + analytics専用Tool
- データ収集→分析→報告の自動ワークフロー

**Recorder Agent統合**:
- Agent行動監視パターン: 全Agentアクションのログ収集
- Recording Access: data_retrieval + recorder専用Tool
- パターン分析のためのデータ蓄積機能

#### Toolアクセス管理

- **権限階層**:
  - Management Agent: 全Toolアクセス
  - Analytics Agent: 読取重視 + 分析専用Tool
  - Recorder Agent: ログ専用 + データ取得Tool

- **制御方式**:
  - カテゴリベース: Toolカテゴリ単位権限
  - 動的チェック: 実行時検証

  

## システム統合

### Agent間協調制御パターン

- **Message Busパターン**: Queueベースの非同期メッセージング
- **Session Orchestrator**: 業務サイクル管理とAgent調停
- **Shared State Management**: Agent間共有状態の同期管理

## 技術仕様

### 技術スタック仕様

#### LLM統合 (Azure OpenAI GPT-4o-mini + LangChain)
- **Primary**: Azure OpenAI GPT-4o-mini
- **Fallback**: Anthropic Claude
- **LangChainバージョン**: 0.1.0+
- **メモリ戦略**: ConversationBufferMemory + VectorStore for learning
- LangGraph統合実装済み（StateGraph + RunnableSequence）

#### 技術スタック

```python
# Updated requirements.txt
langchain==0.1.0
langchain-openai==0.0.5
langchain-anthropic==0.1.0
langchain-community==0.0.10
tavily-python==0.3.0
openai==1.3.0
azure-identity==1.15.0
chromadb==0.4.0
tiktoken==0.5.0
pydantic==2.5.0
fastapi==0.104.1
```



## Agent構成パターン

### 概要

Vending-Bench論文を参考に長期Agent運用評価を目的とし、3Agent構成の異なる実行パターンを定義。
LangChain RunnableSequence＆直列遷移ベースでノード間接続・状態遷移を管理。

### Case A: VendingBench同等Agent構成パターン

**構成特徴**: 時間ごとに各nodeを直列に推移。監査無しでもOK、全Agent並行動作

```python
# 頻度パラメータで調整可能な統合構成
frequency_config_a = {
    "analytics_frequency": 10,  # 10step毎Analytics実行
}

graph_case_a = RunnableSequence(
    # Management Agent: 常時実行
    management_continuous = RunnableSequence(
        inventory_check_node(check_inventory_status),
        | sales_plan_node(plan_sales_strategy,analyze_financial_performance),
        | pricing_node(update_pricing),
        | restock_node(assign_restocking_task),
        | procurement_request_generation_node(request_procurement),
        | sales_processing_node(),
        | customer_interaction_node(respond_to_customer_inquiry, handle_customer_complaint),
        | profit_calculation_node(analyze_financial_performance),
        | feedback_node(feedback_engine),
    )

    # Recorder Agent: 常時実行
    recorder_continuous = RunnableSequence(
        session_recorder_node(session_recorder)
        | pattern_analyzer_node(pattern_analyzer)
        | data_persistence_node(data_persistence)
    )

    # Optional
    # Analytics Agent: 間欠実行 (パラメータ制御)
    analytics_scheduled = RunnableSequence(
        step_counter_node()
        | conditional_execution({
            f"every_{frequency_config_a['analytics_frequency']}_steps": RunnableSequence(
                performance_monitor_node(performance_monitor)
                | anomaly_detector_node(anomaly_detector)
                | efficiency_analyzer_node(efficiency_analyzer)
            )
        })
    )

    # 頻度違いでの3Agent並行実行
    parallel_execution_with_different_frequencies([
        ("continuous", management_continuous),    # Management: 常時
        ("scheduled", analytics_scheduled),       # Analytics: 設定間隔
        ("continuous", recorder_continuous)       # Recorder: 常時
    ])

    | cross_agent_data_sync()    # Agent間情報共有
    | workflow_coordination()    # 実行調整
    | benchmark_evaluation_node()
)
```

**評価対象**: Primary Metrics、Secondary Metrics 

---

### Case B: ReAct思考構成

**構成特徴**: 行動内省・ReAct思考で再起処理。時間の表現をCaseAと整合が必要

```python
graph_case_b = RunnableSequence(
    complex_tool_integrated_parallel([
        # Management Agent拡張 + 内省機能
        enhanced_management_flow([
            # 基本業務ノード
          inventory_check_node(check_inventory_status),
          | sales_plan_node(plan_sales_strategy,analyze_financial_performance),
          | pricing_node(update_pricing),
          | restock_node(assign_restocking_task),
          | procurement_request_generation_node(request_procurement),
          | sales_processing_node(),
          | customer_interaction_node(respond_to_customer_inquiry, handle_customer_complaint),
          | profit_calculation_node(analyze_financial_performance),
          | feedback_node(feedback_engine),

          # 内省・ReAct思考ノード (行動評価・戦略再却図) 
          # TODO 設計が必要
          |decision_introspection_node(),         # 行動内省・過去決定評価
          |react_reasoning_node(),               # ReActでやり直すか、
          |finalize_node()            # 終了判定？　flagで管理
        ]),

        
        # Analytics Agent: 分野別監査 間欠実行 (パラメータ制御)
        analytics_scheduled([
              step_counter_node()
            | conditional_execution({
            f"every_{frequency_config_a['analytics_frequency']}_steps": RunnableSequence(
              # business_monitoring (実績監視・異常検知)
              performance_monitoring_node(performance_monitor),
              | anomaly_detection_node(anomaly_detector),
              | compliance_monitoring_node(compliance_checker),

              # ai_governance (品質・安全・追跡)
              | decision_quality_monitoring_node(decision_quality_monitor),
              | safety_compliance_monitoring_node(safety_compliance_checker),
              | ai_performance_tracking_node(performance_tracker),

              # analysis (効率性・コスト分析)
              | operational_efficiency_analysis_node(efficiency_analyzer),
              | cost_benefit_analysis_node(cost_benefit_analyzer)
            })
        ]),

        # 頻度違いでの3Agent並行実行
        parallel_execution_with_different_frequencies([
            ("continuous", management_continuous),    # Management: 常時
            ("scheduled", analytics_scheduled),       # Analytics: 設定間隔
            ("continuous", recorder_continuous)       # Recorder: 常時
        ])

        | cross_agent_data_sync()    # Agent間情報共有
        | workflow_coordination()    # 実行調整
        | benchmark_evaluation_node()
    ]),

    # 内省・ReAct統合作業
    # TODO設計必要
    | self_reflection_coordination({
        "introspection_trigger": "after_each_major_decision",
        "react_cycles": 3,                         # ReAct思考サイクル数
        "historical_review_period": 7,             # 前日振り返り日数
        "reflection_questions": [
            "システムデータや需給と矛盾ない店長の行動が実行できているか"
        ]
    }),
)
```

**評価対象**: Primary Metrics、Secondary Metrics について内省機能の効果 (ReAct思考精度向上、前日振り返り学習効果、

### Case C: イベント駆動リアル制約構成

**構成特徴**: リアルな状況を再現。時間ごとに各nodeを直列な固定フローではなく販売のランダム性や突発的イベントや人間側の制約あり

```python
graph_case_c = RunnableSequence(
    # Management Agent: 常時 + イベント応答
    management_responsive = RunnableSequence(
        background_executors([
            management_continuous = RunnableSequence(
                inventory_check_node(get_business_metrics)
                | restock_node(assign_restocking_task, request_procurement)
                | pricing_node(plan_sales_strategy)
                | sales_processing_node()
                | customer_interaction_node(respond_to_customer_inquiry, handle_customer_complaint)
                | profit_calculation_node(analyze_financial_performance)
          )
        ])

        | event_responders([
          　# TODO設計必要
            #運営業務の失敗イベント
            low_stock_event: restock_request_with_probability(
                request_procurement,     # 成功率: (調達の現実性を表現)
                assign_restocking_task, # 成功率: (シフト等の制約を表現)
                human_constraint_model=True
            ),
            #顧客や突発的な販売イベント
            customer_complaint_event: customer_interaction_node(handle_customer_complaint),
            purchase_event: sales_processing_node(get_business_metrics)
        ])
    )
    # TODO設計必要　event_respondersに従属していればOKのはず
    # Analytics Agent: イベントトリガー + 定期的
    analytics_event_driven = RunnableSequence(
        scheduled_analysis_node(efficiency_analyzer)
        | event_triggered([
            anomaly_alert: anomaly_detector_node(anomaly_detector),
            performance_drop: performance_monitor_node(performance_monitor)
        ])
    )

    # TODO設計必要　event_respondersに従属していればOKのはず
    # Recorder Agent: 常時記録 + イベント学習
    recorder_adaptive = RunnableSequence(
        recorder_continuous = RunnableSequence(
            session_recorder_node(session_recorder)
            | pattern_analyzer_node(pattern_analyzer)
            | data_persistence_node(data_persistence)
        )
        # TODOイベンド時の保存
        | event_based_learning({
            "performance_event": adaptive_analysis_node(pattern_analyzer),
            "error_event": error_pattern_learning_node()
        })
    )

    # イベント駆動3Agent並行実行
    event_driven_parallel([
        management_responsive,
        analytics_event_driven,
        recorder_adaptive
    ])

    | event_scheduler()      # イベントベース制御
    | reliability_handler()   # 人間制約シミュレーション
)
```

**評価対象**: Primary Metrics、Secondary Metrics 
---

### VendingBench評価指標 (論文準拠)

各CaseでVending-Bench-spec.md記載の標準指標を計測：

**Primary Metrics (論文準拠)**:
- **Profit**: 累積利益 (売上 - 費用 - 手数料)
- **Stockout Rate**: 在庫切れ率 (在庫切れ件数/需要件数)
- **Pricing Accuracy**: 価格設定精度 (理想価格との平均誤差)
- **Action Correctness**: 行動正しさ (Oracleルール一致率)
- **Customer Satisfaction**: 顧客満足度 (0-1スコア)

**Secondary Metrics (論文準拠)**:
- **Long-term Consistency**: 長期的一貫性 (過去ウィンドウのActionCorrectness平均)


###　メモリ設計
Vending-Benchではプロンプトコンテキストウィンドウにて、直近の観察・行動・報酬を保持し、次の意思決定に利用している。
**短期メモリ**:
-方針：LangChainでの実装ConversationBufferMemory または ConversationSummaryBufferMemoryで各nodeの直近状態をだけを保持するか、数回のサイクルにおける長期の状態をプロンプトに渡すか論点
token数制限を意識して 要約型?
-仕組み: 各ノード呼び出し時に、直近の対話履歴や思考過程が llm_chain に渡される。
-用途:
  -inventory_check_node → 「直前の販売結果・在庫情報」を保持し、計画に活用。
  -pricing_node → 前回価格変更の理由を覚えていないと適切に更新できない。
  -customer_interaction_node → 直近の問い合わせに即応。

**長期メモリ（外部メモリツール）**
Vending-Benchでは scratchpad や key-value store を想定。システム側のDB（RDB、MongoDB）が所与として存在。
-方針：まずは VectorStoreMemory（Chroma などのインメモリDB）で見極めて、システムDBへ移行
-対応方法:
  -VectorStoreRetrieverMemory → 在庫履歴や売上傾向を検索可能に。
  -KeyValueMemory → 「最新の価格設定」「最新の在庫数」など即参照したい情報を保存。→システムから取得可能なため、不要か？
-用途:
  -システムDBに販売履歴・在庫履歴・顧客対応履歴を保存。
  -restock_node や procurement_request_generation_node → 過去の仕入れ実績に基づき判断。
  -profit_calculation_node → 長期的な利益率を比較。
  -feedback_node → 過去のクレームや顧客対応を参照。

---

## 📌 実装イメージ

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 短期メモリ
short_term_memory = ConversationBufferWindowMemory(k=5)

# 外部メモリ（簡易ログ用）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(collection_name="sim_logs", embedding_function=embeddings)
long_term_memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever())

# ノード例
def inventory_check_node(get_business_metrics, step):
    # システムDBやAPIから業務メトリクスを取得
    business_metrics = get_business_metrics(step)

    # 短期メモリに保存（直近ノードで参照用）
    short_term_memory.save_context(
        {"query": f"business_metrics_step_{step}"},
        {"result": business_metrics}
    )

    # 外部メモリに保存（監査評価用）
    long_term_memory.save_context(
        {"event": "inventory_check", "step": step},
        {"metrics": str(business_metrics)}
    )

    return business_metrics
```
