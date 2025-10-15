# AI Agent Autonomous Vending Machine System 設計書

## 概要

AI Agent Autonomous Vending Machine Systemは、3-Agent + XX個-toolのクリーンアーキテクチャによる完全自律型自動販売シミュレーションシステムです。


## アーキテクチャ概要


```
ai-vending-system/
├── agents/                      # 🚀 自律Agent（主要3つ + 共有Tool）
│   ├── management_agent/        # 店長Agent
│   │   ├── management_tools/    # 経営判断Tool
│   │   │   ├── plan_agent_operations.py
│   │   │   ├── plan_sales_strategy.py
│   │   │   ├── update_pricing.py
│   │   │   ├── analyze_financial_performance.py
│   │   │   ├── feedback_engine.py
│   │   ├── procurement_tools/
│   │   │   ├── request_procurement.py
│   │   │   ├── assign_restocking_task.py
│   │   │   └── coordinate_employee_tasks.py
│   │   │   └── calculate_optimal_order.py
│   │   ├── customer_tools/
│   │   │   ├── respond_to_customer_inquiry.py
│   │   │   ├── handle_customer_complaint.py
│   │   └── orchestrator.py
│   ├── analytics_agent/         # 監査Agent
│   │   ├── business_monitoring/ # ビジネス監視
│   │   │   ├── performance_monitor.py
│   │   │   ├── anomaly_detector.py
│   │   │   └── compliance_checker.py
│   │   ├── ai_governance/       # AIガバナンス
│   │   │   ├── decision_quality_monitor.py
│   │   │   ├── safety_compliance_checker.py
│   │   │   └── performance_tracker.py
│   │   ├── advisory/            # アドバイス業務
│   │   │   ├── efficiency_analyzer.py
│   │   │   └── cost_benefit_analyzer.py
│   │   └── orchestrator.py
│   ├── recorder_agent/          # 記録Agent
│   │   ├── learning_tools/
│   │   │   ├── session_recorder.py
│   │   │   ├── data_persistence.py
│   │   │   ├── pattern_analyzer.py
│   │   └── orchestrator.py
│   ├── shared_tools/            # 🔧 共有Tool 
│   │   ├── data_retrieval/
│   │   │   ├── check_inventory_status.py
│   │   │   ├── get_business_metrics.py
│   │   │   └── collect_customer_feedback.py
│   │   └── market_tools/
│   │       ├── search_products.py
│   │       ├── supplier_research.py
│   └── orchestrator/            # 全Agent協調制御
├── domain/                      # ビジネスルール・モデル
├── infrastructure/              # 技術実装・API・DB
├── application/                 # アプリケーションロジック
├── shared/                      # 全層共通ユーティリティ
├── tests/                       # 統合テスト
├── docs/                        # 設計・API仕様・ガイド
├── static/                      # Web資産
├── scripts/                     # ビルド・運用
└── pyproject.toml               # パッケージ管理
```


### 3-Agent + tool共有アーキテクチャ

- **定義**: 店長Agent(経営判断)・監査Agent(データ分析)・記録Agent(学習蓄積)の3つの独立Agentが、必要なToolを使用して、業務を遂行
- **特徴**: toolは店長Agent主導設計だが、監査Agent・記録Agentもシステム情報取得時に共有Tollからシステムデータへアクセス可能
- **目的**: 店長Agentによる自律的運営の実現。記録AgentがAgentの長期タスク実行を補助し、監査Agentは独立して店長Agentを監督

## Agent定義

### 店長Agent (Management Agent)

- **役割**: 経営判断の最終決定者。3つの専用カテゴリと２つの共通カテゴリのtoolを活用して意思決定
- **動作形態**: セッション型 時間ごとの業務サイクル)
- **責務**: 戦略立案, 人間協働指示, 事業KPI管理
- **主な機能**:
概念的には以下。Langchain,graph nodeで時間を表現
  - 朝ルーチン: 前日データの分析と今日の業務計画立案
  - 昼間チェック: 午前実績の分析と昼間戦略調整
  - 夕方総括: 1日全体の実績評価と改善策立案
  - 戦略的意思決定: AI分析に基づく経営判断の実行
- **実装状況**: セッションベースの意思決定システム実装済み

### 監査Agent (Analytics Agent)

- **役割**: データ分析と店長を監査、業務改善提案
- **動作形態**: 指定した頻度で稼働
- **責務**: KPI算出, 無駄削減提案, 業務監査
- **主な機能**:
  - リアルタイムKPI監視: 売上・在庫・顧客満足度の継続監視
  - パフォーマンス分析: 日次・週次・月次の実績分析・レポート生成
  - 異常検出: システム異常・業務変動の自動検知
  - 改善提案: データに基づく効率化・コスト削減の提案
- **実装状況**: 分析フレームワーク実装済み、実際データ連携機能は作成中

### 記録Agent (Recorder Agent)

- **役割**: 全行動の記録と学習データ蓄積
- **動作形態**: 常時稼働
- **責務**: 店長の行動と在庫、実績の客観的な記録、パターン分析, 成功事例抽出, 失敗学習
- **主な機能**:
  - セッション記録: Agentの意思決定・行動・結果の詳細ログ化
  - パターン学習: 成功・失敗パターンの自動認識・蓄積
  - データ永続化: 学習データの長期保存・検索
- **実装状況**: 記録機能実装済み、学習・分析機能は作成中

## Tool定義

以下のTool定義は、アーキテクチャ概要のフォルダ構成と対応しています。各AgentのToolを整理分類して定義します。

### Management Agent Tools (経営判断特化)

**management_agent/management_tools/**: 店長Agentの戦略的意思決定を支援するTool群
- **plan_agent_operations.py**: ❌ 未完成(拡張案) - 日次・週次業務計画の立案・実行手順策定。
- **plan_sales_strategy.py**: ❌ 未完成 - 売上目標設定・プロモーション戦略立案。
- **analyze_financial_performance.py**: ✅ 実装済み - 財務データ分析、収益性評価、改善提案。
- **update_pricing.py**: ✅ 実装済み - 商品価格の需要変動・在庫状況に基づく最適価格設定。
- **feedback_engine.py**: ✅ 実装済み -  一日の業績を総括。翌日への改善点を特定。

**management_agent/customer_tools/**: 顧客対応Tool群 (Management Agentに統合実装)
- **respond_to_customer_inquiry.py**: ✅ 実装済み - 顧客問い合わせ対応・AI支援回答生成。
- **handle_customer_complaint.py**: ✅ 実装済み - 顧客苦情処理・補償措置実施。
- **collect_customer_feedback.py**: ✅ 実装済み - 顧客フィードバック収集・満足度分析。
- **recommend_product.py**: ❌ 未完成 - 購入履歴・嗜好分析に基づく商品推薦。パーソナライズ提案・クロスセル機会創出。

**management_agent/procurement_tools/**: 調達・在庫管理Tool群 (Management Agentに統合実装)
- **assign_restocking_task.py**: ✅ 実装済み - 従業員に商品補充作業を指示。
- **request_procurement.py**: ✅ 実装済み - 商品発注依頼生成・サプライヤ連絡依頼。
- **calculate_optimal_order.py**: ❌ 未完成(拡張案) - 在庫回転率・需要予測に基づく最適発注数量計算。過剰/不足在庫防止。

# TODO
単一ファイルから各ツールファイルへの転換
実際の実装では、各階層フォルダにおけるtoolではなく、`agent.py`に記載の
9つのノード関数が実装されている：

- `inventory_check_node()` - 在庫確認（LLM統合）
- `sales_plan_node()` - 売上計画（財務・売上分析）
- `pricing_node()` - 価格戦略（LLM駆動ツール活用）
- `automatic_restock_node()` - 在庫補充（LLM戦略分析）
- `procurement_request_generation_node()` - 発注依頼（STORAGE在庫判断）
- `sales_processing_node()` - 売上処理（シミュレーション統合）
- `customer_interaction_node()` - 顧客対応（LLM戦略分析）
- `profit_calculation_node()` - 利益計算（財務分析ツール活用）
- `feedback_node()` - 戦略的フィードバック（包括的LLM分析）

### Analytics Agent Tools (監視・分析・ガバナンス特化)

**analytics_agent/business_monitoring/**: ビジネスKPIの継続監視Tool群
- **performance_monitor.py**: ❌ 未完成 -売上・収益・回転率などの実績データを継続監視、ビジネス継続のポリシーに応じた振る舞いかを定量評価。
- **anomaly_detector.py**: ❌ 未完成 -売上急変・在庫異常・顧客苦情急増などの異常を統計的手法で検出。
- **compliance_checker.py**: ❌ 未完成 -経費精算・消費税計算・会計基準遵守を確認、顧客対応における法令やモラル順守を評価。

**analytics_agent/ai_governance/**: AI Agentの品質・安全監視Tool群
- **decision_quality_monitor.py**: ❌ 未完成 -AI意思決定の正しさ・一貫性・成功率を評価、判断品質スコア算出。管理Agentへの改善フィードバック。
- **safety_compliance_checker.py**: ❌ 未完成 -AIガードレールの遵守状況を確認、安全基準逸脱を検知。
- **performance_tracker.py**: ❌ 未完成 -AI応答時間・成功率・コストを監視。性能低下をアラート。

**analytics_agent/advisory/**: 業務分析・アドバイスTool群
- **cost_benefit_analyzer.py**: ❌ 未完成 -店長（Management）Agentの新施策・改善案の費用対効果を定量評価、ビジネス環境において投資判断が妥当かを評価。

# TODO
すべて未設計&未検証。店長Agentのprofit_calculation_nodeを参考に構築。
### Recorder Agent Tools (学習・記録蓄積特化)

**recorder_agent/learning_tools/**: データ蓄積・学習パターン生成Tool群
- **session_recorder.py**: ❌ 未完成 - Agentの意思決定・行動・結果を時系列で記録、参照用データ蓄積。詳細ログ分析用構造化保存。


# TODO
すべて未設計&未検証。店長Agentのfeedback_nodeを参考に構築。そもそも不要な可能性も検討

### Shared Tools (全Agent共有・一部はManagement Agentに統合実装)

**shared_tools/data_retrieval/**: 基本データ取得Tool群
- **check_inventory_status.py**: ✅ 実装済み - 在庫レベル・有効期限・補充必要量をリアルタイム確認。補充判断の前提データ収集。定期的に全Agentが参照。
- **get_business_metrics.py**: ✅ 実装済み - 売上・在庫・顧客満足度・利益率などのKPIデータを収集・集計。経営判断の基礎データとして使用。定期的に全Agentが参照。
- **collect_customer_feedback.py**: ❌ 未完成 - 顧客満足度・要望・クレームを収集・傾向分析。全Agentが利用する顧客データベース更新。

**shared_tools/market_tools/**: 市場・競合情報Tool群
- **search_products.py**: ✅ 実装済み - 新商品・競合価格・トレンド情報をweb検索。外部データ統合・価格比較。
- **supplier_research.py**: ❌ 未完成 - 仕入れ先情報・信頼性評価・価格比較。発注先選定支援・リスク評価。

# TODO
- 実際の`agent.py`では`create_tool_registry()`関数からツールを取得

## コンポーネント実装状況

### 1. AI統合システム [実装済み]

#### 自律型AI経営システム - 主軸機能
- **定義**: 人手干渉を最小限に事業運営を行うシステム
- **特徴**: 3-Agent + 17-tool協働による自動意思決定・実行
- **動作形態**: セッション型実行 (店長Agentが朝/昼/夕サイクルで業務実行)
- **目的**: 効率的・自律的・拡張可能な自動販売機事業運営

#### AI処理技術スタック

**Core Framework**
- Python 3.9+ | FastAPI 0.104.1 | Uvicorn 0.24.0 | Pydantic 2.5.0

```
AI Processing Stack
├── AI Models 
│   ├── Large Language Models (Azure OpenAI)✅実装済み
│   ├── Fallback Models (Anthropic 0.5.0)✅実装済み
│   └── ReAct Execution (Decision Reasoning・Tool Chain)❌ 未完成
├── Agent Frameworks 
│   ├── Orchestration (LangChain Tool System)✅実装済み
│   ├── Memory Management (LangChain・会話履歴・ChromaDB 0.4.22)❌ 未完成
│   └── Tool Integration (17個Tool連携・Tool Calling)✅実装済み
├── SaaS External Tools 
│   ├── Search APIs (Tavily 0.3.0+・バックアップDuckDuckGo)✅実装済み
│   ├── Payment APIs (Stripe統合検討中・PayPal・Square対応)❌ 未完成
│   └── Communication APIs (検討中: Mail・Slack・Teams)❌ 未完成
└── Runtime Infrastructure 
    ├── Local Development (Python AsyncIO・Concurrent処理)✅実装済み
    ├── Container Deployment (Docker対応)❌ 未完成
    ├── Web Framework (FastAPI・Jinja2 Template)❌ 未完成
    └── Database (SQLite/MongoDB)❌ 未完成
```


####  AI Models Agent Frameworks - AIモデル統合管理
- **目的**: 複数AIモデル統合管理
- **実装状況**: Azure OpenAI GPT-4o-mini ・Tavily検索対応
- **主機能**: マルチモデル切替・セーフティチェック・APIコスト管理・直列WF・ReAct実行パターン

#### 会話サービス - 会話履歴管理システム
- **目的**: 顧客会話ログ管理・パーソナライズデータ提供・AI学習データ蓄積
- **実装状況**: SQLiteベース実装（MongoDB NoSQL拡張検討中）
- **主機能**: セッション管理・メッセージ永続化・履歴検索・感情分析統合

#### データ管理層 - 多様なデータ処理
**会話履歴データ**
- **ストレージ**: NoSQL形式 (MongoDB対応検討中)
- **用途**: 顧客パーソナライズ・センチメント分析・推薦システム

**学習・アテンション・データ**
- **タイプ**: Agent意思決定ログ・行動結果・パターン学習データ
- **ストレージ**: ChromaベクトルDB・時系列DB・JSON永続化
- **用途**: システム改善・振る舞い最適化・振り返り分析

**業務実績データ**
- **タイプ**: 売上・在庫・顧客・業務KPI・財務指標
- **ストレージ**: SQLite
- **用途**: リアルタイム監視・トレンド分析・経営判断支援

### 2. API層 [部分実装]

**主要APIエンドポイント (基礎機能)**
- `vending.py` - 販売API [実装済み]: 商品購入・在庫確認
- `tablet.py` - タブレットAPI [実装済み]: 顧客接客インタフェース（AI Agent連携）
- `procurement.py` - 調達API [実装済み]: 商品発注・仕入れ先管理
- `dashboard.py` - ダッシュボードAPI [未実装]: 管理画面・レポート表示・データ分析


### システム性能・セキュリティ仕様

**AI安全性能基準（実装済み）**
- **レスポンス検証**: API成功/エラーチェック・タイムアウト監視（5.0秒）
- **アクション制限**: 禁止パターンマッチング（override_safety, bypass_payment等）

**将来拡張目標（未実装）**
- 🔄 **信頼性スコアリング**: 決定前LLM信頼性評価体制
- 🔄 **ガードレール機能**: 有効（設定でenable_guardrails制御）

**パフォーマンス基準**
- **API応答時間**: 平均2秒以内 (顧客体験保証)
- **並行処理**: AsyncIO対応・同時接続50本以上
- **データ処理**: リアルタイムKPI計算・1秒以内


### 環境設定仕様

**必須環境変数 (.env.example準拠)**
```
# 基本設定 (必須・現在のシステム動作に必要)
APP_NAME=AI Agent Autonomous Vending Machine System
DATABASE_URL=sqlite:///./ai_vending.db
HOST=0.0.0.0
PORT=8000

# AI API設定 (ANTHROPIC_API_KEYは必須、他オプション)
ANTHROPIC_API_KEY=your_anthropic_api_key_here          # 必須
OPENAI_API_KEY=your_openai_api_key_here                # Azure OpenAI用
OPENAI_API_BASE=https://your-resource.openai.azure.com/ # Azure OpenAI用
TAVILY_API_KEY=your_tavily_api_key_here                # 検索API用



# データ収集・検証設定
ENABLE_DATA_COLLECTION=True                           # データ収集有効
DATA_VALIDATION_LEVEL=strict                           # 検証レベル
LOG_AI_DECISIONS=True                                # AI判定ログ出力

# 決済API設定 (オプション)
STRIPE_API_KEY=your_stripe_api_key_here
PAYPAL_CLIENT_ID=your_paypal_client_id_here
```

**将来準備設定 (.env.exampleに記載・使用ロジック未実装)**
```
ENABLE_GUARDRAILS=True                               # 🔄 AIガードレール制御 (将来実装)
ENABLE_DECISION_MONITORING=True                      # 🔄 判定監視 (将来実装)
# セキュリティ設定 (ENCRYPTION_KEYは暗号化機能で必須)
JWT_SECRET_KEY=your_jwt_secret_key_here                # 必須 (認証用)
ENCRYPTION_KEY=your_encryption_key_here               # 必須 (データ暗号化用)
```

**Agent目的設定パラメータ**
```
agent_objectives:
  primary: ["収益最適化", "顧客満足度向上"]
  optimization_period:
    short_term: "今月売上最大化"
    medium_term: "顧客維持率向上"
    long_term: "資産価値増加"
  constraints: ["品質保証", "法令遵守", "リスク管理"]


# Search Agent追加設定
real_web_search: True                                 # Web実検索有効
search_timeout: 30                                    # 検索タイムアウト
search_max_retries: 3                                 # 検索再試行回数
```


### 3. 会計システム [実装済み]

**Journal Entry** - 仕訳処理
- **目的**: 自動販売取引の会計仕訳生成
- **実装状況**: 標準会計処理対応（売上・仕入・在庫計上）
- **主機能**: 取引自動仕訳・総勘定元帳更新

**Management Accounting** - 管理会計
- **目的**: 商品別採算性分析・業務効率評価
- **実装状況**: 基本分析機能実装
- **主機能**: 商品別利益計算・在庫回転率分析

### 4. データモデル [実装済み]

**主要モデルのフィールド定義**:

**Product Model (productsテーブル)**
```
- id: INTEGER PRIMARY KEY (自動採番)
- name: VARCHAR(100) NOT NULL (商品名)
- price: DECIMAL(10,2) NOT NULL (単価)
- category: VARCHAR(50) (カテゴリ: 飲料・食品・雑貨等)
- description: TEXT (商品説明)
- image_url: VARCHAR(200) (商品画像URL)
- is_active: BOOLEAN DEFAULT TRUE (販売中フラグ)
- created_at: DATETIME (作成日時)
- updated_at: DATETIME (更新日時)
```

**Transaction Model (transactionsテーブル)**
```
- id: INTEGER PRIMARY KEY (自動採番)
- product_id: INTEGER FOREIGN KEY → products.id
- machine_id: VARCHAR(20) (自販機ID)
- quantity: INTEGER NOT NULL (数量)
- total_amount: DECIMAL(10,2) NOT NULL (合計金額)
- payment_method: VARCHAR(20) (決済方法: cash/card/mobile)
- transaction_date: DATETIME NOT NULL (取引日時)
- session_id: VARCHAR(100) (AIセッションID)
- customer_feedback: TEXT (顧客フィードバック)
- metadata: JSON (拡張データ: AI分析結果等)
```

**Inventory Model (inventoryテーブル)**
```
- id: INTEGER PRIMARY KEY (自動採番)
- product_id: INTEGER FOREIGN KEY → products.id
- machine_id: VARCHAR(20) (自販機ID)
- current_stock: INTEGER NOT NULL (現在在庫数)
- min_stock_level: INTEGER DEFAULT 5 (最低在庫水準)
- max_stock_level: INTEGER DEFAULT 50 (最大在庫水準)
- last_restock_date: DATETIME (最終補充日時)
- expiry_date: DATE (賞味期限)
- location: VARCHAR(50) (商品陳列位置)
- status: VARCHAR(20) DEFAULT 'active' (現有・廃棄・返品等)
```

**Conversation Model (会話履歴 - plansコレクション) [未実装 - 計画中のデータモデル]**
```
- _id: ObjectId (MongoDB自動生成)
- session_id: String (セッション識別子)
- messages: Array [
  {
    role: String (user/assistant/system)
    content: String (メッセージ内容)
    timestamp: DateTime (送信時刻)
    metadata: Object (感情分析結果・意図分類等)
  }
]
- context: Object {
  agent_type: String (店長Agent・監査Agent等)
  tool_used: Array (使用したTool一覧)
  decision_made: Object (意思決定内容)
  outcome: String (実行結果)
}
- ai_insights: Object {
  sentiment_score: Float (感情スコア -1.0～1.0)
  intent_category: String (意図分類)
  action_needed: Boolean (フォローアップ必要フラグ)
}
- created_at: DateTime
- updated_at: DateTime
```

### 5. シミュレーションエンジン [実装済み]

**Sales Simulation** - 販売イベント生成
- **目的**: テスト・検証のための擬似販売データ生成
- **実装状況**: 実装済み
- **主機能**: 確率ベース取引生成・シナリオ制御

## 未実装項目と今後方針

### ⚠️ 高優先度未実装

**データストレージ拡張**
- **対象**: `models/conversation.py`, `services/conversation_service.py`
- **課題**: カンバンDBのみ。MongoDB対応未実装
- **方針**: Motorドライバ導入・NoSQL会話永続化
- **期日**: 次フェーズ

**ダッシュボードシステム**
- **対象**: `api/dashboard.py`, `static/dashboard/`
- **課題**: 管理画面・レポート表示・データ分析レイヤー未実装
- **方針**: データAPI提供・既存BIツール連携（Tableau/PowerBI等対応）
- **期日**: 次フェーズ

**セキュリティ強化**
- **対象**: 全API・設定管理
- **課題**: 本番向け認証・暗号化未実装
- **方針**: JWT認証・暗号化モジュール追加
- **期日**: セキュリティ検証後

**Customer Tools拡張**
- **対象**: `shared_tools/customer_tools/`
- **課題**: 感情分析・商品推薦機能を未実装
- **方針**: `analyze_customer_sentiment()`・`recommend_product()`機能追加
- **期日**: 次フェーズ (eコマース拡張時)

### 🔄 部分実装改善

**Analytics Agent拡張**
- **課題**: 基本分析のみ。異常検出・予測分析なし
- **方針**: 時系列分析・機械学習モデル追加

**Recorder Agent学習機能**
- **課題**: 記録のみ。フィードバックサイクルなし
- **方針**: 学習データ蓄積・意思決定改善反映

## テスト戦略

### 単体テスト [実装済み]
- **場所**: `tests/`
- **対象**: Agent, Service, Modelクラスの各メソッド
- **カバレッジ**: pytest使用・CI統合済み

### 統合テスト [整理完了]
- **場所**: `tests/test_three_day_integration.py`, `tests/test_multi_agent_integration.py`
- **対象**: エンドツーエンド業務フロー検証・全Agent協働テスト
- **実装状況**: pytest統合・テストアサーション強化

### シミュレーションテスト [整理完了]
- **場所**: `src/simulations/`
- **対象**: 販売イベント・Agent動作シミュレーション
- **実装状況**: 本番コードと分離・ルート散在スクリプト削除

### スクリプトツール群 [整理完了]
- **場所**: `scripts/`
- **対象**: ビジネスシミュレーション・データ分析・継続検証スクリプト
- **実装状況**: 本番環境検証・長期テスト用スクリプト整理完了

**ビジネスシミュレーションスクリプト**
- `continuous_multi_day_simulation.py`: 長期営業サイクル（複数日間）の自律運用シミュレーション・Agent協働検証
- `continuous_procurement_simulation.py`: 自動調達システムの継続的動作テスト・在庫最適化検証

**データ分析・可視化スクリプト**
- `kpi_visualization.py`: KPIデータ収集・グラフ生成・パフォーマンスダッシュボード作成

**実行方法と意義**
- ビジネスシミュレーション: `python scripts/continuous_multi_day_simulation.py` で実行、意思決定品質・運用効率の長期検証
- データ分析: `python scripts/kpi_visualization.py` でKPIレポート生成、経営判断支援・改善策立案

**ビジネス価値**
- リスク低減: 本番投入前の包括的システムテスト
- 効率化: KPI自動追跡・異常早期検知
- 継続改善: シミュレーションデータからの学習反映

## 開発ワークフロー

### 開発環境制約
- Windows環境・VSCode
- Python 3.9+・FastAPI 0.104.1
- Azure OpenAI API・Tavily Search API・Anthropic Claude

### デプロイメント構成

**ローカル開発環境**
```
Python AsyncIO + Uvicorn
├── SQLite DB (主DB)
├── ChromaDB (ベクトル検索)
├── MongoDB (会話履歴・オプション)
└── ホットリロード対応
```

**本番デプロイメント環境**
```
未定
```

**インフラ要件**
- **CPU**: 4コア以上 (AI処理並行実行)
- **RAM**: 8GB以上 (モデルロード・並行処理)
- **Storage**: 10GB以上 (ログ・データ蓄積)
- **Network**: 安定したインターネット接続 (AI API呼び出し)

**スケーリング戦略**
未定

### CI/CD状況 [未構築]
未定
