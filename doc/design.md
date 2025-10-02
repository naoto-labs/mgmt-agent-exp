# AI Agent Autonomous Vending Machine System 設計書

## 概要

AI Agent Autonomous Vending Machine Systemは、3-Agent + 17-tool自律型アーキテクチャによる完全自律型自動販売システムです。
Azure OpenAI + Tavily統合により、高度なAI意思決定と実Web検索を実現します。

## アーキテクチャ概要

```
ai-vending-system/
├── agents/                      # 🚀 自律Agent（主要3つ + 共有Tool）
│   ├── management_agent/        # 店長Agent
│   │   ├── management_tools/    # 経営判断Tool
│   │   │   ├── get_business_metrics.py
│   │   │   ├── analyze_financial_performance.py
│   │   │   ├── feedback_engine.py
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
│   │   ├── analysis/            # 分析業務
│   │   │   ├── efficiency_analyzer.py
│   │   │   └── cost_benefit_analyzer.py
│   │   └── orchestrator.py
│   ├── recorder_agent/          # 記録Agent
│   │   ├── learning_tools/
│   │   │   ├── session_recorder.py
│   │   │   ├── data_persistence.py
│   │   │   ├── pattern_analyzer.py
│   │   │   └── objective_data_manager.py
│   │   └── orchestrator.py
│   ├── shared_tools/            # 🔧 共有Tool (4カテゴリ)
│   │   ├── data_retrieval/
│   │   │   ├── check_inventory_status.py
│   │   │   └── collect_customer_feedback.py
│   │   ├── customer_tools/
│   │   │   ├── respond_to_customer_inquiry.py
│   │   │   ├── handle_customer_complaint.py
│   │   │   └── create_customer_engagement_campaign.py
│   │   ├── procurement_tools/
│   │   │   ├── request_procurement.py
│   │   │   ├── assign_restocking_task.py
│   │   │   └── coordinate_employee_tasks.py
│   │   └── market_tools/
│   │       ├── search_products.py
│   │       ├── supplier_research.py
│   │       └── market_analysis.py
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

## アーキテクチャ原則

1. **AIエージェント自律性**: AI Agentが自律的に意思決定・実行
2. **AI安全性第一**: ガードレールと異常検出システム
3. **拡張可能なデータ基盤**: 将来的なデータ充実に対応
4. **モジュラー設計**: 新機能の容易な追加
5. **総合監視**: AI意思決定プロセス・システム状態追跡

### 3-Agent + 17-tool共有アーキテクチャ

- **定義**: 店長Agent(経営判断)・監査Agent(データ分析)・記録Agent(学習蓄積)の3つの独立Agentが、4カテゴリ17関数からなる共有toolシステムを協調して使用
- **特徴**: toolは店長Agent主導設計だが、監査Agent・記録Agentもシステム情報取得時に共有アクセス可能
- **目的**: Agent間連携による効率的な分工と自律的運営の実現

## Agent定義

### 店長Agent (Management Agent)

- **役割**: 経営判断の最終決定者。4カテゴリのtoolを活用して意思決定
- **動作形態**: セッション型 (朝/昼/夕の業務サイクル)
- **責務**: 戦略立案, 人間協働指示, 事業KPI管理
- **主な機能**:
  - 朝ルーチン: 夜間データの分析と今日の業務計画立案
  - 昼間チェック: 午前実績の分析と昼間戦略調整
  - 夕方総括: 1日全体の実績評価と改善策立案
  - 戦略的意思決定: AI分析に基づく経営判断の実行
- **実装状況**: セッションベースの意思決定システム実装済み

### 監査Agent (Analytics Agent)

- **役割**: データ分析と業務改善提案
- **動作形態**: 常時稼働
- **責務**: KPI算出, 無駄削減提案, 業務効率化監査
- **主な機能**:
  - リアルタイムKPI監視: 売上・在庫・顧客満足度の継続監視
  - パフォーマンス分析: 日次・週次・月次の実績分析・レポート生成
  - 異常検出: システム異常・業務変動の自動検知
  - 改善提案: データに基づく効率化・コスト削減の提案
- **実装状況**: 分析フレームワーク実装済み、実際データ連携機能は作成中

### 記録Agent (Recorder Agent)

- **役割**: 全行動の記録と学習データ蓄積
- **動作形態**: 常時稼働
- **責務**: パターン分析, 成功事例抽出, 失敗学習
- **主な機能**:
  - セッション記録: Agentの意思決定・行動・結果の詳細ログ化
  - パターン学習: 成功・失敗パターンの自動認識・蓄積
  - データ永続化: 学習データの長期保存・検索
  - 改善フィードバック: 蓄積データからの意思決定改善
- **実装状況**: 記録機能実装済み、学習・分析機能は作成中

## Tool定義

以下のTool定義は、アーキテクチャ概要のフォルダ構成と完全に対応しています。各AgentのToolを整理分類して定義します。

### Management Agent Tools (経営判断特化)

**management_agent/management_tools/**: 店長Agentの戦略的意思決定を支援するTool群

- **get_business_metrics.py**: ✅ 実装済み - 売上・在庫・顧客満足度・利益率などのKPIデータを収集・集計。経営判断の基礎データとして使用。定期的に全Agentが参照し、戦略立案の前提となる。
- **analyze_financial_performance.py**: ✅ 実装済み - 財務データ分析、収益性評価、改善提案。貸借対照表・損益計算書の自動生成・連続比較分析。
- **update_pricing_strategy.py**: ✅ 実装済み - 商品価格の動的調整・競合価格分析。需要変動・在庫状況に基づく最適価格設定。
- **assign_restocking_task.py**: ✅ 実装済み - 従業員に商品補充作業を指示。緊急度設定・タスクID生成。
- **request_procurement.py**: ✅ 実装済み - 商品発注依頼生成・サプライヤ連絡。発注数量・タイミング計算。
- **coordinate_employee_tasks.py**: ✅ 実装済み - 従業員業務配分・進捗管理。新商品発注通知・タスク調整。
- **respond_to_customer_inquiry.py**: ✅ 実装済み - 顧客問い合わせ対応・AI支援回答生成。
- **handle_customer_complaint.py**: ✅ 実装済み - 顧客苦情処理・補償措置実施。
- **collect_customer_feedback.py**: ✅ 実装済み - 顧客フィードバック収集・満足度分析。
- **create_customer_engagement_campaign.py**: ✅ 実装済み - 顧客エンゲージメント施策企画・実行。
- **plan_agent_operations.py**: ❌ 未完成 - 日次・週次業務計画の立案・実行手順策定。
- **plan_sales_strategy.py**: ❌ 未完成 - 売上目標設定・プロモーション戦略立案。

### Analytics Agent Tools (監視・分析・ガバナンス特化)

**analytics_agent/business_monitoring/**: ビジネスKPIの継続監視Tool群
- **performance_monitor.py**: 売上・収益・回転率などの実績データを継続監視、閾値逸脱を検知。ダッシュボード更新・リアルタイム表示。
- **anomaly_detector.py**: 売上急変・在庫異常・顧客苦情急増などの異常を統計的手法で検出。通知システムとの連携で即時対応。
- **compliance_checker.py**: 経費精算・消費税計算・会計基準遵守を確認、法令順守を確保。監査レポート生成。

**analytics_agent/ai_governance/**: AI Agentの品質・安全監視Tool群
- **decision_quality_monitor.py**: AI意思決定の正しさ・一貫性・成功率を評価、判断品質スコア算出。管理Agentへの改善フィードバック。
- **safety_compliance_checker.py**: AIガードレールの遵守状況を確認、安全基準逸脱を検知。緊急停止トリガー機能。
- **performance_tracker.py**: AI応答時間・成功率・学習進捗を監視、性能低下をアラート。最適化提案生成。

**analytics_agent/analysis/**: 業務分析・最適化Tool群
- **efficiency_analyzer.py**: 業務プロセス効率・無駄削減機会を分析、改善提案を生成。KPI比較・ベンチマーク分析。
- **cost_benefit_analyzer.py**: 新施策・改善案の費用対効果を定量評価、投資判断支援。ROI計算・シナリオ分析。

### Recorder Agent Tools (学習・記録蓄積特化)

**recorder_agent/learning_tools/**: データ蓄積・学習パターン生成Tool群
- **session_recorder.py**: Agentの意思決定・行動・結果を時系列で記録、学習データ蓄積。詳細ログ分析用構造化保存。
- **data_persistence.py**: 学習データ・分析結果を永続化、検索・再利用可能な形で保存。長期学習データのメンテナンス。
- **pattern_analyzer.py**: 成功・失敗パターンを自動認識、将来判断の参考データに変換。機械学習モデルへの入力データ生成。
- **objective_data_manager.py**: バイアスのかからない客観データを収集・整理・提供。信頼性の高い学習データベース構築。

### Shared Tools (全Agent共有・一部はManagement Agentに統合実装)

**shared_tools/data_retrieval/**: 基本データ取得Tool群
- **check_inventory_status.py**: ✅ 実装済み - 在庫レベル・有効期限・補充必要量をリアルタイム確認。補充判断の前提データ収集。
- **collect_customer_feedback.py**: ✅ 実装済み - 顧客満足度・要望・クレームを収集・傾向分析。全Agentが利用する顧客データベース更新。

**shared_tools/customer_tools/**: 顧客対応Tool群 (Management Agentに統合実装)
- **respond_to_customer_inquiry.py**: ✅ 実装済み - 顧客問い合わせ内容を分析、適切な回答を自動生成。会話ログとの連携分析。
- **handle_customer_complaint.py**: ✅ 実装済み - クレーム内容解決策を提案、補償措置を実施。エスカレート判断・対応履歴蓄積。
- **create_customer_engagement_campaign.py**: ✅ 実装済み - 顧客エンゲージメント施策企画・実行。ターゲティング分析・効果測定。
- **analyze_customer_sentiment.py**: ❌ 未完成 - 会話内容から感情分析・満足度スコア算出。顧客フィードバックの感情傾向把握。
- **recommend_product.py**: ❌ 未完成 - 購入履歴・嗜好分析に基づく商品推薦。パーソナライズ提案・クロスセル機会創出。

**shared_tools/procurement_tools/**: 調達・在庫管理Tool群 (Management Agentに統合実装)
- **request_procurement.py**: ✅ 実装済み - 補充必要品の発注依頼生成、サプライヤ連絡。最適発注数量・タイミング計算。
- **assign_restocking_task.py**: ✅ 実装済み - 従業員への在庫補充作業割り当て、タスク管理。優先順位付け・スケジューリング。
- **coordinate_employee_tasks.py**: ✅ 実装済み - 調達・補充関連全業務の進捗管理・調整。ワークフロー最適化・効率化。
- **calculate_optimal_order.py**: ❌ 未完成 - 在庫回転率・需要予測に基づく最適発注数量計算。過剰/不足在庫防止。
- **generate_procurement_request.py**: ❌ 未完成 - 発注書自動生成・発注プロセス管理。備品・消耗品の一括発注対応。

**shared_tools/market_tools/**: 市場・競合情報Tool群
- **search_products.py**: 新商品・競合価格・トレンド情報をTavily検索。外部データ統合・価格比較。
- **supplier_research.py**: 仕入れ先情報・信頼性評価・価格比較。発注先選定支援・リスク評価。
- **market_analysis.py**: 需要変動・価格動向・競合戦略分析。予測分析・戦略立案支援。

## コンポーネント実装状況

### 1. AI統合システム [実装済み]

#### 自律型AI経営システム - 主軸機能
- **定義**: 人手干渉を最小限に事業運営を行うシステム
- **特徴**: 3-Agent + 17-tool協働による自動意思決定・実行
- **動作形態**: セッション型実行 (店長Agentが朝/昼/夕サイクルで業務実行)
- **目的**: 効率的・自律的・拡張可能な自動販売機事業運営

#### AI処理技術スタック詳細

**Core Framework**
- Python 3.9+ | FastAPI 0.104.1 | Uvicorn 0.24.0 | Pydantic 2.5.0

```
AI Processing Stack
├── AI Models ✅実装済み
│   ├── Large Language Models (Azure OpenAI GPT-4o-mini)
│   ├── Fallback Models (Anthropic 0.5.0・OpenAI 1.3.7)
│   └── ReAct Execution (Decision Reasoning・Tool Chain)
├── Agent Frameworks ✅実装済み
│   ├── Orchestration (LangChain 0.1.0 Tool System)
│   ├── Memory Management (LangChain・会話履歴・ChromaDB 0.4.22)
│   └── Tool Integration (17個Tool連携・Function Calling)
├── SaaS External Tools ✅実装済み
│   ├── Search APIs (Tavily 0.3.0+・バックアップDuckDuckGo)
│   ├── Payment APIs (Stripe統合検討中・PayPal・Square対応)
│   └── Communication APIs (検討中: Mail・Slack・Teams)
└── Runtime Infrastructure ✅実装済み
    ├── Local Development (Python AsyncIO・Concurrent処理)
    ├── Container Deployment (Docker対応・Kubernetes予定)
    ├── Web Framework (FastAPI・Jinja2 Template)
    └── Database (SQLite/MongoDB・Motor 3.3.2ドライバ)
```

**将来拡張予定:**
- Safety & Alignment検証 (Guardrails.ai検討)
- Communication APIs (Mail・Slack・Teams統合検討)

#### Model Manager - AIモデル統合管理
- **目的**: 複数AIモデル統合管理・Web検索対応・コスト最適化
- **実装状況**: Azure OpenAI GPT-4o-mini 統合済み・Tavily検索対応
- **主機能**: マルチモデル切替・セーフティチェック・APIコスト管理・ReAct実行パターン

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
- **ストレージ**: SQLite + 時系列DB
- **用途**: リアルタイム監視・トレンド分析・経営判断支援

### 2. API層 [部分実装]

**主要APIエンドポイント (基礎機能)**
- `vending.py` - 販売API [実装済み]: 商品購入・在庫確認
- `tablet.py` - タブレットAPI [実装済み]: 顧客接客インタフェース（AI Agent連携）
- `procurement.py` - 調達API [実装済み]: 商品発注・仕入れ先管理
- `dashboard.py` - ダッシュボードAPI [未実装]: 管理画面・レポート表示・データ分析

**将来拡張API **
- __人間介入・通知レイヤー__ (検討中):
  - `alert_system.py`: 緊急アラート・異常通知API
  - `human_override.py`: 人間による緊急介入API

### システム性能・セキュリティ仕様

**AI安全性能基準（実装済み）**
- **レスポンス検証**: API成功/エラーチェック・タイムアウト監視（5.0秒）
- **アクション制限**: 禁止パターンマッチング（override_safety, bypass_payment等）
- **冗長性確保**: マルチモデルファallback（Azure OpenAI → OpenAI → Anthropic）
- **ガードレール機能**: 有効（設定でenable_guardrails制御）

**将来拡張目標（未実装）**
- 🔄 **信頼性スコアリング**: 決定前LLM信頼性評価体制
- 🔄 **安心度閾値**: ai_safety_threshold設定値活用（現在: 0.95）

**パフォーマンス基準**
- **API応答時間**: 平均2秒以内 (顧客体験保証)
- **並行処理**: AsyncIO対応・同時接続50本以上
- **データ処理**: リアルタイムKPI計算・1秒以内

**セキュリティ仕様**
- **許可アクション**: select_product, process_payment, dispense_product, customer_service
- **禁止パターン**: override_safety, bypass_payment, unlimited_dispense
- **データ検証レベル**: strict (入力値完全バリデーション)
- **認証方式**: JWTトークン + APIキー (本番移行時)

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

# セキュリティ設定 (ENCRYPTION_KEYは暗号化機能で必須)
JWT_SECRET_KEY=your_jwt_secret_key_here                # 必須 (認証用)
ENCRYPTION_KEY=your_encryption_key_here               # 必須 (データ暗号化用)

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
AI_SAFETY_THRESHOLD=0.95                              # 🔄 AI安全性閾値 (将来実装)
ENABLE_GUARDRAILS=True                               # 🔄 AIガードレール制御 (将来実装)
ENABLE_DECISION_MONITORING=True                      # 🔄 判定監視 (将来実装)
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
  priority_weight: {"short_term": 0.6, "medium_term": 0.3, "long_term": 0.1}

# Search Agent追加設定
real_web_search: True                                 # Web実検索有効
search_timeout: 30                                    # 検索タイムアウト
search_max_retries: 3                                 # 検索再試行回数
```

**システム起動検証プロセス**
- ✅ 必須キー検証: ANTHROPIC_API_KEY + ENCRYPTION_KEY (必須検証)
- ✅ オプション設定確認: Azure OpenAI, Stripe等 (未設定警告)
- ✅ ガードレール設定確認: enable_guardrails (推奨警告)
- ✅ .env.example準拠チェック: 設定値書式検証

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
Docker + Kubernetes (計画)
├── PostgreSQL/MySQL (本番DB)
├── Redis (キャッシュ・セッション)
├── MongoDB (NoSQLデータ)
├── Nginx (リバースプロキシ)
└── 水平スケーリング対応
```

**インフラ要件**
- **CPU**: 4コア以上 (AI処理並行実行)
- **RAM**: 8GB以上 (モデルロード・並行処理)
- **Storage**: 50GB以上 (ログ・データ蓄積)
- **Network**: 安定したインターネット接続 (AI API呼び出し)

**スケーリング戦略**
- **水平スケーリング**: Agent別コンテナ化
- **垂直スケーリング**: AIモデルGPU対応検討
- **データ分割**: 時系列データ・地理的分割

### CI/CD状況 [未構築]
- **計画中のテスト自動化**: GitHub Actionsベース・セキュリティスキャン
- **要件計画**: 全テスト通過・セキュリティチェッククリア
- **計画中**: pip-audit・safety・secrets検知
