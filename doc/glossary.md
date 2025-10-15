# 用語集

## システム全体用語

### AI Agent Autonomous Vending Machine System
- **意味**: 3つの自律型Agent (店長Agent, 監査Agent, 記録Agent) と17の専門Toolが協働して、自動販売機事業を完全自律的に運営するAIシステム
- **別名**: 自律型AI経営システム
- **特徴**: Azure OpenAI + Tavily統合により、高度なAI意思決定と実Web検索を実現
- **アーキテクチャ**: 3-Agent + tool共有アーキテクチャ
- **実装技術**: Python 3.9+ | FastAPI | Azure OpenAI GPT-4o-mini | Tavily API
- **関連**: セッション型実行, AI安全性第一, 拡張可能なデータ基盤

### 3-Agent + 17-tool共有アーキテクチャ
- **意味**: 店長Agent(経営判断)・監査Agent(データ分析)・記録Agent(学習蓄積)の3つの独立Agentが、4カテゴリ17関数からなる共有toolシステムを協調して使用する設計
- **特徴**: toolは店長Agent主導設計だが、監査Agent・記録Agentもシステム情報取得時に共有アクセス可能
- **目的**: Agent間連携による効率的な分工と自律的運営の実現


## Agent関連用語

### 店長Agent (Management Agent)
- **意味**: 経営判断の最終決定者。各種toolを活用して事業運営の戦略的意思決定を行う
- **動作形態**: セッション型 (朝/昼/夕の業務サイクルで自動実行)
- **責務**: 戦略立案, 人間協働指示, 事業KPI管理, 業務計画立案, 売上戦略策定
- **主な機能**:
  - 朝ルーチン: 夜間データの分析と今日の業務計画立案
  - 昼間チェック: 午前実績の分析と昼間戦略調整
  - 夕方総括: 1日全体の実績評価と改善策立案


### 監査Agent (Analytics Agent)
- **意味**: データ分析と業務改善提案を専門とするAgent
- **動作形態**: 常時稼働・自律型
- **責務**: KPI算出, 無駄削減提案, 業務効率化監査, AI品質監視, 安全監視
- **主な機能**:
  - リアルタイムKPI監視
  - パフォーマンス分析・レポート生成
  - 異常検出システム
  - 改善提案・コスト削減提案


### 記録Agent (Recorder Agent)
- **意味**: 全行動の記録と学習データ蓄積を専門とするAgent
- **動作形態**: 常時稼働・自律型
- **責務**: パターン分析, 成功事例抽出, 失敗学習, データ永続化
- **主な機能**:
  - セッション記録・行動・結果の詳細ログ化
  - パターン学習・成功・失敗パターンの自動認識
  - データ永続化・検索・再利用可能な形で保存
  - 改善フィードバック・蓄積データからの意思決定改善


### 調達Agent (Procurement Agent)
- **意味**: 調達・在庫管理特化の補助Agent
- **動作形態**: 店長Agentの下位Agentとして機能
- **責務**: 商品発注・仕入れ先管理・在庫最適化
### 顧客Agent (Customer Agent)
- **意味**: 顧客接客・対応特化の補助Agent
- **動作形態**: 店長Agentの下位Agentとして機能
- **責務**: 顧客問い合わせ対応・苦情処理・満足度向上

### 検索Agent (Search Agent)
- **意味**: Web検索・市場情報収集特化の外部Agent
- **動作形態**: 独立した検索機能として稼働
- **責務**: 新商品情報・競合価格・仕入れ先調査

## Tool関連用語

### Toolシステム
- **意味**: Agentが自律的に業務を実行するための機能単位・関数ライブラリ
- **構成**: Management Tools (経営判断) / Analytics Tools (監視分析) / Recorder Tools (学習記録) / Shared Tools (全Agent共有)
- **特徴**: 4カテゴリ17個の専門Toolが協働・店長Agent主導設計だが全Agentが共有アクセス可能

### Shared Tools (共有Tools)
- **意味**: 全Agentが使用可能な共通機能ツール
- **種類**: データ取得・顧客対応・調達管理・市場情報収集の4カテゴリ
- **用途**: Agent間連携と効率化


## 技術用語

### AI統合システム
- **意味**: 自律型AI経営システムの基幹機能・3-Agent + 17-tool協働による自動判断・実行
- **特徴**: 人手干渉最小限・効率的・自律的・拡張可能事業運営
- **実装技術**: Python AsyncIO | Azure OpenAI GPT-4o-mini | Tavily API統合

### Azure OpenAI
- **意味**: プライマリAIプロバイダー・OpenAIのAzureホスト版
- **モデル**: GPT-4o-mini優先・ReAct Execution + Tool Chain連携
- **特徴**: マルチモデルfallback (Azure → OpenAI → Anthropic)
- **用途**: Agent意思決定支援・セーフティチェック・コスト最適化

### Tavily API
- **意味**: 実Web検索を実現する外部API・高精度検索サービス
- **特徴**: DuckDuckGoより高精度・バックアップ機能あり
- **用途**: search_agentの情報収集基盤・市場分析・競合調査

### DuckDuckGo
- **意味**: Tavily APIのバックアップ検索エンジン
- **特徴**: プライバシー重視・API不要
- **用途**: Tavily不可時の代替検索・Fallback機能

### Chroma DB
- **意味**: ベクトル検索・データベースシステム
- **用途**: 学習データ・会話履歴・ベクトル化ストレージ
- **特徴**: 高速類似検索・埋め込みベクトル管理

### Motor (MongoDBドライバ)
- **意味**: MongoDBの非同期Pythonドライバ
- **用途**: 会話履歴・NoSQLデータ処理
- **特徴**: AsyncIO対応・高速I/O

### FastAPI
- **意味**: 高速Webフレームワーク・Python製APIサーバー
- **特徴**: AsyncIOネイティブ・自動APIドキュメント生成 (Swagger/OpenAPI)
- **用途**: APIエンドポイント実装・リアルタイム処理

## 業務用語

### セッション型実行 (Session-based Execution)
- **意味**: 店長Agentが朝/昼/夕固定サイクルで業務自動実行する形態
- **目的**: 効率的リソース利用と定期戦略見直し
- **サイクル**: 朝分析/昼調整/夕評価

### 多シナリオテスト (Multi-Scenario Testing)
- **意味**: 様々な業務状況再現テスト・5シナリオ検証
- **シナリオ**: high_sales/ low_inventory/ price_competition/ customer_demand/ default
- **目的**: AI Agentの状況適応能力検証

### 専門Tool連携 (Tool Cooperation)
- **意味**: 店長Agentが検索・顧客対話・調達のTool適宜活用運用
- **特徴**: Tool独立機能単位・柔軟協働

## データ管理用語

### データモデル
主要なデータモデルを定義で統一管理：

- **Product Model**: 商品マスタ・id/name/price/category/description/image_url/is_active/created_at/updated_at
- **Transaction Model**: 取引履歴・id/product_id/machine_id/quantity/total_amount/payment_method/transaction_date/session_id/customer_feedback/metadata
- **Inventory Model**: 在庫情報・id/product_id/machine_id/current_stock/min_stock_level/max_stock_level/last_restock_date/expiry_date/location/status
- **Conversation Model**: 会話履歴・_id/session_id/messages/context/ai_insights/created_at/updated_at

### ベクトルストア (Vector Store)
- **意味**: 記録Agent用学習データ格納システム・Chroma DB使用
- **用途**: 行動パターン検索と類似事例参照・高速ベクトル検索

### 会話履歴 (Conversation History)
- **意味**: 顧客対話データ蓄積・パーソナライズ・感情分析
- **ストレージ**: NoSQL形式 (MongoDB対応・JSONファイルfallback)
- **用途**: パーソナライズ推薦・顧客分析・センチメント分析連携

## テスト・検証用語

### 単体テスト (Unit Tests)
- **意味**: Agent/Service/Model各メソッド個別機能テスト
- **ツール**: pytest使用・CI統合済み
- **場所**: `tests/` ディレクトリ

### 統合テスト (Integration Tests)
- **意味**: エンドツーエンド業務フロー検証・全Agent協働テスト
- **実装**: `test_three_day_integration.py` / `test_multi_agent_integration.py`
- **ツール**: pytest統合・アサーション強化

### シミュレーションテスト (Simulation Tests)
- **意味**: 販売イベント・Agent動作疑似生成テスト
- **場所**: `src/simulations/` ディレクトリ
- **実装**: 本番コード分離・確率ベース取引生成

### シナリオテスト (Scenario Test)
- **意味**: 特定業務状況再現テストケース
- **種類**: 売上好調/在庫不足/価格競争/需要変動/標準

## 属性用語

### 自律型システム (Autonomous System)
- **意味**: 人手干渉最小限事業運営システム
- **特徴**: 3-Agent + 17-tool協働自動意思決定・実行
- **目的**: 効率・自律・拡張可能運営

### 拡張可能設計 (Extensible Design)
- **意味**: 新機能追加・外部API連携容易アーキテクチャ
- **特徴**: モジュラー構造・クリーンインターフェース
- **実装**: プラグイン式Tool追加・コンポーネント分離

### AI安全性第一 (AI Safety First)
- **意味**: ガードレールと異常検出システム優先設計
- **実装**: レスポンス検証・アクション制限・冗長性確保・ガードレール機能

### モジュラー設計 (Modular Design)
- **意味**: 新機能容易追加設計
- **特徴**: ビジネスルール・インフラ・アプリケーションロジック分離
- **実装**: domain/infrastructure/application/sharedレイヤー構成
