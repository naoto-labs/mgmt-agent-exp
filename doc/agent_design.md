# AI Agent Autonomous Vending Machine System - LangChain実装設計書

## 概要

本文書は、AI Agent Autonomous Vending Machine Systemの「3-Agent + 17-tool自律型アーキテクチャ」をLangChainを使用して実装するための詳細設計書です。

## Agent設計原則とパターンのセクション

### Agent設計原則

- **設計方針**: LangChainはAIモデル統合・会話管理・ツール実行・メモリ管理のフレームワークとして利用。本質はAgentベースの自律分散ソフトウェアアーキテクチャ設計
- **原則**:
  - **単一責務原則 (SRP)**: 各Agentは明確な責務（経営判断/監査/記録）に専用化
  - **疎結合 (Loose Coupling)**: Agent間メッセージングでインタラクション，多すぎる依存回避
  - **開放閉鎖原則 (OCP)**: 新Tool追加でAgent拡張
  - **依存性逆転 (DIP)**: 高レベルAgentが低レベルToolに依存せず，抽象インターフェース経由
  - **学習指向設計**: 全アクション記録・パターン分析・継続改善指向

### 採用Agentデザインパターン

- **Shared Tools Registry Pattern**:
  - 全Agentが共有可能なツールの集中管理パターン
  - Registryを通じてToolの登録・検索・実行を統一的に管理
  - 権限ベースアクセス制御で安全性を確保

- **Orchestrator Pattern**:
  - マルチAgent間の協調制御パターン
  - セッション型の業務実行サイクルを管理
  - Queueベースのメッセージバスで非同期通信

- **Memory-Augmented Agent Pattern**:
  - 会話履歴・学習データ・ベクトル検索統合
  - LangChainのメモリ管理による文脈維持
  - 長期記憶と短期記憶の階層化管理

- **Permission-Based Tool Access Pattern**:
  - Agent役職に応じたTool利用権限の動的制御
  - 経営判断Agentの専用Tool → 全社共有Tool → 分析専用Toolのアクセス階層

### LangChainとしての実装課題

- **現状認識**: 3Agentシンプル構成だが、Nodeベースの複雑な状態管理を実装できていない
  - LangChainライクな「Directed Acyclic Graph (DAG)」構造で振る舞い遷移・条件分岐を管理できない
  - セッション推移をしているが、Node定義による状態遷移の豊富さ・確実性に欠ける
- **課題**: 自律Agentの状態管理が会話メモリベースで、複雑なワークフロー実行が困難
- **設計すべきか**: シンプル構成を維持しつつ、Nodeベース状態遷移を導入するか検討中
  - Agent間メッセージングで代用可能な妥当性 vs Nodeベースの状況適合性

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
| 監査Agent | データ分析・KPI監査 + Tool活用 + 共通ツール利用 | 常時稼働 + Toolループ実行 | 分析Chain + Tool統合 |
| 記録Agent | 学習蓄積・パターン分析 + Tool記録 + 共通ツール利用 | 常時稼働 + Tool監視記録 | ベクトル検索 + Toolログ |

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

- **役割**: 経営判断の最終決定者。4カテゴリのShared Toolsを活用して意思決定
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

## Shared Tools Framework設計

### Shared Tools Framework概要

Shared Tools Frameworkは、3-Agentが共有可能なツールを集中管理・実行するアーキテクチャです。
4カテゴリ（データ取得・顧客対応・調達管理・市場分析）のツールを統一的に管理します。

### BaseTool抽象クラス設計

BaseToolは共有ツールの基底設計として、以下の機能を規定：

- **抽象インターフェース**:
  - execute(): ツール実行メソッド
  - validate_input(): 入力パラメータ検証
  - can_access(): Agentアクセス権限チェック

- **共通プロパティ**:
  - tool_id: ユニーク識別子
  - category: 分類カテゴリ
  - agent_access: アクセス可能なAgentリスト
  - usage_count: 使用回数トラッキング
  - last_used: 最終使用時刻

### Shared Tools Registryパターン

Registryパターンはツールの集中管理を実現：

- **管理機能**:
  - register_tool(): 新規ツール登録
  - get_tool(): Agentアクセス権限付き取得
  - execute_tool(): Registry経由実行
  - get_registry_stats(): 使用統計取得

- **アクセス制御**:
  - カテゴリベース権限制御
  - Agentタイプベース自動権限設定
  - 実行時動的権限検証

### Shared Tools統合方法

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
- **メモリ戦略**: ConversationSummaryBufferMemory + VectorStore for learning

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

#### 設定管理 (LangChain対応)

```python
class LangChainConfig(BaseSettings):
    """LangChain統合設定"""

    # LLM設定
    azure_openai_deployment: str = "gpt-4o-mini"
    anthropic_fallback: bool = True

    # LangChain Memory設定
    conversation_memory_limit: int = 4000
    vector_memory_dimension: int = 1536
    vector_search_k: int = 5

    # Tool Chain設定
    max_tools_per_chain: int = 10
    tool_timeout: int = 30

    # Agent Orchestration設定
    max_concurrent_agents: int = 3
    coordination_memory_limit: int = 8000

    class Config:
        env_file = ".env"
```

## 実装優先順位

### Phase 1: Shared Tools Framework実装
1. BaseTool抽象クラスおよびRegistryパターン実装
2. 17個Shared Toolの具体クラス実装
3. Shared Toolsアクセス権限システム実装

### Phase 2: Agent統合とShared Tools連携
1. Management AgentとのFull Access統合
2. Analytics AgentのLimited Access統合実装
3. Recorder AgentのRecording Access統合実装
4. LangChain Tool Calling統合強化

### Phase 3: 常時稼働Agent拡張
1. Analytics Agent常時監視機能実装
2. Recorder Agent学習・パターン分析機能追加
3. Agent間協調Message Bus実装

### Phase 4: システム拡張・安定化
1. ダッシュボードAPI・レポートシステム実装
2. セキュリティ強化・JWT認証・暗号化機能追加
3. Human-in-the-Loop統合（フィードバック学習）


