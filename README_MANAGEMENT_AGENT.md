# LangChain Management Agent 実装ガイド

## 概要

LangChainを使用したセッション型経営管理Agentシステムの実装です。自動販売機事業の統合経営管理を行い、記録・学習・人間協働を実現します。

## 実装されたコンポーネント

### 1. SessionBasedManagementAgent (`src/agents/management_agent.py`)

セッション型で動作する経営管理Agent。必要な時だけ起動し、戦略的意思決定と業務実行を行います。

**主要機能:**
- セッション管理（開始/終了）
- ビジネスメトリクス取得と分析
- 戦略的意思決定
- システム連携ツール（在庫管理、価格戦略等）
- 人間協働ツール（タスク割り当て、調達依頼等）
- 顧客対応ツール（問い合わせ対応、苦情処理等）

**一日の業務フロー:**
- `morning_routine()` - 朝のルーチン（9:00）
- `midday_check()` - 昼のチェック（12:00）
- `evening_summary()` - 夕方の総括（17:00）

### 2. RecorderAgent (`src/agents/recorder_agent.py`)

行動記録・分析専用Agent。全ての管理行動と結果を記録し、パターン分析と学習データの蓄積を行います。

**主要機能:**
- 行動記録（ManagementActionRecord）
- 結果記録（BusinessOutcomeRecord）
- ベクトルストアを使用した類似検索
- 成功パターン抽出
- 失敗事例からの学習
- セッション推奨事項の生成

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**追加された依存関係:**
- `langchain==0.1.0`
- `langchain-openai==0.0.5`
- `langchain-anthropic==0.1.0`
- `chromadb==0.4.22`
- `tiktoken==0.5.2`

### 2. 環境変数の設定

`.env`ファイルに以下を設定:

```env
# Anthropic API (推奨)
ANTHROPIC_API_KEY=your_anthropic_api_key

# OpenAI API (代替)
OPENAI_API_KEY=your_openai_api_key
```

## 使用方法

### デモスクリプトの実行

実装を検証するためのデモスクリプトを用意しています:

```bash
python demo_management_agent.py
```

**デモ内容:**
1. Management Agentの基本機能
2. セッション型実行フロー
3. 一日の業務フロー（朝・昼・夕）
4. 人間協働機能
5. 顧客対応機能
6. Recorder Agent機能

### プログラムでの使用

#### 基本的な使用例

```python
import asyncio
from src.agents.management_agent import SessionBasedManagementAgent

async def main():
    # Agentの初期化
    agent = SessionBasedManagementAgent(provider="anthropic")
    
    # ビジネスメトリクスの取得
    metrics = agent.get_business_metrics()
    print(f"売上: {metrics['sales']}")
    
    # セッション型実行
    session_id = await agent.start_management_session("morning_routine")
    
    # 戦略的意思決定
    context = "在庫が少なくなっています"
    decision = await agent.make_strategic_decision(context)
    print(f"決定: {decision['decision']}")
    
    # セッション終了
    summary = await agent.end_management_session()
    print(f"セッション時間: {summary['duration']}")

asyncio.run(main())
```

#### 一日の業務フロー

```python
async def daily_operations():
    agent = SessionBasedManagementAgent(provider="anthropic")
    
    # 朝のルーチン
    morning = await agent.morning_routine()
    
    # 昼のチェック
    midday = await agent.midday_check()
    
    # 夕方の総括
    evening = await agent.evening_summary()
    
    return {
        "morning": morning,
        "midday": midday,
        "evening": evening
    }
```

#### RecorderAgentの使用

```python
from datetime import datetime
from src.agents.recorder_agent import (
    RecorderAgent,
    ManagementActionRecord,
    BusinessOutcomeRecord
)

async def record_management_activity():
    recorder = RecorderAgent()
    
    # 行動を記録
    action = ManagementActionRecord(
        record_id="action_001",
        session_id="session_001",
        timestamp=datetime.now(),
        action_type="decision",
        context={"sales": 100000},
        decision_process="価格調整を決定",
        executed_action="価格を10%値上げ",
        expected_outcome="利益率5%向上"
    )
    
    await recorder.record_action(action)
    
    # 結果を記録
    outcome = BusinessOutcomeRecord(
        record_id="outcome_001",
        session_id="session_001",
        related_action_id="action_001",
        timestamp=datetime.now(),
        outcome_type="sales",
        metrics={"profit_margin": 0.35},
        success_level="excellent",
        lessons_learned=["価格調整が効果的"]
    )
    
    await recorder.record_outcome(outcome)
```

## テストの実行

### 単体テスト

```bash
# Management Agentのテスト
pytest tests/test_management_agent.py -v

# Recorder Agentのテスト
pytest tests/test_recorder_agent.py -v

# 全テスト実行
pytest tests/ -v
```

### テストカバレッジ

```bash
pytest tests/ --cov=src/agents --cov-report=html
```

## アーキテクチャ

```
┌─────────────────────────────────────────┐
│   SessionBasedManagementAgent           │
│   (セッション型実行)                     │
├─────────────────────────────────────────┤
│  - morning_routine()                    │
│  - midday_check()                       │
│  - evening_summary()                    │
│                                         │
│  ツールセット:                          │
│  - システム連携 (データ取得、分析)       │
│  - 人間協働 (タスク管理、調達)          │
│  - 顧客対応 (問い合わせ、苦情処理)       │
└─────────────────────────────────────────┘
              ↓ 記録
┌─────────────────────────────────────────┐
│   RecorderAgent                         │
│   (記録・学習専用)                       │
├─────────────────────────────────────────┤
│  - record_action()                      │
│  - record_outcome()                     │
│  - extract_successful_patterns()        │
│  - extract_failure_lessons()            │
│  - analyze_session_patterns()           │
│                                         │
│  ベクトルストア:                        │
│  - action_store (行動記録)              │
│  - decision_store (意思決定記録)        │
│  - outcome_store (結果記録)             │
└─────────────────────────────────────────┘
```

## ツール一覧

### システム連携ツール
- `get_business_data` - ビジネスデータ取得
- `analyze_financials` - 財務分析
- `check_inventory` - 在庫確認
- `update_pricing` - 価格更新

### 人間協働ツール
- `assign_restocking` - 補充タスク割り当て
- `request_procurement` - 調達依頼
- `schedule_maintenance` - メンテナンススケジュール
- `coordinate_tasks` - タスク調整

### 顧客対応ツール
- `customer_response` - 顧客問い合わせ対応
- `handle_complaint` - 苦情処理
- `collect_feedback` - フィードバック収集
- `create_campaign` - キャンペーン企画

## データモデル

### SessionInfo
セッション情報を管理

```python
SessionInfo(
    session_id: str,
    session_type: str,
    start_time: datetime,
    end_time: Optional[datetime],
    decisions_made: List[Dict],
    actions_executed: List[Dict]
)
```

### ManagementActionRecord
管理行動の記録

```python
ManagementActionRecord(
    record_id: str,
    session_id: str,
    timestamp: datetime,
    action_type: str,
    context: Dict,
    decision_process: str,
    executed_action: str,
    expected_outcome: str,
    actual_outcome: Optional[str],
    success_score: Optional[float]
)
```

### BusinessOutcomeRecord
事業結果の記録

```python
BusinessOutcomeRecord(
    record_id: str,
    session_id: str,
    related_action_id: Optional[str],
    timestamp: datetime,
    outcome_type: str,
    metrics: Dict[str, float],
    success_level: str,
    lessons_learned: List[str]
)
```

## トラブルシューティング

### ベクトルストアの初期化エラー

RecorderAgentでベクトルストアの初期化に失敗する場合:

```python
# ベクトルストアなしで動作（記録機能は無効）
recorder = RecorderAgent()
# エラーログが出力されますが、他の機能は動作します
```

### APIキーのエラー

環境変数が正しく設定されているか確認:

```bash
python -c "import os; print(os.getenv('ANTHROPIC_API_KEY', 'NOT SET'))"
```

### メモリ不足

大量のセッションを実行する場合、メモリ制限を調整:

```python
agent = SessionBasedManagementAgent(provider="anthropic")
agent.session_memory = ConversationSummaryBufferMemory(
    llm=agent.llm,
    max_token_limit=2000  # デフォルト: 4000
)
```

## 今後の拡張

設計書に基づき、以下の機能を段階的に実装予定:

1. **HistoricalLearner** - 履歴からの学習システム
2. **HumanTaskInterface** - 高度な人間協働機能
3. **CustomerServiceInterface** - 統合顧客対応システム
4. **ManagementSessionOrchestrator** - セッション実行統制

## 参考資料

- 詳細設計: `doc/langchain_management_agent_design.md`
- LangChain公式ドキュメント: https://python.langchain.com/
- Anthropic Claude API: https://docs.anthropic.com/
- ChromaDB: https://docs.trychroma.com/

## ライセンス

このプロジェクトは実験的な実装です。
