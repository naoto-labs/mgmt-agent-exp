# 🤖 AIマルチエージェント自律型自動販売機システム

**Tavily検索連携＆多シナリオテスト対応の完全自律型AI経営支援システム**

## 🎯 最新機能 - 2025年リリース

### ✅ **Tavily実検索統合** (NEW!)
- **実Web検索**: DuckDuckGo/Tavily APIによる実際の商品検索
- **スマートクエリ生成**: 在庫状況から適した検索クエリを自動生成
- **例**: 売上好調 → "人気飲料 新商品" / 在庫不足 → "供給安定したボトル飲料"

### ✅ **多様なテストシナリオ** (NEW!)
- **5つのテストモード**: 状況に応じて異なるAI応答テスト
- **最新のチューニング**: LLMが同じ応答を繰り返さない工夫
- **インタラクティブ選択**: 起動時にシナリオを選択

### ✅ **Management Agent強化**
- **データ連動ロジック**: 売上・在庫データを基に戦略意思決定
- **ログ完全クリーンアップ**: 重要な情報のみ詳細表示
- **LLMプロンプト最適化**: 初回のみシステムプロンプト表示

### ✅ **システム全体の安定化**
- **Azure OpenAI統合**: GPT-4o-mini優先使用
- **エラーハンドリング強化**: 各Agentの堅牢性向上
- **6種AI Agent協働**: 顧客・検索・経営・分析・記録・調達の連携

---

## 🚀 クイックスタート

### 1. 環境構築
```bash
# リポジトリクローン
git clone https://github.com/naoto-labs/mgmt-agent-exp.git
cd mgmt-agent-exp

# 依存関係インストール
pip install -r requirements.txt

# .envファイル作成（Tavily APIキー設定）
cp .env.example .env
# TAVILY_API_KEY=your_api_key_here  # 実検索には設定が必要
```

### 2. AI Agent協働テスト実行
```bash
python multi_agent_simulation.py
```

**🎯 シナリオ選択例:**
```
🎯 テストシナリオを選択してください:
1. high_sales     - 売上好調シナリオ (拡張戦略テスト)
2. low_inventory  - 在庫不足集中シナリオ (緊急調達テスト)
3. price_competition - 価格競争シナリオ (価格戦略テスト)
4. customer_demand - 顧客需要変動シナリオ (トレンド分析テスト)
5. default        - 標準シナリオ (現在の動作)
```

### 3. 各シナリオのAI応答確認
- **high_sales**: "売上好調のため、新商品飲料を検索" → Tavily検索実行
- **low_inventory**: カップヌードル補充以外にも多様なアクション
- **price_competition**: 価格戦略調整などのレスポンス
- **customer_demand**: 在庫最適化提案など

---

## 📊 テスト結果例

### High Salesシナリオ
```
2025-10-02 09:21:54,190 - src.agents.management_agent - INFO - 売上好調のため、新商品飲料を検索
2025-10-02 09:21:54,190 - src.agents.management_agent - INFO - 生成された検索クエリ: 人気飲料 新商品
2025-10-02 09:21:56,721 - src.agents.search_agent - INFO - DuckDuckGo実検索完了: 5件の結果
```

### Logクリーン化
```
2025-10-02 09:18:01,037 - src.agents.management_agent - DEBUG - === LLM PROMPT === (初回のみ)
2025-10-02 09:18:01,037 - src.agents.management_agent - DEBUG - === LLM RESPONSE ===
2025-10-02 09:20:58,270 - src.agents.management_agent - DEBUG - LLM called with established system prompt (2回目以降)
```

---

## 🎯 システムアーキテクチャ

```
🤖 AIマルチエージェントシステム
├── 🎯 Management Agent (LangChainベース)
│   ├── 売上・在庫データ分析
│   ├── 戦略意思決定 (LLMプロンプト生成)
│   ├── アクション実行 (在庫補充・価格調整)
│   └── 会計データ連携
├── 🔍 Search Agent (Tavily/DuckDuckGo)
│   ├── 商品価格比較検索
│   ├── 新商品発見・調達提案
│   └── 在庫状況連動クエリ生成
├── 👥 Customer Agent
│   ├── 顧客問い合わせ対応
│   └── パーソナライズ推薦
├── 📊 Analytics Agent
│   └── 営業分析レポート
├── 🧠 Recorder Agent
│   └── 学習データ記録
└── 📦 Procurement Agent
    └── 発注・調達管理
```

## ✨ 主要機能

### AI Agentシステム
- **Management Agent**: ビジネスデータ分析 → LLM意思決定 → アクション実行
- **Search Agent**: Tavily/DuckDuckGo使用の本物ネットワーク検索
- **6Agent統合**: 顧客対応・在庫管理・財務会計・分析・記録・調達

### 実ビジネスロジック
- **在庫最適化**: 需要予測に基づく自動補充
- **財務自動化**: 複式簿記による自動仕訳
- **決済シミュレーション**: 複数決済方法対応

### テスト＆検証
- **5種テストシナリオ**: 異なる状況でのAI応答検証
- **Tavily実検索**: 本物Web検索結果を使用
- **ログ完全監視**: Management Agentの思考プロセス追跡

## 📈 進化のポイント

### AI Agentの知性向上
- **データ連動**: 売上・在庫状況を基にしたスマートな意思決定
- **検索適応**: ビジネスデータから最適な検索クエリを生成
- **アクション多様化**: 価格調整・在庫補充・新商品提案

### システム実用性
- **本物Web統合**: Tavily APIを使用した現実的商品検索
- **テスト柔軟性**: 5つのシナリオで様々な状況を検証
- **ログ最適化**: 重要な情報のみをクリーンに表示

## 🎮 使用方法

### 1. テスト実行
```bash
python multi_agent_simulation.py
# ↑シナリオを選択してテスト開始
```

### 2. 分析確認
- **LLM応答の変化**: 各シナリオで異なる意思決定を確認
- **Tavily検索結果**: 実Web検索による新商品/価格情報
- **アクション実行**: 在庫補充・価格調整・新商品発注

### 3. ログ監視
- Business Metrics: 売上・在庫・顧客満足度
- LLM Prompt/Response: AI意思決定プロセス
- Action Execute: システムアクションの実行結果

## 🔧 設定

### 必須環境変数
```bash
TAVILY_API_KEY=your_tavily_api_key  # 実検索使用時に必要
OPENAI_API_KEY=your_openai_key      # Azure OpenAI対応
```

### ダミーモード
Tavily APIキーがない場合もテスト可能（シミュレーション検索にフォールバック）

---

## 🎉 進化の成果

### Before
- 単一シナリオのみテスト可能
- シミュレーション検索のみ
- LLM応答が同じ内容繰り返し

### After
- **5つの多様なテストシナリオ**
- **Tavily実Web検索統合**
- **データ連動型AI意思決定**
- **ログ完全クリーンアップ**

**これにより、AI Agentシステムの真の実力を様々な状況で検証できるようになりました！ 🚀**
