# 🤖 AIエージェント自律型自動販売機システム

Anthropic PJ Vend構成の実験的実装 - AIエージェントによる完全自律型の自動販売機ビジネスシミュレーター

**🎯 モデルベース販売シミュレーションシステム**
- Stripeなどの実際の決済サービスを使用せず、現実的な販売モデルをシミュレーション
- 時間帯・曜日・季節・顧客行動を考慮した高度な需要予測
- 市場シナリオ分析によるビジネス戦略の検証・分析
- **✅ 動作確認済み**: ダミー値で正常に起動・動作
- **✅ Azure OpenAI対応**: 最新のAzure OpenAI APIをサポート
- **✅ オプション暗号化**: ENCRYPTION_KEY設定不要で動作可能

### 💡 モデルベースアプローチの利点

#### 現実的なビジネス検証
- **リスクなしの実験**: 実際の損失リスクなく戦略をテスト
- **迅速な検証**: 数ヶ月かかる市場テストを数分でシミュレーション
- **パラメータ調整**: 様々な条件での結果を即座に確認
- **データ駆動型意思決定**: 客観的な指標に基づく戦略立案

#### 包括的な影響分析
- **マルチファクター考慮**: 時間・曜日・季節・経済状況の複合影響を分析
- **顧客行動モデリング**: 心理的・行動経済学的要因を考慮した予測
- **市場ダイナミクス**: 競合・トレンド・外的要因の相互作用をシミュレーション
- **戦略的洞察**: 長期的なビジネス影響を予測・評価

#### 実用的ビジネス支援
- **価格戦略最適化**: 需要弾力性に基づく動的最適価格設定
- **在庫戦略立案**: 需要予測精度を活かした効率的な在庫管理
- **マーケティング効果測定**: プロモーションのROIを事前評価
- **リスク管理**: 様々な市場シナリオでのリスク評価と対策立案

### 🏪 販売シミュレーションモデル特徴

#### 高度な需要予測
- **時間帯別パターン**: 24時間の需要変動をモデル化（朝・昼・夕・夜のピークタイム）
- **曜日別変動**: 月曜日〜日曜日の需要パターン（平日・週末の違い）
- **季節・月別要因**: 夏・冬・春・秋の季節変動と月別トレンド
- **商品別人気度**: 商品ごとのベース人気度スコア管理

#### 顧客行動シミュレーション
- **購入確率予測**: 商品・時間帯・顧客属性に基づく購入確率計算
- **満足度予測**: 商品体験後の顧客満足度シミュレーション
- **リピート購入率**: 満足度とブランドロイヤルティに基づく再訪予測
- **価格感度分析**: 顧客の価格に対する反応度評価

#### 市場環境シミュレーション
- **経済状況**: 好景気・不景気・リセッションの影響シミュレーション
- **競合状況**: 競合他社の参入・撤退による市場シェア変動
- **マーケティング効果**: キャンペーン・プロモーションの影響分析
- **外的要因**: 天候・イベント・社会情勢の影響モデル化

#### ビジネス戦略支援
- **価格最適化**: コスト・需要・競合を考慮した最適価格算出
- **在庫管理**: 需要予測に基づく最適在庫量と発注タイミング
- **プロモーション効果**: 割引・キャンペーンの効果測定とROI分析
- **リスク評価**: 様々な市場シナリオでのリスクと機会の分析

## 📋 概要

このシステムは、AIエージェントによる自動販売機ビジネスの完全なシミュレーション環境を提供します。複数のAIモデルを統合し、顧客対応、在庫管理、決済処理、財務会計、分析機能を備えた包括的なプラットフォームです。

## ✨ 主な機能

### 🤖 AIエージェントシステム
- **検索エージェント**: AIによる価格比較と最適仕入れ先選定
- **顧客エージェント**: 会話履歴を活用したパーソナライズ対応
- **多モデル対応**: Anthropic Claude + OpenAI GPT統合

### 💰 ビジネスシステム
- **決済シミュレーター**: 現実的な決済フロー検証
- **在庫管理**: 自販機・保管庫統合管理システム
- **財務自動化**: 複式簿記による自動仕訳処理
- **管理会計**: 収益性・効率性分析・トレンド分析

### 📊 監視・分析システム
- **リアルタイム監視**: イベント追跡とシステム健全性チェック
- **統合管理**: オーケストレーターによる全システム統制
- **RESTful API**: 外部システムとの連携機能

## 🚀 クイックスタート

### 1. 環境設定

```bash
# リポジトリのクローン
git clone https://github.com/naoto-labs/mgmt-agent-exp.git
cd mgmt-agent-exp

# 依存関係のインストール（rye使用）
rye sync
```

### 2. 環境変数の設定

```bash
# .envファイルの作成
cp .env.example .env

# .envファイルの編集（ダミー値でテスト可能）
# ANTHROPIC_API_KEY=sk-ant-api03-test-dummy-key-for-development-only
# ENCRYPTION_KEY=dummy_encryption_key_32_characters_long_123456789012
```

### 3. システム起動

```bash
# FastAPIサーバー起動
rye run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# またはオーケストレーター起動
rye run python -m src.services.orchestrator
```

システムが起動すると、以下のURLでアクセス可能になります：
- **メインアプリケーション**: http://localhost:8000
- **APIドキュメント**: http://localhost:8000/docs
- **代替ドキュメント**: http://localhost:8000/redoc

## 📚 システム構成

```
mgmt-agent-exp/
├── src/
│   ├── main.py                 # FastAPIメインアプリケーション
│   ├── config/                 # 設定・セキュリティ管理
│   ├── models/                 # データモデル（商品・取引・在庫）
│   ├── ai/                     # AIモデル管理システム
│   ├── services/               # ビジネスロジックサービス
│   ├── agents/                 # AIエージェント（検索・顧客対応）
│   ├── accounting/             # 会計システム（仕訳・管理会計）
│   ├── analytics/              # 分析システム（イベント追跡）
│   └── api/                    # REST APIエンドポイント
├── tests/                      # 包括的なテストスイート
├── data/                       # データストレージ
├── static/                     # Webダッシュボード
└── doc/                        # ドキュメント
```

## 🔗 APIエンドポイント

### ヘルスチェック
```bash
GET /health
```

### 商品管理
```bash
GET /api/v1/vending/products      # 商品一覧取得
GET /api/v1/vending/inventory     # 在庫状況確認
POST /api/v1/vending/payment      # 決済処理
```

### 顧客対応
```bash
POST /api/v1/tablet/chat/start    # 顧客チャット開始
POST /api/v1/tablet/chat/message  # メッセージ送信
GET /api/v1/tablet/recommendations/{customer_id}  # パーソナライズ推奨
```

### 調達・検索
```bash
GET /api/v1/procurement/suppliers/search  # 仕入れ先検索
GET /api/v1/procurement/prices/compare    # 価格比較
```

### 会計・分析
```bash
GET /api/v1/accounting/reports           # 会計レポート
GET /api/v1/analytics/events             # イベント分析
GET /api/v1/analytics/system-health      # システム健全性
```

## 🧪 テスト実行

### ユニットテスト
```bash
# 全テスト実行
rye run pytest tests/ -v

# 特定テスト実行
rye run pytest tests/test_payment.py -v      # 決済システムテスト
rye run pytest tests/test_accounting.py -v   # 会計システムテスト
rye run pytest tests/test_agents.py -v       # AIエージェントテスト
```

### システムテスト
```bash
# システム診断実行
rye run python -c "
import asyncio
from src.services.orchestrator import orchestrator

async def test():
    await orchestrator.initialize_system()
    health = await orchestrator.check_system_health()
    print(f'システム状態: {health.overall_status}')

asyncio.run(test())
"
```

## ⚙️ 設定詳細

### 必須環境変数

| 変数名 | 説明 | 例 |
|--------|------|----|
| `ANTHROPIC_API_KEY` | Anthropic APIキー | `sk-ant-api03-...` |
| `ENCRYPTION_KEY` | データ暗号化キー（32文字以上） | `dummy_encryption_key_32_characters_long_...` |

### オプション環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `OPENAI_API_KEY` | OpenAI APIキー | なし |
| `OPENAI_API_BASE` | Azure OpenAIエンドポイント | なし |
| `STRIPE_API_KEY` | Stripe決済キー | なし |
| `DEBUG` | デバッグモード | `False` |
| `HOST` | サーバーホスト | `0.0.0.0` |
| `PORT` | サーバーポート | `8000` |

### Azure OpenAI設定例

Azure OpenAIを使用する場合は、以下の環境変数を設定してください：

```bash
# Azure OpenAI設定
OPENAI_API_KEY=your_azure_openai_api_key_here
OPENAI_API_BASE=https://your-resource-name.openai.azure.com/
```

Azure OpenAIを使用する場合の設定例：
```bash
# Azure OpenAI（優先使用）
OPENAI_API_KEY=1234567890abcdef1234567890abcdef
OPENAI_API_BASE=https://your-company.openai.azure.com/

# または標準OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here
# OPENAI_API_BASEは設定不要
```

### ダミー値でのテスト

開発環境では以下のダミー値でシステムをテストできます：

```bash
ANTHROPIC_API_KEY=sk-ant-api03-test-dummy-key-for-development-only
OPENAI_API_KEY=sk-test-dummy-key-for-development-only
OPENAI_API_BASE=https://dummy.openai.azure.com/
ENCRYPTION_KEY=dummy_encryption_key_32_characters_long_123456789012
```

## 🎯 使用例

### 1. 商品購入シミュレーション

```bash
# 商品一覧取得
curl http://localhost:8000/api/v1/vending/products

# 決済処理
curl -X POST http://localhost:8000/api/v1/vending/payment \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150,
    "payment_method": "card",
    "product_ids": ["prod_001", "prod_002"]
  }'
```

### 2. 顧客対応シミュレーション

```bash
# チャット開始
curl -X POST http://localhost:8000/api/v1/tablet/chat/start \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "customer_123",
    "initial_message": "おすすめの商品を教えてください"
  }'

# メッセージ送信
curl -X POST http://localhost:8000/api/v1/tablet/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "message": "ありがとうございます。購入します。"
  }'
```

### 3. 仕入れ先検索

```bash
# 価格比較検索
curl "http://localhost:8000/api/v1/procurement/suppliers/search?product_name=コーラ&max_price=100"
```

### 4. 販売シミュレーション（モデルベース）

```python
# Pythonスクリプトでの販売シミュレーション
from src.services.payment_service import payment_service

# 現実的な販売シミュレーション
sale_result = payment_service.simulate_realistic_sale('drink_cola', quantity=2)
print(f"予測需要: {sale_result['predicted_demand']:.2f}")
print(f"実際販売数: {sale_result['actual_quantity']}")
print(f"総売上: ¥{sale_result['total_amount']}")

# 需要予測（7日間）
forecast = payment_service.get_demand_forecast('drink_cola', days=7)
print(f"7日間総需要予測: {forecast['summary']['total_predicted_demand']:.2f}")

# 市場シナリオ分析
scenario = payment_service.simulate_market_scenario('economic_boom')
print(f"好景気シナリオの推奨事項: {scenario['recommendations'][0]}")
```

### 5. 高度な分析

```python
# 高度なビジネス分析
analytics = payment_service.get_advanced_analytics()
print(f"在庫効率性評価: {analytics['inventory_efficiency']['drink_cola']['efficiency_rating']}")
print(f"最適価格: ¥{analytics['product_analytics']['drink_cola']['optimal_price']}")
```

## 🔧 開発情報

### 技術スタック

- **フレームワーク**: FastAPI (Python 3.9+)
- **AIモデル**: Anthropic Claude, OpenAI GPT
- **データベース**: SQLite (開発環境), MongoDB対応
- **決済**: Stripe, PayPal統合対応
- **テスト**: pytest, pytest-asyncio
- **依存関係管理**: rye

### 開発コマンド

```bash
# 依存関係インストール
rye sync

# 開発サーバー起動
rye run uvicorn src.main:app --reload

# テスト実行
rye run pytest tests/ -v

# コードフォーマット
rye run black src/ tests/
rye run isort src/ tests/

# 型チェック
rye run mypy src/
```

## 📊 システム機能詳細

### 🤖 AIエージェント機能

#### 検索エージェント
- 複数ソースからの価格情報収集
- AIによる最適仕入れ先選定
- 市場動向分析と予測

#### 顧客エージェント
- 自然言語による顧客対応
- 購入履歴に基づくパーソナライズ
- 満足度分析と改善提案

### 💰 決済・財務機能

#### 決済システム
- 複数決済方法対応（カード、現金、クーポン）
- 決済成功率シミュレーション
- 返金・キャンセル処理

#### 会計システム
- 自動仕訳生成（複式簿記）
- 財務諸表自動作成
- 収益性分析（商品別・期間別）

### 📦 在庫管理機能

#### 在庫最適化
- 自販機・保管庫統合管理
- 需要予測による自動発注
- 在庫回転率分析

#### 補充計画
- 最適補充タイミング算出
- コスト効率考慮した発注計画
- 在庫切れリスク管理

## 🔒 セキュリティ

- **APIキー管理**: 環境変数によるセキュアな設定管理
- **入力検証**: 全エンドポイントの型安全なデータ検証
- **ログセキュリティ**: APIキーの自動マスキング
- **エラーハンドリング**: 包括的な例外処理

## 📈 監視・分析

### リアルタイム監視
- システム健全性チェック（60秒間隔）
- イベント追跡とログ記録
- パフォーマンス指標収集

### 分析機能
- 売上トレンド分析
- 顧客行動パターン分析
- 在庫効率性評価

## 🤝 貢献方法

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 📄 ライセンス

このプロジェクトは研究・開発目的で作成されたものです。実際の商用利用には適切な法的確認が必要です。

## 🆘 トラブルシューティング

### よくある問題と解決策

#### Q: 「有効なANTHROPIC_API_KEYが必要です」エラー
A: `.env`ファイルに適切なAPIキーを設定してください。ダミー値でテストする場合は以下の値を設定：
```
ANTHROPIC_API_KEY=sk-ant-api03-test-dummy-key-for-development-only
```

#### Q: 依存関係のインストールエラー
A: rye環境が正しく設定されているか確認してください：
```bash
rye --version  # バージョン確認
rye sync       # 依存関係再インストール
```

#### Q: サーバー起動エラー
A: ポート8000が使用中の場合は他のポートを指定：
```bash
rye run uvicorn src.main:app --port 8001 --reload
```

## 📞 連絡先

開発者: naoto-labs
GitHub: https://github.com/naoto-labs/mgmt-agent-exp

---

**🎉 このシステムはAIエージェントによる完全自律型の自動販売機ビジネスをシミュレートし、研究開発・検証に活用できる包括的なプラットフォームです。**
