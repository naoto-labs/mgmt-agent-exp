# 実装計画

## 概要

AIエージェント自律型自動販売機システムの実装を段階的に進めます。各タスクは前のタスクの成果物を基に構築され、最終的に完全な事業運営システムを実現します。

## 実装タスク

- [ ] 1. プロジェクト基盤構築
  - プロジェクトディレクトリ構造の作成（src/, tests/, docs/, data/等）
  - pyproject.tomlとrequirements.txtの作成（FastAPI, Pydantic, SQLAlchemy, Alembic, anthropic, openai等）
  - .gitignoreファイルの作成（.env, __pycache__, *.log, *.db等を除外）
  - _要件: 1.1_

- [ ] 1.1 セキュリティ設定の実装
  - .env.exampleファイルの作成（ANTHROPIC_API_KEY、OPENAI_API_KEY、STRIPE_API_KEY等をダミー値で設定）
  - src/config/security.pyの実装（複数APIキー検証、暗号化機能）
  - src/config/settings.pyの実装（Pydantic BaseSettingsでセキュア設定管理）
  - _要件: 1.2_

- [ ] 1.2 基本FastAPIアプリケーション
  - src/main.pyの実装（FastAPIアプリ初期化、ルーター登録、起動/終了イベント）
  - ヘルスチェックエンドポイント（/health）の実装
  - CORS設定とセキュリティヘッダーの設定
  - _要件: 1.3_

- [ ] 2. 基本データモデルの実装
  - src/models/product.pyの実装（Product, ProductCategoryクラス）
  - src/models/transaction.pyの実装（Transaction, PaymentMethod, TransactionStatusクラス）
  - src/models/inventory.pyの実装（InventorySlot, InventoryStatusクラス）
  - _要件: 4.1, 7.1_

- [ ] 2.1 顧客・会話データモデル（NoSQL対応）
  - src/models/customer.pyの実装（Customer, CustomerProfileクラス）
  - src/models/conversation.pyの実装（NoSQL対応ConversationSession, ConversationMessageクラス）
  - src/models/supplier.pyの実装（Supplier, ProcurementOrderクラス）
  - MongoDBまたはJSONファイルベースの会話履歴管理システム
  - _要件: 6.1, 4.2_

- [ ] 2.2 会計データモデル
  - src/models/account.pyの実装（Account, ChartOfAccountsクラス）
  - src/models/journal_entry.pyの実装（JournalEntry, AccountingEntryクラス）
  - 勘定科目マスタデータの初期化
  - _要件: 7.1, 7.2_

- [ ] 2.3 データベース初期化（ハイブリッド構成）
  - SQLAlchemyエンジンとセッション管理の実装（SQLite接続：会計・取引データ）
  - NoSQL会話履歴管理システム（MongoDB/JSONファイル：会話データ）
  - データベーステーブル作成とマイグレーション機能（Alembic使用）
  - 初期データ投入スクリプトの作成（商品マスタ、勘定科目等）
  - _要件: 8.1, 6.1_

- [ ] 3. AIモデル管理システム
  - src/ai/model_manager.pyの実装（ModelManagerクラス）
  - 複数AIモデル（Anthropic Claude、OpenAI GPT）の統一インターフェース
  - モデル選択機能とフォールバック機能（プライマリ→セカンダリ）
  - _要件: 2.1_

- [ ] 3.1 AIエージェント基盤システム
  - src/agents/vending_agent.pyの実装（基本クラス構造、初期化メソッド）
  - ModelManagerを使用した柔軟なAI推論システム
  - 基本的なプロンプト構築とレスポンス解析機能
  - _要件: 2.1_

- [ ] 3.2 AI安全性システム
  - src/agents/safety_monitor.pyの実装（SafetyMonitorクラス、安全性チェック）
  - AIガードレール機能（禁止パターン検出、アクション制限）
  - エラーカウント管理と緊急停止機能
  - _要件: 3.1, 3.2_

- [ ] 3.3 AI意思決定プロセス
  - 購入リクエスト処理メソッドの実装
  - 在庫確認と商品情報取得の統合
  - AI推論結果の検証とフォールバック機能
  - _要件: 2.2, 2.3_

- [ ] 4. 決済システムシミュレーター
  - src/services/payment_service.pyの実装（PaymentServiceクラス）
  - 決済処理のシミュレーション（実際のStripe APIなし）
  - 決済成功/失敗のランダム生成とログ記録
  - _要件: 2.2, 5.2_

- [ ] 4.1 タブレットシミュレーター
  - src/simulator/tablet_simulator.pyの実装（TabletSimulatorクラス）
  - 商品選択と決済フローのシミュレーション（実際のUIなし）
  - シミュレートされた顧客操作とレスポンス生成
  - _要件: 5.1_

- [ ] 4.2 決済シミュレーション
  - 決済処理のモック実装（実際のStripe連携なし）
  - 成功/失敗パターンのシミュレーション
  - 決済ログとトランザクション記録
  - _要件: 5.3, 5.4_

- [ ] 5. 在庫管理サービス
  - src/services/inventory_service.pyの実装（InventoryServiceクラス）
  - 自販機内在庫と保管庫在庫の分離管理
  - 在庫レベル監視と低在庫アラート機能
  - _要件: 4.3_

- [ ] 5.1 商品排出シミュレーター
  - src/simulator/dispense_simulator.pyの実装（DispenseSimulatorクラス）
  - 商品排出のシミュレーション（仮想モーター、仮想センサー）
  - 排出成功/失敗のランダム生成とログ記録
  - _要件: 2.4_

- [ ] 5.2 補充計画システム
  - 補充タイミング算出アルゴリズムの実装
  - 自販機への補充指示生成機能
  - 補充履歴の記録と分析機能
  - _要件: 4.3_

- [ ] 6. 検索エージェント統合
  - src/agents/search_agent.pyの実装（SearchAgentクラス）
  - ModelManagerを使用した複数モデル対応の価格検索
  - 検索結果の構造化と価格比較機能（AIが検索結果を解析）
  - _要件: 4.1_

- [ ] 6.1 仕入れ先管理システム
  - src/procurement/supplier_finder.pyの実装（SupplierFinderクラス）
  - 仕入れ先評価アルゴリズム（価格、信頼性、配送時間）
  - 最適仕入れ先選定ロジックの実装
  - _要件: 4.2_

- [ ] 6.2 発注システムシミュレーター
  - src/procurement/order_generator.pyの実装（OrderGeneratorクラス）
  - 発注指示書の自動生成（テキスト形式、PDF出力は簡易版）
  - 発注承認のシミュレーション（自動承認/拒否）
  - _要件: 4.4_

- [ ] 7. 顧客対話エンジン（NoSQL統合）
  - src/agents/customer_agent.pyの実装（CustomerAgentクラス）
  - ModelManagerを使用した柔軟な会話AI（Claude/GPT選択可能）
  - NoSQL会話履歴を活用したコンテキスト管理とパーソナライゼーション
  - src/services/conversation_service.pyの実装（NoSQL会話履歴管理）
  - _要件: 6.1_

- [ ] 7.1 顧客分析システム
  - src/customer_engagement/preference_analyzer.pyの実装
  - 購入履歴分析とパーソナライゼーションアルゴリズム
  - 顧客セグメンテーションと嗜好予測機能
  - _要件: 6.2_

- [ ] 7.2 クーポン管理システム
  - src/customer_engagement/coupon_manager.pyの実装
  - 動的割引率計算アルゴリズム（収益性考慮）
  - クーポン生成、配布、使用追跡機能
  - _要件: 6.3_

- [ ] 7.3 新商品希望収集
  - 顧客要望の記録と分類システム
  - 要望頻度分析と仕入れ計画への反映機能
  - 要望フィードバックシステムの実装
  - _要件: 6.4_

- [ ] 8. 仕訳処理システム
  - src/accounting/journal_entry.pyの実装（JournalEntryProcessorクラス）
  - 売上仕訳自動生成（現金/売上高）
  - 仕入仕訳自動生成（仕入高/買掛金、商品/仕入高）
  - _要件: 7.1, 7.2_

- [ ] 8.1 総勘定元帳管理
  - src/accounting/ledger_manager.pyの実装（LedgerManagerクラス）
  - 勘定科目別残高計算と履歴管理
  - 試算表作成と勘定科目検索機能
  - _要件: 7.2_

- [ ] 8.2 財務諸表生成
  - src/accounting/financial_reports.pyの実装（FinancialReportGeneratorクラス）
  - 損益計算書自動生成（売上高、売上原価、粗利益、営業利益）
  - 貸借対照表自動生成（資産、負債、純資産）
  - _要件: 7.4_

- [ ] 9. 管理会計分析
  - src/accounting/management_accounting.pyの実装（ManagementAccountingAnalyzerクラス）
  - 商品別収益性分析（売上、原価、粗利率計算）
  - 期間別収益比較とトレンド分析
  - _要件: 7.3_

- [ ] 9.1 在庫分析システム
  - 在庫回転率計算アルゴリズムの実装
  - 在庫効率評価とデッドストック検出
  - 在庫最適化提案機能
  - _要件: 7.3_

- [ ] 9.2 KPI算出システム
  - 主要業績指標（売上成長率、顧客単価、リピート率）の算出
  - ダッシュボード用データ集計機能
  - KPIアラートとしきい値管理
  - _要件: 8.3_

- [ ] 10. 事象追跡システム
  - src/analytics/event_tracker.pyの実装（EventTrackerクラス）
  - 全事象（購入、補充、対話、エラー）の構造化ログ記録
  - イベントストリーム処理とリアルタイム監視
  - _要件: 8.1_

- [ ] 10.1 異常検出システム
  - src/analytics/anomaly_detector.pyの実装（AnomalyDetectorクラス）
  - 統計的異常検出アルゴリズム（Z-score、IQR法）
  - 異常パターン分類と重要度評価
  - _要件: 8.2_

- [ ] 10.2 予測エンジン
  - src/analytics/prediction_engine.pyの実装（PredictionEngineクラス）
  - 需要予測モデル（移動平均、季節調整）
  - 在庫需要予測と収益予測機能
  - _要件: 8.4_

- [ ] 10.3 レポート生成システム
  - src/analytics/report_generator.pyの実装（ReportGeneratorクラス）
  - 包括的な分析レポート自動生成
  - PDF/Excel形式でのレポート出力機能
  - _要件: 8.3_

- [ ] 10.4 ダッシュボード可視化システム
  - src/dashboard/dashboard_service.pyの実装（DashboardServiceクラス）
  - リアルタイム売上・在庫・顧客データの可視化
  - インタラクティブなチャートとグラフ（Chart.js/Plotly使用）
  - KPIメトリクスのリアルタイム表示
  - _要件: 8.3, 7.3_

- [ ] 11. API エンドポイント実装
  - src/api/vending.pyの実装（販売関連API）
  - src/api/tablet.pyの実装（タブレット用API）
  - src/api/procurement.pyの実装（調達管理API）
  - _要件: 2.3_

- [ ] 11.1 顧客・会計API
  - src/api/customer.pyの実装（顧客エンゲージメントAPI）
  - src/api/accounting.pyの実装（会計データAPI）
  - src/api/admin.pyの実装（管理者用API）
  - _要件: 2.4_

- [ ] 11.2 ダッシュボードAPI
  - src/api/dashboard.pyの実装（ダッシュボード用データAPI）
  - リアルタイムメトリクス配信エンドポイント
  - WebSocket対応によるリアルタイム更新機能
  - チャート用データ形式の標準化
  - _要件: 8.3_

- [ ] 11.3 Webダッシュボード実装
  - static/dashboard/index.htmlの実装（メインダッシュボード画面）
  - JavaScript/Chart.jsによるインタラクティブなグラフ実装
  - リアルタイムデータ更新機能（WebSocket/SSE使用）
  - レスポンシブデザインとモバイル対応
  - _要件: 8.3_

- [ ] 11.4 エラーハンドリング統合
  - 全APIエンドポイントの統一エラーハンドリング
  - HTTPステータスコードとエラーメッセージの標準化
  - ログ記録とアラート機能の統合
  - _要件: 3.4_

- [ ] 12. ユニットテスト実装
  - tests/test_agents.pyの実装（AIエージェントテスト）
  - tests/test_payment.pyの実装（決済システムテスト）
  - tests/test_accounting.pyの実装（会計システムテスト）
  - _要件: 6.1_

- [ ] 12.1 統合テスト実装
  - tests/test_vending_flow.pyの実装（購入フロー統合テスト）
  - tests/test_procurement.pyの実装（調達システム統合テスト）
  - tests/test_customer_chat.pyの実装（顧客対話テスト）
  - _要件: 6.2_

- [ ] 12.2 AI安全性テスト
  - tests/test_ai_safety.pyの実装（ガードレール、異常検出テスト）
  - セキュリティテスト（APIキー漏洩、インジェクション攻撃）
  - パフォーマンステスト（応答時間、メモリ使用量）
  - _要件: 6.3_

- [ ] 13. システム統合とオーケストレーション
  - src/services/orchestrator.pyの実装（全システム統制）
  - 起動シーケンスと依存関係管理
  - システム全体のヘルスチェック機能
  - _要件: 2.3, 2.4_

- [ ] 13.1 ドキュメント作成
  - docs/api_documentation.mdの作成（OpenAPI仕様書）
  - docs/deployment_guide.mdの作成（デプロイメント手順）
  - docs/operation_manual.mdの作成（運用マニュアル）
  - _要件: 1.3_

- [ ] 13.2 本番環境準備
  - 本番用環境変数設定とセキュリティ強化
  - ログローテーションとバックアップ機能
  - 監視アラートとメンテナンス機能の実装
  - _要件: 1.3_