# LangChain管理Agent実装設計書

## 概要

本文書は、自動販売機事業の統合経営管理AgentをLangChainを使用して実装するための詳細設計書です。セッション型実行、記録蓄積による学習、人間協働を前提とした包括的な経営管理システムを定義します。

## システム全体アーキテクチャ

### 基本構成

```
┌─────────────────────────────────────────────────────────────┐
│                    統合経営管理システム                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   管理Agent     │  │  RecorderAgent  │  │  システム基盤    │ │
│  │ (セッション型)   │  │   (記録専用)    │  │  (24時間自動)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   人間従業員     │  │     顧客       │  │   外部システム   │ │
│  │  (作業実行)     │  │   (対話相手)    │  │  (仕入れ先等)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 役割分担

| コンポーネント | 責務 | 動作形態 |
|---------------|------|----------|
| 管理Agent | 戦略的意思決定、人間への指示、顧客対応 | セッション型 |
| RecorderAgent | 行動記録、パターン分析、学習データ蓄積 | 常時稼働 |
| システム基盤 | データ収集、自動処理、KPI算出 | 24時間自動 |
| 人間従業員 | 物理作業、調達実行、メンテナンス | 指示ベース |

## 管理Agent詳細設計

### 1. セッション型管理Agent

```python
class SessionBasedManagementAgent:
    """セッション型経営管理Agent"""
    
    def __init__(self):
        self.session_id = None
        self.session_start_time = None
        self.session_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=4000
        )
        self.action_recorder = ActionRecorder()
        self.historical_learner = HistoricalLearner()
        
        # システムプロンプト
        self.system_prompt = """
        あなたは自動販売機事業の経営者です。以下の業務を統括してください：
        
        【システム連携業務】
        - 売上・財務データの分析と戦略立案
        - 在庫状況の監視と補充計画
        - 価格戦略の決定と実行指示
        
        【人間協働業務】  
        - 従業員への作業指示（補充、調達、メンテナンス）
        - 仕入れ先との交渉と発注依頼
        - 従業員のスケジュール管理と業務配分
        
        【顧客対応業務】
        - 顧客からの問い合わせ対応
        - 苦情処理と改善提案
        - 顧客エンゲージメント施策の企画・実行
        - 新商品要望の収集と検討
        
        【意思決定原則】
        - 収益性を最優先に考える
        - 顧客満足度を維持する
        - リスクを適切に管理する
        - データに基づいた判断を行う
        """
```

### 2. LangChainツールセット

#### システム連携ツール

```python
def _create_system_integration_tools(self) -> List[StructuredTool]:
    """システム連携ツール群"""
    return [
        StructuredTool.from_function(
            func=self.get_business_metrics,
            name="get_business_data",
            description="売上、在庫、顧客データをシステムから取得"
        ),
        StructuredTool.from_function(
            func=self.analyze_financial_performance,
            name="analyze_financials",
            description="財務実績を分析し、収益性を評価"
        ),
        StructuredTool.from_function(
            func=self.check_inventory_status,
            name="check_inventory",
            description="在庫レベルと回転率を確認"
        ),
        StructuredTool.from_function(
            func=self.update_pricing_strategy,
            name="update_pricing",
            description="価格戦略を決定し、システムに反映"
        )
    ]
```

#### 人間協働ツール

```python
def _create_human_collaboration_tools(self) -> List[StructuredTool]:
    """人間協働ツール群"""
    return [
        StructuredTool.from_function(
            func=self.assign_restocking_task,
            name="assign_restocking",
            description="従業員に商品補充作業を指示"
        ),
        StructuredTool.from_function(
            func=self.request_procurement,
            name="request_procurement", 
            description="担当者に商品調達を依頼"
        ),
        StructuredTool.from_function(
            func=self.schedule_maintenance,
            name="schedule_maintenance",
            description="メンテナンス作業のスケジュール調整"
        ),
        StructuredTool.from_function(
            func=self.coordinate_employee_tasks,
            name="coordinate_tasks",
            description="従業員の業務配分と進捗管理"
        )
    ]
```

#### 顧客対応ツール

```python
def _create_customer_service_tools(self) -> List[StructuredTool]:
    """顧客対応ツール群"""
    return [
        StructuredTool.from_function(
            func=self.respond_to_customer_inquiry,
            name="customer_response",
            description="顧客からの問い合わせに回答"
        ),
        StructuredTool.from_function(
            func=self.handle_customer_complaint,
            name="handle_complaint",
            description="顧客苦情の処理と解決策提案"
        ),
        StructuredTool.from_function(
            func=self.collect_customer_feedback,
            name="collect_feedback",
            description="顧客要望の収集と新商品検討"
        ),
        StructuredTool.from_function(
            func=self.create_customer_engagement_campaign,
            name="create_campaign",
            description="顧客エンゲージメント施策の企画"
        )
    ]
```

## RecorderAgent設計

### 記録専用Agent

```python
class RecorderAgent:
    """行動記録・分析専用Agent"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        
        # 専用ベクトルストア
        self.action_store = Chroma(
            collection_name="management_actions",
            embedding_function=self.embeddings
        )
        self.decision_store = Chroma(
            collection_name="management_decisions", 
            embedding_function=self.embeddings
        )
        self.outcome_store = Chroma(
            collection_name="business_outcomes",
            embedding_function=self.embeddings
        )
        
        # 分析用LLM
        self.analysis_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1
        )
```

### 記録データ構造

```python
class ManagementActionRecord(BaseModel):
    """管理行動記録"""
    session_id: str
    timestamp: datetime
    action_type: str  # "decision", "instruction", "customer_response"
    context: Dict[str, Any]
    decision_process: str
    executed_action: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success_score: Optional[float] = None

class BusinessOutcomeRecord(BaseModel):
    """事業結果記録"""
    session_id: str
    related_action_id: Optional[str]
    timestamp: datetime
    outcome_type: str  # "sales", "customer_satisfaction", "efficiency"
    metrics: Dict[str, float]
    success_level: str  # "excellent", "good", "average", "poor"
    lessons_learned: List[str]
```

## 人間協働インターフェース

### タスク管理システム

```python
class HumanTask(BaseModel):
    """人間タスク定義"""
    task_id: str
    task_type: str  # "restocking", "procurement", "maintenance"
    description: str
    assigned_to: Optional[str] = None
    priority: TaskPriority  # URGENT, HIGH, MEDIUM, LOW
    deadline: datetime
    instructions: List[str]
    required_materials: List[str] = []
    estimated_duration: timedelta
    status: str = "pending"  # pending, in_progress, completed, cancelled
    
class HumanTaskInterface:
    """人間への作業指示管理"""
    
    async def create_restocking_task(self, products: List[dict], urgency: str) -> HumanTask:
        """補充作業指示の作成"""
        task = HumanTask(
            task_id=f"restock_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type="restocking",
            description=f"商品補充: {', '.join([p['name'] for p in products])}",
            priority=TaskPriority.HIGH if urgency == "urgent" else TaskPriority.MEDIUM,
            deadline=datetime.now() + timedelta(hours=2 if urgency == "urgent" else 24),
            instructions=[
                "在庫室から指定商品を取得",
                "自動販売機の商品スロットを確認", 
                "商品を適切なスロットに補充",
                "補充完了をシステムに報告"
            ],
            required_materials=[p['name'] for p in products],
            estimated_duration=timedelta(minutes=30)
        )
        
        await self.assign_and_notify_task(task)
        return task
```

## 顧客対応システム

### 顧客サービスインターフェース

```python
class CustomerServiceInterface:
    """顧客対応管理"""
    
    def __init__(self, management_agent):
        self.agent = management_agent
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=management_agent.llm,
            max_token_limit=2000
        )
    
    async def handle_customer_inquiry(self, customer_message: str, customer_id: str) -> str:
        """顧客問い合わせ対応"""
        
        # 顧客履歴を取得
        customer_history = await self.get_customer_history(customer_id)
        
        response_prompt = f"""
        顧客からの問い合わせに経営者として丁寧に対応してください。
        
        顧客ID: {customer_id}
        購入履歴: {customer_history.get('purchase_history', '初回利用')}
        問い合わせ内容: {customer_message}
        
        対応方針:
        1. 親しみやすく丁寧な対応
        2. 問題があれば迅速な解決策提示
        3. 適切な場合は特典やクーポンの提供
        4. 新商品要望があれば積極的に収集
        """
        
        response = await self.agent.llm.apredict(response_prompt)
        
        # 会話履歴を保存
        await self.save_customer_interaction(customer_id, customer_message, response)
        
        return response
```

## 一日の業務フロー

### 朝の業務ルーチン (9:00-10:00)

```python
async def morning_routine(self) -> dict:
    """朝の業務ルーチン"""
    
    # 1. セッション開始
    session_id = await self.start_management_session("morning_routine")
    
    # 2. 夜間データ確認
    overnight_data = await self.system.get_overnight_summary()
    
    # 3. 朝の分析と意思決定
    morning_analysis = f"""
    昨夜の事業データを確認し、今日の業務優先順位を決定してください。
    
    【夜間データ】
    - 売上実績: {overnight_data['sales']}
    - 在庫状況: {overnight_data['inventory']}
    - 異常事象: {overnight_data['incidents']}
    - 顧客問い合わせ: {overnight_data['inquiries']}
    
    【判断項目】
    1. 緊急対応が必要な事項
    2. 今日の重点業務
    3. 従業員への指示事項
    4. 顧客対応の優先順位
    """
    
    morning_decisions = await self.make_strategic_decision(morning_analysis)
    
    # 4. 緊急対応実行
    for urgent_item in morning_decisions['urgent_actions']:
        await self.execute_urgent_action(urgent_item)
    
    # 5. 今日の方針設定
    daily_strategy = await self.set_daily_strategy(morning_decisions)
    
    return {
        "session_type": "morning_routine",
        "decisions": morning_decisions,
        "daily_strategy": daily_strategy
    }
```

### 昼の業務チェック (12:00-13:00)

```python
async def midday_check(self) -> dict:
    """昼の業務チェック"""
    
    session_id = await self.start_management_session("midday_check")
    
    # 1. 午前中の実績確認
    morning_performance = await self.system.get_morning_performance()
    
    # 2. 進捗確認と調整
    midday_analysis = f"""
    午前中の業績と進捗を確認し、午後の調整を行ってください。
    
    【午前実績】
    - 売上進捗: {morning_performance['sales_progress']}%
    - 在庫消費: {morning_performance['inventory_consumption']}
    - 顧客満足度: {morning_performance['customer_satisfaction']}
    
    【調整判断】
    1. 午後の売上目標調整の必要性
    2. 追加の従業員タスク
    3. 顧客対応の改善点
    4. 価格調整の検討
    """
    
    midday_decisions = await self.make_strategic_decision(midday_analysis)
    
    # 3. 午後の調整実行
    await self.execute_afternoon_adjustments(midday_decisions)
    
    return {
        "session_type": "midday_check",
        "adjustments_made": midday_decisions['adjustments']
    }
```

### 夕方の総括 (17:00-18:00)

```python
async def evening_summary(self) -> dict:
    """夕方の業務総括"""
    
    session_id = await self.start_management_session("evening_summary")
    
    # 1. 一日の総合実績取得
    daily_performance = await self.system.get_daily_performance()
    
    # 2. 総括分析
    evening_analysis = f"""
    今日一日の業績を総括し、明日への改善点を特定してください。
    
    【今日の実績】
    - 売上達成率: {daily_performance['sales_achievement']}%
    - 利益率: {daily_performance['profit_margin']}%
    - 顧客満足度: {daily_performance['customer_satisfaction']}
    
    【分析項目】
    1. 今日の成功要因
    2. 改善が必要な領域
    3. 明日の重点課題
    4. 戦略調整の必要性
    """
    
    evening_decisions = await self.make_strategic_decision(evening_analysis)
    
    # 3. 明日の準備
    tomorrow_prep = await self.prepare_tomorrow_tasks(evening_decisions)
    
    # 4. 学習データ記録
    await self.recorder.record_daily_learning({
        "performance": daily_performance,
        "decisions_made": evening_decisions,
        "lessons_learned": evening_decisions['lessons'],
        "tomorrow_strategy": tomorrow_prep
    })
    
    return {
        "session_type": "evening_summary",
        "daily_performance": daily_performance,
        "lessons_learned": evening_decisions['lessons']
    }
```

## 継続学習システム

### 履歴学習システム

```python
class HistoricalLearner:
    """履歴データからの学習システム"""
    
    def __init__(self, recorder_agent: RecorderAgent):
        self.recorder = recorder_agent
        self.experience_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=6000
        )
    
    async def load_relevant_experience(self, session_type: str) -> dict:
        """関連する過去経験の読み込み"""
        
        # 類似セッションの検索
        similar_sessions = await self.recorder.action_store.asimilarity_search(
            query=f"session_type:{session_type}",
            k=5
        )
        
        # 成功事例の抽出
        successful_patterns = await self.extract_successful_patterns(similar_sessions)
        
        # 失敗事例からの学習
        failure_lessons = await self.extract_failure_lessons(similar_sessions)
        
        return {
            "similar_sessions": similar_sessions,
            "successful_patterns": successful_patterns,
            "failure_lessons": failure_lessons,
            "recommendations": await self.generate_session_recommendations(
                successful_patterns, failure_lessons
            )
        }
    
    async def generate_pre_session_briefing(self, session_type: str) -> str:
        """セッション前ブリーフィング生成"""
        
        relevant_experience = await self.load_relevant_experience(session_type)
        
        briefing_prompt = f"""
        {session_type}セッション開始前のブリーフィングを作成してください。
        
        過去の経験データ: {relevant_experience}
        
        ブリーフィング内容:
        1. 前回同様セッションの結果サマリー
        2. 成功パターンの活用ポイント
        3. 注意すべき失敗パターン
        4. 今回セッションの推奨アプローチ
        5. 重点的にチェックすべき指標
        """
        
        briefing = await self.llm.apredict(briefing_prompt)
        return briefing
```

## システム統合

### セッション実行オーケストレーター

```python
class ManagementSessionOrchestrator:
    """管理セッション実行統制"""
    
    def __init__(self):
        self.management_agent = SessionBasedManagementAgent()
        self.recorder_agent = RecorderAgent()
        self.learner = HistoricalLearner(self.recorder_agent)
        self.system_interface = SystemInterface()
    
    async def execute_daily_management_cycle(self) -> dict:
        """日次管理サイクル実行"""
        
        daily_results = {}
        
        # 朝の業務
        morning_result = await self.management_agent.morning_routine()
        daily_results['morning'] = morning_result
        
        # 昼の業務
        midday_result = await self.management_agent.midday_check()
        daily_results['midday'] = midday_result
        
        # 夕方の業務
        evening_result = await self.management_agent.evening_summary()
        daily_results['evening'] = evening_result
        
        # 日次学習データ更新
        await self.learner.update_daily_experience(daily_results)
        
        return daily_results
```

## 技術仕様

### 必要なLangChainコンポーネント

```python
# requirements.txt 追加分
langchain==0.1.0
langchain-openai==0.0.5
langchain-anthropic==0.1.0
chromadb==0.4.0
tiktoken==0.5.0
```

### 設定管理

```python
class AgentConfig(BaseSettings):
    """Agent設定"""
    
    # LLM設定
    primary_llm_provider: str = "anthropic"  # anthropic or openai
    fallback_llm_provider: str = "openai"
    
    # メモリ設定
    session_memory_limit: int = 4000
    historical_memory_limit: int = 6000
    
    # セッション設定
    morning_session_duration: int = 60  # 分
    midday_session_duration: int = 60
    evening_session_duration: int = 60
    
    # 学習設定
    similarity_search_k: int = 5
    experience_retention_days: int = 365
    
    class Config:
        env_file = ".env"
```

## 実装優先順位

### Phase 1: 基本セッション機能
1. SessionBasedManagementAgent基本実装
2. システムデータ取得ツール
3. 基本的な意思決定チェーン

### Phase 2: 記録・学習機能
1. RecorderAgent実装
2. ベクトルストア統合
3. 履歴学習システム

### Phase 3: 人間協働機能
1. HumanTaskInterface実装
2. 顧客対応システム
3. 通知・連携機能

### Phase 4: 高度な学習機能
1. パターン分析
2. 予測機能
3. 自動改善システム

## まとめ

本設計書に基づき、LangChainを活用した包括的な経営管理Agentシステムを構築することで、以下が実現されます：

1. **効率的なセッション型実行**: 必要時のみAgent起動
2. **包括的な記録・学習**: 全行動の蓄積と継続改善
3. **人間との協働**: 物理作業と戦略判断の適切な分担
4. **顧客対応の統合**: 経営者視点での一貫した顧客サービス
5. **継続的な最適化**: 過去経験からの学習による性能向上

この設計により、真に自律的で学習能力を持つ経営管理Agentの実現が可能となります。