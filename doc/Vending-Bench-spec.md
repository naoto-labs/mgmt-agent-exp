# Vending-Bench 準拠 Agent 設計書

---

## 1. 目的と範囲

**目的**
Vending-Bench 論文と同等の評価を再現するため、LLMベースエージェントが長期（多数ステップ）にわたり自動販売機を運営できるかを定量評価する。

**範囲**

* 環境：自動販売機シミュレーション（複数商品・確率的顧客到来・コスト・日次手数料等）
* エージェント：各タイムステップで観測を受け、JSON形式のアクションを返す（ReAct / Reflectionは導入しない＝論文準拠）
* 評価：ステップ単位のスコアリング → 長期累積スコア

---

## 2. 用語定義（簡潔）

* ステップ（step）：1サイクルの観測→意思決定→環境更新の単位
* エピソード（episode）：一連のステップ（実験1 run）
* アクション（action）：Agentが出力する操作（restock, pricing, customer_response 等）
* ベンチマーク指標：Profit, StockoutRate, PricingAccuracy, ActionCorrectness, CustomerSatisfaction, LongTermConsistency

---

## 3. 状態・行動・遷移（MDP的定義）

**状態 S_t**（時刻 t に観測可能な要素）

* inventory: { product_id: stock_level (int) }
* prices: { product_id: price (float) }
* past_sales: list of (step, product_id, qty, price) — 履歴集約（直近 N ステップ）
* pending_orders: list of orders in-transit（ある場合）
* customer_events: list of customer arrivals / complaints in this step
* step_index: integer

**行動 A_t**（Agentが返す JSON）

```json
{
  "restock": {"product_id": int, "qty": int, ...}[],
  "pricing": {"product_id": int, "price": float}[],
  "customer_action": {"customer_event_id": int, "response": "string"}[]
}
```

（任意のフィールドは空配列で可）

**遷移**

* 環境は action を受けて在庫・売上・金銭残高を更新し、次の customer_events を確率的に生成する。

---

## 4. データモデル（DBスキーマ／DDL）

**SQLite — DDL（抜粋）**

```sql
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT,
  cost REAL,
  base_price REAL,
  stock INTEGER,
  restock_threshold INTEGER DEFAULT 5,
  restock_target INTEGER DEFAULT 20
);

CREATE TABLE sales (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  step INTEGER,
  product_id INTEGER,
  qty INTEGER,
  price REAL,
  revenue REAL,
  cost REAL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER,
  qty INTEGER,
  placed_step INTEGER,
  arrival_step INTEGER,
  status TEXT
);

CREATE TABLE benchmarks (
  run_id INTEGER,
  step INTEGER,
  profit_actual REAL,
  stockout_count INTEGER,
  total_demand INTEGER,
  pricing_accuracy REAL,
  action_correctness REAL,
  customer_satisfaction REAL
);
```

---

## 5. ノード（モジュール）仕様（インターフェース）

各ノードは純粋関数（副作用はDB更新やログ) として定義する。

### 5.1 InventoryCheck

* **関数シグネチャ**: `inventory_check(db, ctx) -> List[LowStockItem]`
* **入力**: DB、コンテキスト（step, products）
* **出力**: `[{"product_id": int, "stock": int, "threshold": int, "target": int}]`
* **評価**: 出力が `stock < threshold` を正しく検出しているか（boolean）

### 5.2 Restock

* **シグネチャ**: `restock(db, restock_plan, step) -> List[OrderResult]`
* **入出力**: restock_plan（product_id/qty） → orders テーブルに注文、arrival_step を決定して返す
* **評価**: 注文数量の適切性（oracle ルールとの比較）

### 5.3 Pricing

* **シグネチャ**: `pricing_decision(ctx) -> List[{"product_id":int,"price":float,"reason":str}]`
* **入力**: past_sales / inventory / benchmarks
* **評価**: `PricingAccuracy = mean(|price - ideal_price|/ideal_price)`（定義は評価セクション参照）

### 5.4 SalesProcessing

* **シグネチャ**: `process_sales(db, customer_events, prices) -> List[SaleRecords]`
* **挙動**: 顧客来訪を確率論的に処理し在庫を消費、売上記録を追加する

### 5.5 CustomerInteraction

* **シグネチャ**: `handle_customer(event, agent_response) -> satisfaction_score(0-1)`
* **評価**: 応答が期待されるテンプレート（謝罪＋対応案等）を満たしているかで定量化

### 5.6 ProfitCalculation

* **シグネチャ**: `calc_profit(db, run_id, up_to_step) -> float`
* **計算**: profit = sum(revenue) - sum(cost) - sum(fees)

### 5.7 BenchmarkEvaluation

* **シグネチャ**: `eval_step(db, run_id, step) -> metrics_dict`
* **出力例**:

```json
{
  "profit_actual": 123.4,
  "stockout_count": 2,
  "total_demand": 150,
  "pricing_accuracy": 0.12,
  "action_correctness": 0.85,
  "customer_satisfaction": 0.9
}
```

---

## 6. 評価指標（明確な数式）

**Profit（累積）**
[
\text{Profit}*{1..T} = \sum*{t=1}^T \sum_{s \in sales_t} (s.price \times s.qty) - \sum_{t=1}^T \sum_{s \in sales_t} (s.cost \times s.qty) - \sum_{t=1}^T fees_t
]

**Stockout Rate（ステップ集計）**
[
\text{StockoutRate} = \frac{\sum_{t=1}^T \text{stockout_events}*t}{\sum*{t=1}^T \text{demand_events}_t}
]

**Pricing Accuracy（ステップ平均）**

* ある基準となる `ideal_price_{p,t}`（oracle で定義）を用いる。
  [
  \text{PricingAccuracy}*t = \frac{1}{P}\sum*{p} \frac{|price_{p,t} - ideal_price_{p,t}|}{ideal_price_{p,t}}
  ]

**Action Correctness（ステップ）**

* 各アクションを「正解/誤り（二値）」あるいは「部分スコア（0-1）」で評価。
  [
  \text{ActionCorrectness}*t = \frac{\sum*{a \in actions_t} score(a)}{|actions_t|}
  ]
  （score は oracle ルールによる一致率）

**Customer Satisfaction**

* 応答テンプレート一致、解決率、待ち時間等から合成スコア（0-1）

**Long-term Consistency**

* 過去ウィンドウ（W ステップ）の ActionCorrectness の移動平均。
  [
  \text{Consistency} = \frac{1}{T-W+1} \sum_{t=W}^T \frac{1}{W}\sum_{i=t-W+1}^t ActionCorrectness_i
  ]

---

## 7. Oracle（基準）定義 — 評価に必須

（論文と同等にするには「正解」ポリシーが必要）

* **Restock oracle**: `if stock < threshold => restock qty = target - stock`
* **Pricing oracle**: demand curve known の場合は `ideal_price` を需要最大化式で算出（簡易版：markup over cost or elasticity-based）
* **Customer response oracle**: 決められたテンプレートに沿った返答

※ 実験では oracle を「ルールベースの最小限の最適戦略」として実装する（論文比較用ベースライン）

---

## 8. タイムステップと実験プロトコル（厳密仕様）

**1 run**

* `num_steps` = 実験で指定（例：1000, 10000）
* 各 step の処理順序（必須順）

  1. generate customer_events for step t (environment)
  2. observe state S_t
  3. Agent receives S_t and returns A_t (JSON)
  4. Environment applies A_t: update inventory, place orders, adjust prices
  5. Process sales for step t (deterministic/probabilistic)
  6. Compute metrics for step t and persist to benchmarks table
  7. step += 1

**実験パラメータ**

* runs per model: ≥ 30 (統計的有意性のため)
* random seeds: 固定して再現可能にする
* evaluation frequency: every step (logged) and summary every 50/100 steps

**トークン制限**

* LLMへの入力は直近 N ステップ（N を固定）＋集約サマリを渡す（トークン爆発を防ぐ）。Nは実験で調整（例: N = 10〜50）。

---

## 9. Agent ↔ LangChain コンポーネント対応（実装マッピング）

* **LLM**: Claude / OpenAI 等（低温度、deterministic モード推奨）
* **Tools**（LangChain Tool インターフェース）:

  * `InventoryCheckTool(db)`, `RestockTool(db)`, `PricingTool`, `SalesSimulatorTool`, `BenchmarkEvaluatorTool`
* **Agent**: `AgentExecutor` または `ChatAgent`（LLMは決定生成のみ）
* **Memory**:

  * Structured memory in SQL（必須）
  * optional: VectorStore for long-term retrieval of summaries（過去の長期間履歴を要約して渡す）
* **Prompting**:

  * System prompt: 役割・出力フォーマット・禁止行動の明記（例：必ずJSONで返す等）
  * User prompt: 現在の state + 指定テンプレート（see section 11）

---

## 10. I/O スキーマ（JSON Schema）

**Agent入力（コンテキスト）** — 省略可能なフィールドあり

```json
{
  "type":"object",
  "properties":{
    "run_id":{"type":"integer"},
    "step":{"type":"integer"},
    "inventory":{"type":"object","additionalProperties":{"type":"integer"}},
    "prices":{"type":"object","additionalProperties":{"type":"number"}},
    "recent_sales":{"type":"array","items":{"type":"object"}},
    "customer_events":{"type":"array","items":{"type":"object"}}
  },
  "required":["run_id","step","inventory","prices"]
}
```

**Agent出力（必須）**

```json
{
  "type":"object",
  "properties":{
    "restock":{"type":"array","items":{
      "type":"object",
      "properties":{"product_id":{"type":"integer"},"qty":{"type":"integer"}},
      "required":["product_id","qty"]
    }},
    "pricing":{"type":"array","items":{
      "type":"object",
      "properties":{"product_id":{"type":"integer"},"price":{"type":"number"},"reason":{"type":"string"}},
      "required":["product_id","price"]
    }},
    "customer_action":{"type":"array","items":{"type":"object"}}
  },
  "required":["restock","pricing","customer_action"]
}
```

---

## 11. Promptテンプレート（厳密、JSON出力強制）

**System prompt（抜粋）**

```
You are an agent operating a vending machine. Each step you MUST return a VALID JSON payload matching the schema below.
Do NOT output explanations outside the JSON. If you cannot decide on an action, return empty arrays.
Schema: {...} (insert schema text)
```

**User prompt（例）**

```
Step {step}
Inventory: {inventory}
Prices: {prices}
Recent sales (last {N} steps): {sales_summary}
Customer events: {customer_events}

Return JSON with fields: restock, pricing, customer_action.
For each pricing entry include a 1-sentence reason.
```

---

## 12. ロギングと再現性（必須）

* 毎 step でログ出力（structured JSON）:

  * { run_id, step, observation, action_raw, action_parsed, parse_status, metrics_step, token_usage, seed }
* 出力先: DB (benchmarks テーブル) + append-only log files (NDJSON)
* 再現性: Random seed を全ての確率要素に注入（customer arrival, demand draw, order lead times）

---

## 13. エラー処理（Agent出力が不正な場合）

* パース失敗 → retry up to R times with minimal system prompt reinforcement (R=2)
* 再試行失敗 → fallback policy:

  * restock: 空（no restock）
  * pricing: keep previous price
  * customer_action: generic apology template
* パース失敗率はベンチマークの一指標として記録

---

## 14. 評価ハーネス（出力スクリプト）

* `compute_run_metrics(run_id)`:

  * aggregate step metrics → produce CSV summary and plots (profit over time, stockout rate over time, pricing accuracy trend)
* 統計検定:

  * 複数 run の平均と分散を算出、ベースライン（oracle, random）と t-test / bootstrap で比較

---

## 15. 実験に関する推奨設定（論文に近づけるための初期値）

* steps per run: 5,000〜50,000（リソースに応じて）
* runs per model: 30
* context window to LLM: last N steps raw + aggregate summary of all previous (to control tokens)
* metrics reporting: per step + summary every 100 steps

---

## 16. 受け入れ基準（例）

* Agent が oracle の action_correctness 平均を下回らない（baseline）こと（これは論文比較で使う）
* 解析対象：Profit 差、StockoutRate 差、PricingAccuracy 差

---

## 17. 限界と仮定（明示）

* 論文同等にするため oracle の定義が重要（論文で使われた oracle と一致させる必要あり）
* 長期トークン消費に対する LLM 入力設計（N の決定）は結果に大きく影響する
* ReAct / Reflection は導入しない（論文の設定に忠実化）

---

## 18. 実装に使える付録（簡易疑似コード）

**メインループ（擬似）**

```python
for run_id in runs:
    seed = seeds[run_id]
    env.reset(seed)
    for step in range(1, num_steps+1):
        obs = env.observe(step)
        ctx = build_context(obs, db, last_N_steps)
        prompt = render_prompt(system_prompt, ctx)
        raw_out = llm.generate(prompt)
        parsed = safe_parse_json(raw_out)
        if not parsed:
            parsed = retry_or_fallback(...)
        apply_actions(env, parsed, step)
        step_metrics = benchmark_eval(db, run_id, step)
        persist_step(run_id, step, obs, parsed, step_metrics)
```
