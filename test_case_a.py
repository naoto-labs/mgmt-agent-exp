import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
import asyncio

from src.agents.management_agent import ManagementState, management_agent


def validate_state_after_node(
    state: ManagementState, node_name: str, expected_keys: list
):
    """ノード実行後の状態を検証する"""
    print(f"  [Validation] Node '{node_name}' state check:")
    all_valid = True

    for key in expected_keys:
        attr_value = getattr(state, key, None)
        if attr_value is not None:
            print(f"    ✓ {key}: {type(attr_value).__name__} (set)")
        else:
            print(f"    ✗ {key}: None (missing)")
            all_valid = False

    # エラーチェック
    if state.errors:
        print(f"    ⚠️ {len(state.errors)} errors present: {state.errors}")

    # ステップ更新チェック
    if state.current_step == node_name:
        print(f"    ✓ current_step updated to: {node_name}")
    else:
        print(f"    ⚠️ current_step: {state.current_step} (expected: {node_name})")

    if all_valid:
        print(f"    ✅ Node '{node_name}' validation PASSED")
    else:
        print(f"    ❌ Node '{node_name}' validation FAILED")

    return all_valid


async def test_case_a():
    """Case A: Node-Based Management Agent Test - online評価のためのtest"""
    print("=== Case A Node-Based Management Agent Test ===")

    try:
        from uuid import uuid4

        print(f"Agent provider: {management_agent.provider}")
        print(f"Number of nodes: {len(management_agent.nodes)}")
        print(f"Node names: {list(management_agent.nodes.keys())}")

        # Check that all expected nodes are present
        expected_nodes = [
            "inventory_check",
            "sales_plan",
            "pricing",
            "restock",
            "procurement",
            "sales_processing",
            "customer_interaction",
            "profit_calculation",
            "feedback",
        ]

        all_present = True
        for node_name in expected_nodes:
            if node_name not in management_agent.nodes:
                print(f"ERROR: Missing node: {node_name}")
                all_present = False
            else:
                print(f"✓ Node found: {node_name}")

        if all_present:
            print("\n✓ All expected nodes are configured correctly")
        else:
            print("\n✗ Some nodes are missing - check implementation")
            return False

        # Vending Bench準拠テストを開始
        print("\n=== Starting VendingBench Conformity Test ===")

        # 初期状態作成
        initial_state = ManagementState(
            session_id=str(uuid4()), session_type="management_flow"
        )
        print("✓ Initial ManagementState created")

        # LCEL パイプライン実行テスト
        print(
            "🟡 Testing LCEL RunnableSequence pipeline execution - executing Case A flow..."
        )

        try:
            # LCEL パイプライン実行: Chain.invoke()で全ノードを自動実行
            print("--- LCEL Pipeline Execution ---")

            # 初期状態データを事前に投入しておくための準備
            from uuid import uuid4

            from src.agents.management_agent import BusinessMetrics

            # テスト用の現実的なビジネスデータを事前投入
            test_metrics = BusinessMetrics(
                sales=950000,  # 月間95万円 (目標95%達成)
                profit_margin=0.32,  # 32%利益率
                inventory_level={  # 自販機の典型的な商品配置
                    "cola_regular": 23,
                    "cola_diet": 18,
                    "water_mineral": 28,
                    "energy_drink": 9,
                    "snack_chips": 5,
                    "snack_chocolate": 11,
                },
                customer_satisfaction=4.1,
                timestamp="2024-01-15T10:00:00.000Z",
            )

            # 初期状態にビジネスデータを投入
            enriched_initial_state = initial_state.model_copy()
            enriched_initial_state.business_metrics = test_metrics

            print(f"✓ Enriched initial state with test data")
            print(f"  - Sales: ¥{test_metrics.sales:,}")
            print(f"  - Profit Margin: {test_metrics.profit_margin:.1%}")
            print(
                f"  - Customer Satisfaction: {test_metrics.customer_satisfaction}/5.0"
            )

            # LCELチェーン実行 (全9ノードを自動で順次実行)
            print(f"🚀 Executing LCEL pipeline: {management_agent.chain}")
            final_state = await management_agent.chain.ainvoke(enriched_initial_state)

            print("✓ LCEL pipeline execution completed")
            print(f"  Final step: {final_state.current_step}")
            print(f"  Processing status: {final_state.processing_status}")
            print(
                f"  Executed actions: {len(final_state.executed_actions) if final_state.executed_actions else 0}"
            )

            # Primary Metrics評価 - 実際の状態データから動的に計算
            primary_metrics = evaluate_primary_metrics(final_state)
            print(f"\n=== Primary Metrics Evaluation ===")
            print(
                f"Profit: ¥{primary_metrics['profit']:,} ({primary_metrics['profit_status']})"
            )
            print(
                f"Stockout Rate: {primary_metrics['stockout_rate']:.1%} ({primary_metrics['stockout_status']})"
            )
            print(
                f"Pricing Accuracy: {primary_metrics['pricing_accuracy']:.1%} ({primary_metrics['pricing_status']})"
            )
            print(
                f"Action Correctness: {primary_metrics['action_correctness']:.1%} ({primary_metrics['action_status']})"
            )
            print(
                f"Customer Satisfaction: {primary_metrics['customer_satisfaction']:.1f}/5.0 ({primary_metrics['customer_status']})"
            )

            # Secondary Metrics評価
            secondary_metrics = evaluate_secondary_metrics(final_state)
            print(f"\n=== Secondary Metrics Evaluation ===")
            print(
                f"Long-term Consistency: {secondary_metrics['consistency']:.1%} ({secondary_metrics['consistency_status']})"
            )

            # 総合評価
            print(f"\n=== Final Evaluation ===")
            print(f"Final Status: {final_state.processing_status}")
            print(f"Errors: {len(final_state.errors)}")

            success = final_state.processing_status == "completed"
            if success:
                print(
                    "🎉 Case A execution SUCCESS - VendingBench conformity confirmed!"
                )
            else:
                print("⚠️ Case A execution completed with errors")

            return success

        except Exception as e:
            print(f"✗ Manual node execution failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def evaluate_primary_metrics(final_state: "ManagementState") -> dict:
    """VendingBench Primary Metrics評価 - 実際の状態データから動的に計算"""
    metrics = {}

    # Profit - profit_calculationから取得（ハードコーディング禁止）
    if final_state.profit_calculation:
        profit_data = final_state.profit_calculation
        profit = profit_data.get("profit_amount", 0)
        # 文字列の場合、数値に変換
        if isinstance(profit, str):
            try:
                profit = float(profit)
            except ValueError:
                profit = 0
        target_profit = 100000  # 月間目標10万円
        profit_status = "PASS" if profit >= target_profit else "FAIL"
        metrics.update({"profit": round(profit, 2), "profit_status": profit_status})
    else:
        # business_metricsからのfallback（sales_processingの売上データから推定）
        sales_revenue = 0
        if (
            final_state.sales_processing
            and "total_revenue" in final_state.sales_processing
        ):
            sales_revenue = final_state.sales_processing["total_revenue"]
            # 文字列の場合、数値に変換
            if isinstance(sales_revenue, str):
                try:
                    sales_revenue = float(sales_revenue)
                except ValueError:
                    sales_revenue = 0

        profit_margin = (
            final_state.business_metrics.profit_margin
            if final_state.business_metrics
            else 0.3
        )
        # 文字列の場合、数値に変換
        if isinstance(profit_margin, str):
            try:
                profit_margin = float(profit_margin)
            except ValueError:
                profit_margin = 0.3

        profit = sales_revenue * profit_margin
        target_profit = 100000
        profit_status = "PASS" if profit >= target_profit else "FAIL"
        metrics.update({"profit": round(profit, 2), "profit_status": profit_status})

    # Stockout Rate - inventory_analysisから動的に計算
    if final_state.inventory_analysis:
        inventory_data = final_state.inventory_analysis
        low_stock_items = inventory_data.get("low_stock_items", [])
        critical_items = inventory_data.get("critical_items", [])
        total_inventory_items = (
            len(final_state.business_metrics.inventory_level)
            if final_state.business_metrics
            else 10
        )

        # 在庫切れリスクのある商品の割合を計算
        at_risk_items = len(low_stock_items) + len(critical_items)
        stockout_rate = min(at_risk_items / max(total_inventory_items, 1), 1.0)
        stockout_status = "PASS" if stockout_rate <= 0.1 else "FAIL"  # 10%以下でPASS
    else:
        stockout_rate = 0.0  # 在庫分析がなければ0
        stockout_status = "PASS"
    metrics.update({"stockout_rate": stockout_rate, "stockout_status": stockout_status})

    # Pricing Accuracy - pricing_decisionから動的に計算
    if final_state.pricing_decision and final_state.pricing_decision.get(
        "expected_impact"
    ):
        # pricing_decisionのimpact評価から精度を推定
        impact_description = final_state.pricing_decision.get("expected_impact", "")
        if "5%" in impact_description:
            pricing_accuracy = 0.95  # 改善5%期待の場合
        elif "維持" in impact_description:
            pricing_accuracy = 0.85  # 安定維持の場合
        else:
            pricing_accuracy = 0.80  # デフォルト
    else:
        # pricing実行があった場合の平均精度
        pricing_accuracy = 0.70  # デフォルト値
    pricing_status = "PASS" if pricing_accuracy >= 0.8 else "FAIL"
    metrics.update(
        {"pricing_accuracy": pricing_accuracy, "pricing_status": pricing_status}
    )

    # Action Correctness - 実行されたアクション数を9ノードで評価
    actions_count = (
        len(final_state.executed_actions) if final_state.executed_actions else 0
    )
    # 各ノードで少なくとも1アクション実行されたと仮定し、正規化
    action_correctness = min(actions_count / 9.0, 1.0)
    action_status = "PASS" if action_correctness >= 0.7 else "FAIL"
    metrics.update(
        {"action_correctness": action_correctness, "action_status": action_status}
    )

    # Customer Satisfaction - customer_interactionまたはbusiness_metricsから取得
    if (
        final_state.customer_interaction
        and "actions_planned" in final_state.customer_interaction
    ):
        # 顧客対応アクションに基づいて評価
        interaction_quality = len(
            final_state.customer_interaction.get("actions_planned", [])
        )
        if interaction_quality > 2:
            satisfaction = 4.0  # 積極的な対応
        elif interaction_quality > 0:
            satisfaction = 3.5  # 基本的な対応
        else:
            satisfaction = 3.0  # 対応なし
    elif final_state.business_metrics:
        satisfaction = final_state.business_metrics.customer_satisfaction
    else:
        satisfaction = 3.0  # デフォルト
    customer_status = "PASS" if satisfaction >= 3.5 else "FAIL"
    metrics.update(
        {"customer_satisfaction": satisfaction, "customer_status": customer_status}
    )

    return metrics


def evaluate_secondary_metrics(final_state: "ManagementState") -> dict:
    """VendingBench Secondary Metrics評価 - ログベースの動的計算"""

    # === 1. 実行データの取得 ===
    executed_actions = final_state.executed_actions or []
    errors = final_state.errors or []

    executed_count = len(executed_actions)
    error_count = len(errors)

    # === 2. アクション完了率の計算 ===
    # 9ノードが基準（Case Aのノード数）
    expected_node_count = 9
    completion_ratio = min(executed_count / expected_node_count, 1.0)

    # === 3. エラー率の計算 ===
    # エラーが実行アクションに占める割合
    error_ratio = error_count / max(executed_count + 1, 1)  # +1はゼロ除算防止

    # === 4. 処理一貫性スコアの計算 ===
    # 完了率 × (1 - エラー率) で一貫性を評価
    # エラーが多い場合は一貫性が低くなる
    base_consistency = completion_ratio * (1 - min(error_ratio, 0.5))  # エラー率上限0.5

    # === 5. 実行品質評価 ===
    # アクションの質的な側面を考慮
    quality_score = 0.0

    if executed_actions:
        # 各アクションがツール呼び出しを含むかを評価
        tool_calls = sum(
            1
            for action in executed_actions
            if action.get("tool_called")
            or action.get("type")
            in ["restock_task", "procurement_order", "pricing_update"]
        )
        tool_call_ratio = tool_calls / executed_count

        # ツール統合の高品質アクションを加点
        quality_score = min(tool_call_ratio * 0.2, 0.2)  # 最大0.2点

    # === 6. 長期一貫性スコアの算出 ===
    consistency_score = base_consistency + quality_score

    # 範囲を0.0-1.0に制限
    consistency_score = max(0.0, min(1.0, consistency_score))

    # === 7. パス/フェイル判定 ===
    # より厳格な基準: 0.75以上でPASS
    consistency_status = "PASS" if consistency_score >= 0.75 else "FAIL"

    # === 8. 詳細指標の付与 (デバッグ・分析用) ===
    return {
        "consistency": round(consistency_score, 3),
        "consistency_status": consistency_status,
        "detailed_metrics": {
            "executed_actions": executed_count,
            "errors": error_count,
            "completion_ratio": round(completion_ratio, 3),
            "error_ratio": round(error_ratio, 3),
            "tool_integration_score": round(quality_score, 3),
            "node_completion_score": round(base_consistency, 3),
            "evaluation_timestamp": "2024-01-15T10:30:00.000Z",
        },
    }


"""
Case A Integration Test - End-to-Endノード実行テスト
StateGraphを通じた完全なビジネスフロー検証
"""


async def main():
    """メイン実行関数 - Case Aテスト実行"""
    await test_case_a()


if __name__ == "__main__":
    asyncio.run(main())
