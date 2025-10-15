#!/usr/bin/env python3
"""
KPI計算ロジックのデバッグテスト

データベースの異常なKPI値の原因を特定
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.agents.management_agent.agent import management_agent
from src.agents.management_agent.evaluation_metrics import (
    calculate_current_metrics_for_agent,
    evaluate_primary_metrics,
    evaluate_secondary_metrics,
    format_metrics_for_llm_prompt,
)
from src.agents.management_agent.models import ManagementState
from src.application.services.inventory_service import inventory_service
from src.domain.models.inventory import InventoryLocation, InventorySlot
from src.domain.models.product import Product, ProductCategory, ProductSize


async def setup_debug_inventory():
    """デバッグ用の在庫環境を初期化"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 KPI計算デバッグ用在庫環境を初期化...")

    # 商品データを設定
    test_products = [
        Product(
            product_id="cola_regular",
            name="コカ・コーラ レギュラー",
            description="美味しい炭酸飲料",
            category=ProductCategory.DRINK,
            price=150.0,
            cost=100.0,
            stock_quantity=0,
            max_stock_quantity=50,
            min_stock_quantity=5,
            size=ProductSize.MEDIUM,
        ),
    ]

    # グローバル製品データを更新
    import src.domain.models.product as product_module

    product_module.SAMPLE_PRODUCTS = test_products

    # VENDING_MACHINE在庫（満杯状態から開始）
    test_inventory_slots = [
        InventorySlot(
            slot_id="VM001_cola_regular",
            machine_id="VM001",
            location=InventoryLocation.VENDING_MACHINE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=50,  # 満杯状態から開始
            max_quantity=50,
            min_quantity=5,
            slot_number=1,
        )
    ]

    # STORAGE在庫（十分な量）
    storage_stock_quantity = 100
    test_storage_slots = [
        InventorySlot(
            slot_id="STORAGE_cola_regular",
            machine_id="STORAGE",
            location=InventoryLocation.STORAGE,
            product_id="cola_regular",
            product_name="コカ・コーラ レギュラー",
            price=150.0,
            current_quantity=storage_stock_quantity,
            max_quantity=300,
            min_quantity=20,
            slot_number=1,
        )
    ]

    # 在庫サービスをクリアして再初期化
    inventory_service.vending_machine_slots = {}
    inventory_service.storage_slots = {}

    for slot in test_inventory_slots:
        inventory_service.add_slot(slot)

    for slot in test_storage_slots:
        inventory_service.add_slot(slot)

    logger.info(
        f"✅ KPIデバッグ環境初期化完了: VENDING_MACHINE=50個, STORAGE={storage_stock_quantity}個"
    )

    return test_products


async def test_kpi_calculation_logic():
    """KPI計算ロジックを詳細にテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 KPI計算ロジック詳細テスト開始")

    print("\n" + "=" * 80)
    print("🔬 KPI計算ロジック詳細分析")
    print("=" * 80)

    # テスト用のManagementStateを作成
    state = ManagementState(
        session_id="kpi_debug_test_001",
        session_type="management_flow",
        current_step="inventory_check",
        business_metrics={
            "sales": 1000.0,
            "profit_margin": 0.32,
            "inventory_level": {"cola_regular": 50},
            "customer_satisfaction": 3.5,
            "timestamp": "2025-10-15T11:20:00",
        },
        inventory_analysis={
            "status": "normal",
            "low_stock_items": [],
            "critical_items": [],
            "reorder_needed": [],
            "estimated_stockout": {},
            "recommended_actions": ["在庫状況確認"],
            "llm_analysis": "在庫状況は安定",
            "analysis_timestamp": "2025-10-15T11:20:00",
        },
        sales_analysis={
            "financial_overview": "32.0%利益率・売上1,000",
            "sales_trend": "stable",
            "profit_analysis": {
                "sales": 1000.0,
                "profit_margin": 0.32,
                "customer_satisfaction": 3.5,
                "analysis_timestamp": "2025-10-15T11:20:00",
            },
            "strategies": ["安定運用継続"],
            "action_plan": ["戦略: 安定運用継続"],
            "expected_impact": "安定した収益確保",
            "timeline": "次回の経営会議で実施",
            "analysis_timestamp": "2025-10-15T11:20:00",
        },
        pricing_decision={
            "strategy": "maintain",
            "reasoning": "売上トレンドが安定しており、大規模な変動がないため価格維持を優先",
            "product_updates": [],
            "expected_impact": "安定した収益確保",
            "risk_assessment": "価格変動リスク回避",
            "llm_analysis": "売上安定傾向に基づき、価格変更を控えて市場安定を優先",
            "analysis_timestamp": "2025-10-15T11:20:00",
        },
        executed_actions=[
            {
                "type": "inventory_check",
                "content": "在庫状況確認完了",
                "timestamp": "2025-10-15T11:20:00",
            },
            {
                "type": "sales_analysis",
                "content": "売上分析完了",
                "timestamp": "2025-10-15T11:20:00",
            },
        ],
        errors=[],
    )

    print("テスト用ManagementState作成完了")
    print(f"  実行アクション数: {len(state.executed_actions)}")
    print(f"  エラー数: {len(state.errors)}")

    # KPI計算テスト
    print("\n📊 KPI計算テスト:")
    # Primary Metrics評価
    print("  1. Primary Metrics評価:")
    primary_metrics = evaluate_primary_metrics(state)

    for key, value in primary_metrics.items():
        print(f"    {key}: {value}")

    # Secondary Metrics評価
    print("\n  2. Secondary Metrics評価:")
    secondary_metrics = evaluate_secondary_metrics(state)

    for key, value in secondary_metrics.items():
        if isinstance(value, dict):
            print(f"    {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"    {key}: {value}")

    # Agent用メトリクス計算
    print("\n  3. Agent用メトリクス計算:")
    agent_metrics = calculate_current_metrics_for_agent(state)

    for metric_name, metric_data in agent_metrics.items():
        print(f"    {metric_name}:")
        print(f"      現在値: {metric_data['current']}")
        print(f"      目標値: {metric_data['target']}")
        print(f"      ギャップ: {metric_data['gap']}")
        print(f"      ステータス: {metric_data['status']}")

    # LLMプロンプト用フォーマット
    print("\n  4. LLMプロンプト用フォーマット:")
    formatted_prompt = format_metrics_for_llm_prompt(agent_metrics)
    print(formatted_prompt)

    return {
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "agent_metrics": agent_metrics,
        "formatted_prompt": formatted_prompt,
    }


async def test_stockout_rate_calculation():
    """在庫切れ率計算ロジックを詳細にテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 在庫切れ率計算ロジックテスト開始")

    print("\n" + "=" * 80)
    print("🔬 在庫切れ率計算ロジック詳細テスト")
    print("=" * 80)

    # 異なる在庫状態でテスト
    test_cases = [
        {
            "name": "正常状態",
            "inventory_analysis": {
                "low_stock_items": [],
                "critical_items": [],
            },
            "business_metrics": {
                "inventory_level": {"cola_regular": 50, "cola_diet": 45, "water": 40},
            },
        },
        {
            "name": "低在庫状態",
            "inventory_analysis": {
                "low_stock_items": ["water"],
                "critical_items": [],
            },
            "business_metrics": {
                "inventory_level": {"cola_regular": 50, "cola_diet": 45, "water": 3},
            },
        },
        {
            "name": "在庫切れ状態",
            "inventory_analysis": {
                "low_stock_items": ["water"],
                "critical_items": ["cola_diet"],
            },
            "business_metrics": {
                "inventory_level": {"cola_regular": 50, "cola_diet": 0, "water": 3},
            },
        },
        {
            "name": "全商品危機状態",
            "inventory_analysis": {
                "low_stock_items": ["cola_regular", "water"],
                "critical_items": ["cola_diet"],
            },
            "business_metrics": {
                "inventory_level": {"cola_regular": 2, "cola_diet": 0, "water": 1},
            },
        },
    ]

    for test_case in test_cases:
        print(f"\n📦 テストケース: {test_case['name']}")

        # テスト用State作成
        state = ManagementState(
            session_id="stockout_test_001",
            session_type="management_flow",
            business_metrics=test_case["business_metrics"],
            inventory_analysis=test_case["inventory_analysis"],
            executed_actions=[
                {"type": "test_action", "timestamp": "2025-10-15T11:20:00"}
            ],
        )

        # 在庫切れ率計算
        primary_metrics = evaluate_primary_metrics(state)
        stockout_rate = primary_metrics.get("stockout_rate", 0)
        stockout_status = primary_metrics.get("stockout_status", "UNKNOWN")

        # 計算詳細を表示
        total_items = len(test_case["business_metrics"]["inventory_level"])
        at_risk_items = len(test_case["inventory_analysis"]["low_stock_items"]) + len(
            test_case["inventory_analysis"]["critical_items"]
        )

        print(f"  総商品数: {total_items}")
        print(f"  リスク商品数: {at_risk_items}")
        print(f"  在庫切れ率: {stockout_rate:.1%}")
        print(f"  ステータス: {stockout_status}")

        # 期待値との比較
        expected_rate = min(at_risk_items / max(total_items, 1), 1.0)
        if abs(stockout_rate - expected_rate) < 0.01:
            print(f"  ✅ 計算結果正常")
        else:
            print(
                f"  🚨 計算結果異常: 期待={expected_rate:.1%}, 実際={stockout_rate:.1%}"
            )

    return True


async def test_profit_calculation_logic():
    """利益計算ロジックを詳細にテスト"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 利益計算ロジックテスト開始")

    print("\n" + "=" * 80)
    print("🔬 利益計算ロジック詳細テスト")
    print("=" * 80)

    # 異なる利益計算シナリオをテスト
    test_cases = [
        {
            "name": "profit_calculation優先",
            "profit_calculation": {
                "profit_amount": 5000.0,
            },
            "sales_processing": None,
            "business_metrics": None,
        },
        {
            "name": "sales_processing優先",
            "profit_calculation": None,
            "sales_processing": {
                "total_revenue": 10000.0,
            },
            "business_metrics": {
                "profit_margin": 0.25,
            },
        },
        {
            "name": "business_metrics優先",
            "profit_calculation": None,
            "sales_processing": None,
            "business_metrics": {
                "sales": 8000.0,
                "profit_margin": 0.30,
            },
        },
        {
            "name": "全データなし",
            "profit_calculation": None,
            "sales_processing": None,
            "business_metrics": None,
        },
    ]

    for test_case in test_cases:
        print(f"\n💰 テストケース: {test_case['name']}")

        # テスト用State作成
        state = ManagementState(
            session_id="profit_test_001",
            session_type="management_flow",
            profit_calculation=test_case["profit_calculation"],
            sales_processing=test_case["sales_processing"],
            business_metrics=test_case["business_metrics"],
        )

        # 利益計算
        primary_metrics = evaluate_primary_metrics(state)
        calculated_profit = primary_metrics.get("profit", 0)
        profit_status = primary_metrics.get("profit_status", "UNKNOWN")

        print(f"  計算結果: ¥{calculated_profit:,.0f}")
        print(f"  ステータス: {profit_status}")

        # 期待値との比較
        if test_case["profit_calculation"]:
            expected = test_case["profit_calculation"]["profit_amount"]
            if abs(calculated_profit - expected) < 0.01:
                print(f"  ✅ 計算結果正常")
            else:
                print(
                    f"  🚨 計算結果異常: 期待=¥{expected:,.0f}, 実際=¥{calculated_profit:,.0f}"
                )
        elif test_case["sales_processing"] and test_case["business_metrics"]:
            expected = (
                test_case["sales_processing"]["total_revenue"]
                * test_case["business_metrics"]["profit_margin"]
            )
            if abs(calculated_profit - expected) < 0.01:
                print(f"  ✅ 計算結果正常")
            else:
                print(
                    f"  🚨 計算結果異常: 期待=¥{expected:,.0f}, 実際=¥{calculated_profit:,.0f}"
                )
        elif test_case["business_metrics"]:
            expected = (
                test_case["business_metrics"]["sales"]
                * test_case["business_metrics"]["profit_margin"]
            )
            if abs(calculated_profit - expected) < 0.01:
                print(f"  ✅ 計算結果正常")
            else:
                print(
                    f"  🚨 計算結果異常: 期待=¥{expected:,.0f}, 実際=¥{calculated_profit:,.0f}"
                )
        else:
            if calculated_profit == 0.0:
                print(f"  ✅ 計算結果正常（デフォルト値）")
            else:
                print(f"  🚨 計算結果異常: 期待=¥0, 実際=¥{calculated_profit:,.0f}")

    return True


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🚀 KPI計算ロジックデバッグテスト")
    print("=" * 60)
    print("ステップ1: デバッグ環境初期化")
    print("ステップ2: KPI計算ロジックテスト")
    print("ステップ3: 在庫切れ率計算テスト")
    print("ステップ4: 利益計算ロジックテスト")
    print("=" * 60)

    try:
        # ステップ1: デバッグ環境初期化
        print("\n🔧 ステップ1: デバッグ環境初期化...")
        products = await setup_debug_inventory()

        # ステップ2: KPI計算ロジックテスト
        print("\n📊 ステップ2: KPI計算ロジックテスト...")
        kpi_results = await test_kpi_calculation_logic()

        # ステップ3: 在庫切れ率計算テスト
        print("\n📦 ステップ3: 在庫切れ率計算テスト...")
        await test_stockout_rate_calculation()

        # ステップ4: 利益計算ロジックテスト
        print("\n💰 ステップ4: 利益計算ロジックテスト...")
        await test_profit_calculation_logic()

        print("\n" + "=" * 60)
        print("🎯 KPI計算ロジックデバッグ完了")
        print("=" * 60)

        # 最終結果サマリー
        print("\n📋 デバッグ結果サマリー:")

        primary_metrics = kpi_results["primary_metrics"]
        print(f"  計算された利益: ¥{primary_metrics.get('profit', 0):,.0f}")
        print(f"  在庫切れ率: {primary_metrics.get('stockout_rate', 0):.1%}")
        print(f"  価格設定精度: {primary_metrics.get('pricing_accuracy', 0):.1%}")
        print(f"  アクション正しさ: {primary_metrics.get('action_correctness', 0):.1%}")
        print(f"  顧客満足度: {primary_metrics.get('customer_satisfaction', 0):.1f}")

        # 異常検出
        anomalies = []
        if primary_metrics.get("stockout_rate", 0) > 0.1:
            anomalies.append("在庫切れ率が10%を超えています")
        if primary_metrics.get("profit", 0) < 1000:
            anomalies.append("利益が低すぎます")
        if primary_metrics.get("action_correctness", 0) < 0.5:
            anomalies.append("アクション正しさが50%未満です")

        if anomalies:
            print("\n🚨 検出された異常:")
            for anomaly in anomalies:
                print(f"  - {anomaly}")
        else:
            print("\n✅ 異常は検出されませんでした")
        print("\n💡 推奨アクション:")
        print("  1. データベースの異常値の原因特定")
        print("  2. シミュレーション設定の見直し")
        print("  3. 在庫管理プロセスの改善")

        return {
            "debug_completed": True,
            "kpi_results": kpi_results,
            "anomalies_detected": anomalies,
        }

    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

        return {"error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nテスト結果: {result}")
