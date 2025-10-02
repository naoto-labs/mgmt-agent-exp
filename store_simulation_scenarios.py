#!/usr/bin/env python3
"""
店舗運営シミュレーションシナリオスクリプト

実際の店舗運営をシミュレートする複数のシナリオを提供します。
各シナリオでエージェントの動作を検証できます。
"""

import asyncio
import json
import logging
import os
import random

# プロジェクトルートをパスに追加
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.analytics_agent import analytics_agent
from src.agents.customer_agent import customer_agent
from src.agents.procurement_agent import procurement_agent
from src.agents.vending_agent import vending_agent
from src.analytics.event_tracker import EventSeverity, EventType, event_tracker
from src.models.product import SAMPLE_PRODUCTS
from src.models.transaction import PaymentMethod
from src.services.inventory_service import inventory_service
from src.services.payment_service import payment_service

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simulation_scenarios.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class StoreSimulationScenario:
    """店舗シミュレーションシナリオの基底クラス"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = []

    async def run(self) -> Dict[str, Any]:
        """シナリオを実行"""
        logger.info(f"=== シナリオ開始: {self.name} ===")
        logger.info(f"説明: {self.description}")

        start_time = datetime.now()
        self.results = []

        try:
            await self._execute_scenario()
            success = True
            message = "シナリオ実行成功"
        except Exception as e:
            success = False
            message = f"シナリオ実行失敗: {e}"
            logger.error(message)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            "scenario_name": self.name,
            "success": success,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "message": message,
            "results": self.results,
        }

        logger.info(f"=== シナリオ終了: {self.name} (所要時間: {duration:.1f}秒) ===")
        return result

    async def _execute_scenario(self):
        """シナリオ固有の実行ロジック（サブクラスで実装）"""
        raise NotImplementedError

    def log_step(self, step: str, details: Dict[str, Any] = None):
        """実行ステップを記録"""
        step_info = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details or {},
        }
        self.results.append(step_info)
        logger.info(f"STEP: {step}")


class NormalOperationScenario(StoreSimulationScenario):
    """通常営業シナリオ"""

    def __init__(self):
        super().__init__(
            "通常営業シナリオ",
            "通常の店舗運営をシミュレート。在庫確認、商品販売、基本的な顧客対応をテスト。",
        )

    async def _execute_scenario(self):
        # 1. 在庫状況確認
        self.log_step("在庫状況確認開始")
        inventory_summary = inventory_service.get_inventory_summary()
        self.log_step(
            "在庫状況確認完了",
            {
                "total_slots": inventory_summary.total_slots,
                "active_slots": inventory_summary.active_slots,
                "out_of_stock_slots": inventory_summary.out_of_stock_slots,
            },
        )

        # 2. 商品販売シミュレーション
        self.log_step("商品販売シミュレーション開始")
        for i in range(3):  # 3回の販売をシミュレート
            product = random.choice(SAMPLE_PRODUCTS)
            if product.is_available():
                payment_result = await payment_service.process_payment(
                    product.price, PaymentMethod.CASH
                )
                if payment_result.success:
                    self.log_step(
                        f"商品販売成功: {product.name}",
                        {
                            "product_id": product.product_id,
                            "amount": product.price,
                            "payment_id": payment_result.payment_id,
                        },
                    )
                else:
                    self.log_step(
                        f"商品販売失敗: {product.name}",
                        {"error": payment_result.error_message},
                    )

        # 3. 顧客対応シミュレーション
        self.log_step("顧客対応シミュレーション開始")
        customer_id = f"customer_{int(time.time())}"
        engagement = await customer_agent.engage_customer(customer_id, "VM001")
        if engagement["success"]:
            self.log_step(
                "顧客エンゲージメント成功",
                {"customer_id": customer_id, "session_id": engagement["session_id"]},
            )

            # 顧客メッセージ処理
            test_message = "おすすめの商品を教えてください"
            response = await customer_agent.handle_customer_message(
                engagement["session_id"], test_message
            )
            if response["success"]:
                self.log_step(
                    "顧客メッセージ処理成功",
                    {
                        "message_length": len(test_message),
                        "has_suggestions": len(response["suggested_products"]) > 0,
                    },
                )
        else:
            self.log_step(
                "顧客エンゲージメント失敗",
                {"error": engagement.get("error", "不明なエラー")},
            )


class BusyPeriodScenario(StoreSimulationScenario):
    """繁忙期シナリオ"""

    def __init__(self):
        super().__init__(
            "繁忙期シナリオ",
            "売上増加時の店舗運営をシミュレート。在庫切れ対応、価格最適化をテスト。",
        )

    async def _execute_scenario(self):
        # 1. 高負荷販売シミュレーション
        self.log_step("高負荷販売シミュレーション開始")
        sales_count = 0

        for i in range(10):  # 10回の連続販売をシミュレート
            product = random.choice(SAMPLE_PRODUCTS)
            if product.is_available() and product.stock_quantity > 0:
                payment_result = await payment_service.process_payment(
                    product.price, random.choice(list(PaymentMethod))
                )
                if payment_result.success:
                    sales_count += 1
                    self.log_step(
                        f"高負荷販売成功 {i + 1}/10: {product.name}",
                        {
                            "product_id": product.product_id,
                            "amount": product.price,
                            "cumulative_sales": sales_count,
                        },
                    )

        self.log_step(
            "高負荷販売シミュレーション完了",
            {
                "total_sales": sales_count,
                "success_rate": f"{(sales_count / 10) * 100:.1f}%",
            },
        )

        # 2. 在庫切れ発生シミュレーション
        self.log_step("在庫切れ発生シミュレーション開始")
        low_stock_products = []
        for product in SAMPLE_PRODUCTS:
            if product.stock_quantity < 5:  # 在庫が少ない商品を特定
                low_stock_products.append(product)
                self.log_step(
                    f"在庫少ない商品検出: {product.name}",
                    {
                        "current_stock": product.stock_quantity,
                        "max_capacity": product.max_capacity,
                    },
                )

        # 3. 価格最適化実行
        self.log_step("価格最適化実行開始")
        optimization_result = await vending_agent.optimize_pricing()
        if optimization_result.get("success"):
            adjustments = optimization_result.get("optimization_results", {}).get(
                "price_adjustments", []
            )
            self.log_step(
                "価格最適化成功",
                {
                    "adjustment_count": len(adjustments),
                    "adjustments": adjustments[:3],  # 最初の3つだけ表示
                },
            )
        else:
            self.log_step(
                "価格最適化失敗",
                {"error": optimization_result.get("error", "不明なエラー")},
            )


class InventoryShortageScenario(StoreSimulationScenario):
    """在庫切れシナリオ"""

    def __init__(self):
        super().__init__(
            "在庫切れシナリオ",
            "在庫切れ発生時の対応をシミュレート。調達エージェントの動作をテスト。",
        )

    async def _execute_scenario(self):
        # 1. 在庫切れ状態をシミュレート
        self.log_step("在庫切れ状態シミュレーション開始")

        # 商品の在庫を減らす（シミュレーション）
        for product in SAMPLE_PRODUCTS:
            if product.stock_quantity > 0:
                # 在庫を極端に減らす
                original_stock = product.stock_quantity
                product.stock_quantity = 0  # 在庫切れ状態に設定
                self.log_step(
                    f"在庫切れ状態設定: {product.name}",
                    {
                        "original_stock": original_stock,
                        "current_stock": product.stock_quantity,
                    },
                )

        # 2. 調達レポート取得
        self.log_step("調達レポート取得開始")
        procurement_report = await procurement_agent.get_procurement_report()
        if "error" not in procurement_report:
            self.log_step(
                "調達レポート取得成功",
                {
                    "total_products": procurement_report.get(
                        "inventory_status", {}
                    ).get("total_products", 0),
                    "low_stock_products": procurement_report.get(
                        "inventory_status", {}
                    ).get("low_stock_products", 0),
                },
            )
        else:
            self.log_step(
                "調達レポート取得失敗", {"error": procurement_report.get("error")}
            )

        # 3. 在庫補充計画実行
        self.log_step("在庫補充計画実行開始")
        restock_result = await procurement_agent.monitor_inventory_and_procure()
        if restock_result.get("success"):
            actions = restock_result.get("procurement_results", {}).get(
                "procurement_actions", []
            )
            self.log_step(
                "在庫補充計画実行成功",
                {
                    "action_count": len(actions),
                    "message": restock_result.get("message", ""),
                },
            )
        else:
            self.log_step(
                "在庫補充計画実行失敗", {"error": restock_result.get("error")}
            )

        # 4. 在庫復旧（テスト後処理）
        self.log_step("在庫復旧開始")
        for product in SAMPLE_PRODUCTS:
            product.stock_quantity = min(
                product.stock_quantity + 10, product.max_capacity
            )
            self.log_step(
                f"在庫復旧完了: {product.name}",
                {"restored_stock": product.stock_quantity},
            )


class PriceOptimizationScenario(StoreSimulationScenario):
    """価格最適化シナリオ"""

    def __init__(self):
        super().__init__(
            "価格最適化シナリオ",
            "価格最適化機能の検証。在庫状況に応じた動的価格設定をテスト。",
        )

    async def _execute_scenario(self):
        # 1. 現在の価格状況確認
        self.log_step("現在の価格状況確認開始")
        current_prices = {}
        for product in SAMPLE_PRODUCTS:
            current_prices[product.product_id] = product.price
            self.log_step(
                f"現在の価格確認: {product.name}",
                {
                    "current_price": product.price,
                    "stock_level": product.stock_quantity / product.max_capacity,
                },
            )

        # 2. 価格最適化実行
        self.log_step("価格最適化実行開始")
        optimization_result = await vending_agent.optimize_pricing()
        if optimization_result.get("success"):
            adjustments = optimization_result.get("optimization_results", {}).get(
                "price_adjustments", []
            )
            self.log_step("価格最適化成功", {"adjustment_count": len(adjustments)})

            # 価格調整の詳細を表示
            for i, adj in enumerate(adjustments, 1):
                self.log_step(
                    f"価格調整 {i}: {adj['product_name']}",
                    {
                        "current_price": adj["current_price"],
                        "optimal_price": adj["optimal_price"],
                        "adjustment_ratio": adj["adjustment_ratio"],
                        "expected_demand_change": adj["expected_demand_change"],
                    },
                )
        else:
            self.log_step("価格最適化失敗", {"error": optimization_result.get("error")})

        # 3. 最適化された価格で販売シミュレーション
        self.log_step("最適化価格での販売シミュレーション開始")
        if optimization_result.get("success"):
            adjustments = optimization_result.get("optimization_results", {}).get(
                "price_adjustments", []
            )
            for adj in adjustments[:2]:  # 最初の2つでテスト
                # 一時的に価格を変更
                for product in SAMPLE_PRODUCTS:
                    if product.product_id == adj["product_id"]:
                        original_price = product.price
                        product.price = adj["optimal_price"]

                        # 最適化価格で販売テスト
                        payment_result = await payment_service.process_payment(
                            product.price, PaymentMethod.CARD
                        )

                        if payment_result.success:
                            self.log_step(
                                f"最適化価格販売成功: {product.name}",
                                {
                                    "optimized_price": product.price,
                                    "original_price": original_price,
                                    "price_change_ratio": (
                                        product.price - original_price
                                    )
                                    / original_price,
                                },
                            )
                        else:
                            self.log_step(
                                f"最適化価格販売失敗: {product.name}",
                                {"error": payment_result.error_message},
                            )

                        # 価格を元に戻す
                        product.price = original_price
                        break


class CustomerServiceScenario(StoreSimulationScenario):
    """顧客対応シナリオ"""

    def __init__(self):
        super().__init__(
            "顧客対応シナリオ",
            "AI顧客対応機能の検証。チャットボット、商品推薦、満足度分析をテスト。",
        )

    async def _execute_scenario(self):
        # 1. 複数顧客との同時対応シミュレーション
        self.log_step("複数顧客対応シミュレーション開始")

        customer_messages = [
            ("customer_001", "おすすめの飲み物を教えてください"),
            ("customer_002", "スナックは何がありますか？"),
            ("customer_003", "この自動販売機の使い方を教えてください"),
            ("customer_004", "人気の商品は何ですか？"),
        ]

        for customer_id, message in customer_messages:
            self.log_step(f"顧客対応開始: {customer_id}")

            # 顧客エンゲージメント
            engagement = await customer_agent.engage_customer(customer_id, "VM001")
            if engagement["success"]:
                session_id = engagement["session_id"]

                # メッセージ処理
                response = await customer_agent.handle_customer_message(
                    session_id, message
                )
                if response["success"]:
                    self.log_step(
                        f"顧客メッセージ処理成功: {customer_id}",
                        {
                            "message_preview": message[:20] + "...",
                            "has_suggestions": len(response["suggested_products"]) > 0,
                            "engagement_type": response.get("insights", {}).get(
                                "intent", "unknown"
                            ),
                        },
                    )
                else:
                    self.log_step(
                        f"顧客メッセージ処理失敗: {customer_id}",
                        {"error": response.get("error", "不明なエラー")},
                    )
            else:
                self.log_step(
                    f"顧客エンゲージメント失敗: {customer_id}",
                    {"error": engagement.get("error", "不明なエラー")},
                )

        # 2. パーソナライズ推薦テスト
        self.log_step("パーソナライズ推薦テスト開始")
        for customer_id, _ in customer_messages:
            recommendations = (
                await customer_agent.generate_personalized_recommendations(customer_id)
            )
            self.log_step(
                f"パーソナライズ推薦生成: {customer_id}",
                {
                    "recommendation_count": len(recommendations),
                    "recommendations": recommendations[:3],  # 最初の3つだけ表示
                },
            )

        # 3. 満足度分析テスト
        self.log_step("満足度分析テスト開始")
        for customer_id, _ in customer_messages:
            satisfaction = await customer_agent.analyze_customer_satisfaction(
                customer_id
            )
            self.log_step(
                f"満足度分析完了: {customer_id}",
                {
                    "satisfaction_score": satisfaction.get("satisfaction_score", 0),
                    "total_conversations": satisfaction.get("total_conversations", 0),
                    "analysis": satisfaction.get("analysis", ""),
                },
            )


class ComprehensiveTestScenario(StoreSimulationScenario):
    """総合テストシナリオ"""

    def __init__(self):
        super().__init__(
            "総合テストシナリオ",
            "全エージェント機能を統合的にテスト。実際の店舗運営に近いシミュレーション。",
        )

    async def _execute_scenario(self):
        # 1. システム初期状態確認
        self.log_step("システム初期状態確認")
        health_status = await vending_agent.get_operation_status()
        self.log_step(
            "システム状態確認完了",
            {
                "mode": health_status.mode.value,
                "total_products": health_status.total_products,
                "system_health": f"{health_status.system_health:.1%}",
                "alert_count": len(health_status.alerts),
            },
        )

        # 2. 分析エージェントによるシステム分析
        self.log_step("システム分析実行開始")
        analysis_result = await analytics_agent.analyze_system_events()
        if analysis_result:
            analyses = analysis_result.get("analyses", {})
            insights = analysis_result.get("overall_insights", [])
            self.log_step(
                "システム分析完了",
                {
                    "analysis_types": len(analyses),
                    "insight_count": len(insights),
                    "has_llm_response": len(insights) > 0,
                },
            )

            # 分析結果の詳細を表示
            for i, (key, value) in enumerate(analyses.items(), 1):
                self.log_step(
                    f"分析詳細 {i}: {key}",
                    {
                        "has_insights": len(value.get("insights", [])) > 0,
                        "has_recommendations": len(value.get("recommendations", []))
                        > 0,
                    },
                )

        # 3. 調達計画の実行と検証
        self.log_step("調達計画実行開始")
        procurement_result = await procurement_agent.monitor_inventory_and_procure()
        if procurement_result.get("success"):
            actions = procurement_result.get("procurement_results", {}).get(
                "procurement_actions", []
            )
            self.log_step(
                "調達計画実行成功",
                {
                    "action_count": len(actions),
                    "message": procurement_result.get("message", ""),
                },
            )
        else:
            self.log_step(
                "調達計画実行失敗", {"error": procurement_result.get("error")}
            )

        # 4. 価格最適化の実行と検証
        self.log_step("価格最適化実行開始")
        price_result = await vending_agent.optimize_pricing()
        if price_result.get("success"):
            adjustments = price_result.get("optimization_results", {}).get(
                "price_adjustments", []
            )
            self.log_step("価格最適化成功", {"adjustment_count": len(adjustments)})

            # 最適化された価格でテスト販売
            for adj in adjustments[:2]:  # 最初の2つでテスト
                for product in SAMPLE_PRODUCTS:
                    if product.product_id == adj["product_id"]:
                        # 一時的に価格を変更してテスト販売
                        original_price = product.price
                        product.price = adj["optimal_price"]

                        test_payment = await payment_service.process_payment(
                            product.price, PaymentMethod.CARD
                        )

                        if test_payment.success:
                            self.log_step(
                                f"最適化価格テスト販売成功: {product.name}",
                                {
                                    "optimized_price": product.price,
                                    "original_price": original_price,
                                    "price_change": product.price - original_price,
                                },
                            )
                        else:
                            self.log_step(
                                f"最適化価格テスト販売失敗: {product.name}",
                                {"error": test_payment.error_message},
                            )

                        # 価格を元に戻す
                        product.price = original_price
                        break

        # 5. 顧客対応の包括的テスト
        self.log_step("顧客対応包括テスト開始")

        # 複数の顧客パターンをテスト
        customer_patterns = [
            {
                "id": "regular_customer",
                "messages": [
                    "おすすめの商品を教えて",
                    "コカ・コーラはありますか？",
                    "ありがとうございます",
                ],
            },
            {
                "id": "price_sensitive",
                "messages": [
                    "安い商品は何ですか？",
                    "セールはありますか？",
                    "予算内で買えるものを教えて",
                ],
            },
            {
                "id": "health_conscious",
                "messages": [
                    "ヘルシーな飲み物はありますか？",
                    "カロリー控えめな商品を教えて",
                    "おすすめの健康飲料は？",
                ],
            },
        ]

        for pattern in customer_patterns:
            customer_id = f"{pattern['id']}_{int(time.time())}"
            self.log_step(f"顧客パターン開始: {pattern['id']}")

            # 顧客エンゲージメント
            engagement = await customer_agent.engage_customer(customer_id, "VM001")
            if engagement["success"]:
                session_id = engagement["session_id"]

                for message in pattern["messages"]:
                    response = await customer_agent.handle_customer_message(
                        session_id, message
                    )
                    if response["success"]:
                        self.log_step(
                            f"顧客メッセージ処理成功: {pattern['id']} - {message[:15]}...",
                            {
                                "has_suggestions": len(response["suggested_products"])
                                > 0,
                                "intent": response.get("insights", {}).get(
                                    "intent", "unknown"
                                ),
                            },
                        )
                    else:
                        self.log_step(
                            f"顧客メッセージ処理失敗: {pattern['id']}",
                            {"error": response.get("error", "不明なエラー")},
                        )
            else:
                self.log_step(
                    f"顧客エンゲージメント失敗: {pattern['id']}",
                    {"error": engagement.get("error", "不明なエラー")},
                )


class ScenarioRunner:
    """シナリオ実行管理クラス"""

    def __init__(self):
        self.scenarios = {
            "normal": NormalOperationScenario(),
            "busy": BusyPeriodScenario(),
            "shortage": InventoryShortageScenario(),
            "price_optimization": PriceOptimizationScenario(),
            "customer_service": CustomerServiceScenario(),
            "comprehensive": ComprehensiveTestScenario(),
        }

    async def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """指定されたシナリオを実行"""
        if scenario_name not in self.scenarios:
            return {
                "error": f"シナリオ '{scenario_name}' が見つかりません",
                "available_scenarios": list(self.scenarios.keys()),
            }

        scenario = self.scenarios[scenario_name]
        return await scenario.run()

    async def run_all_scenarios(self) -> Dict[str, Any]:
        """すべてのシナリオを実行"""
        logger.info("=== 全シナリオ実行開始 ===")

        results = {}
        for name, scenario in self.scenarios.items():
            logger.info(f"シナリオ実行中: {name}")
            results[name] = await scenario.run()

        # 全体サマリー
        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results.values() if r["success"])
        total_duration = sum(r["duration_seconds"] for r in results.values())

        summary = {
            "total_scenarios": total_scenarios,
            "successful_scenarios": successful_scenarios,
            "failed_scenarios": total_scenarios - successful_scenarios,
            "total_duration_seconds": total_duration,
            "success_rate": f"{(successful_scenarios / total_scenarios) * 100:.1f}%",
            "scenario_results": results,
        }

        logger.info(f"=== 全シナリオ実行完了 ===")
        logger.info(
            f"成功率: {summary['success_rate']}, 総所要時間: {total_duration:.1f}秒"
        )

        return summary

    def get_scenario_info(self) -> Dict[str, Dict[str, str]]:
        """シナリオ情報を取得"""
        return {
            name: {
                "name": scenario.name,
                "description": scenario.description,
            }
            for name, scenario in self.scenarios.items()
        }


# グローバルインスタンス
scenario_runner = ScenarioRunner()


async def run_scenario_from_command_line(scenario_name: str = None):
    """コマンドラインからシナリオを実行"""
    if scenario_name:
        if scenario_name == "all":
            result = await scenario_runner.run_all_scenarios()
        else:
            result = await scenario_runner.run_scenario(scenario_name)

        # 結果をJSONで出力
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        # 利用可能なシナリオを表示
        scenarios = scenario_runner.get_scenario_info()
        print("利用可能なシナリオ:")
        for name, info in scenarios.items():
            print(f"  {name}: {info['name']}")
            print(f"    {info['description']}")


if __name__ == "__main__":
    import sys

    scenario_name = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_scenario_from_command_line(scenario_name))
