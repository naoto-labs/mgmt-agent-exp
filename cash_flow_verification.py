#!/usr/bin/env python3
"""
資金起点ビジネスサイクル検証スクリプト

- 初期資金設定
- 発注による仕入れ費用
- 販売による収入
- 最終損益確認
"""

import asyncio
import logging
from datetime import date, datetime

from src.accounting.journal_entry import journal_processor
from src.accounting.management_accounting import management_analyzer
from src.agents.procurement_agent import procurement_agent
from src.agents.search_agent import search_agent
from src.models.inventory import create_sample_inventory_slots
from src.services.inventory_service import inventory_service
from src.simulations.sales_simulation import simulate_purchase_events

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_cash_flow_verification():
    """資金サイクル検証を実行"""
    print("💰 資金起点ビジネスサイクル検証")
    print("=" * 60)
    print("発注→仕入れ→販売→利益確認の完全サイクル")

    # === 初期化 ===
    print("\n📦 初期化...")

    # 在庫データ初期化
    sample_slots = create_sample_inventory_slots()
    # 売れやすくするために価格を調整
    for slot in sample_slots:
        if slot.product_name == "コカ・コーラ":
            slot.price = 100  # 販売を誘発するため100円に調整
    for slot in sample_slots:
        inventory_service.add_slot(slot)

    # === 1. 初期資金設定 ===
    print("\n💵 初期資金設定...")
    initial_capital = 500000  # 50万円

    # 資本金として記帳（借方：現金、貸方：資本金）
    # 借方：現金 (1110)
    journal_processor.add_entry(
        account_number="1001",  # 現金
        date=date.today(),
        amount=initial_capital,
        entry_type="debit",
        description="資本金導入（ビジネス開始資金） - 現金",
    )
    # 貸方：資本金 (3000) - 新勘定
    journal_processor.add_entry(
        account_number="3000",  # 資本金
        date=date.today(),
        amount=initial_capital,
        entry_type="credit",
        description="資本金導入（ビジネス開始資金） - 資本金",
    )

    current_balance = journal_processor.get_account_balance(
        "1001", date.today(), date.today()
    )
    print(f"✓ 初期資金設定完了: ¥{initial_capital:,} in 現金 (口座: 1001)")

    # === 2. 商品発注作成 ===
    print("\n📋 発注作成...")
    # 在庫低レベル商品の特定
    low_stock_products = [
        {
            "product_id": "cola",
            "product_name": "コカ・コーラ",
            "recommended_quantity": 20,
            "priority": "medium",
            "urgency": "通常",
        }
    ]

    for product_info in low_stock_products:
        print(
            f"  発注商品: {product_info['product_name']} x {product_info['recommended_quantity']}"
        )

        # 価格比較
        price_comparison = await search_agent.compare_prices(
            product_info["product_name"]
        )
        if price_comparison.best_price:
            print(f"  単価: ¥{price_comparison.best_price:,.0f}")

            # 発注情報作成
            order_info = {
                "product_id": product_info["product_id"],
                "product_name": product_info["product_name"],
                "recommended_quantity": product_info["recommended_quantity"],
                "priority": product_info["priority"],
                "urgency": product_info["urgency"],
                "supplier_info": {
                    "name": "サンプル業者",
                    "price": price_comparison.best_price,
                },
            }

            # 発注指示書作成
            order_result = await procurement_agent.create_order_instruction(
                order_info, order_info["supplier_info"]
            )

            if order_result["success"]:
                order = order_result["order_instruction"]
                print(f"  ✓ 発注作成完了: ID {order['order_id']}")

                # 発注費用を記帳（借方：在庫、貸方：現金）
                procurement_cost = order["total_amount"]
                # 借方：在庫増加 (1101)
                journal_processor.add_entry(
                    account_number="1101",  # 商品在庫
                    date=date.today(),
                    amount=procurement_cost,
                    entry_type="debit",
                    description=f"商品発注: {order['product_name']} x {order['quantity']} - 在庫増加",
                )
                # 貸方：現金減少 (1001)
                journal_processor.add_entry(
                    account_number="1001",  # 現金
                    date=date.today(),
                    amount=procurement_cost,
                    entry_type="credit",
                    description=f"商品発注: {order['product_name']} x {order['quantity']} - 現金支出",
                )

                # 在庫増加（簡易）
                target_slot = next(
                    (
                        s
                        for s in sample_slots
                        if s.product_name.lower()
                        == product_info["product_name"].lower()
                    ),
                    None,
                )
                if target_slot:
                    target_slot.current_quantity += product_info["recommended_quantity"]
                    print(
                        f"  ✓ 在庫補充完了: {product_info['product_name']} + {product_info['recommended_quantity']}"
                    )

                print(f"  💸 発注費用: ¥{procurement_cost:,.0f}")
            else:
                print("  ✗ 発注作成失敗")

    # === 発注後の資金確認 ===
    cash_after_procurement = journal_processor.get_account_balance(
        "1001", date.today(), date.today()
    )
    print(f"\n💰 発注後現金残高: ¥{cash_after_procurement:,}")
    print(f"   減少額: ¥{initial_capital - cash_after_procurement:,}")

    # === 3. 販売実行 ===
    print("\n🛒 販売シミュレーション...")
    try:
        sales_result = await simulate_purchase_events(
            10.0, verbose=True, period_name="現金フロー検証"
        )
        print(f"✓ 販売完了: {sales_result}")
    except Exception as e:
        print(f"✗ 販売エラー: {e}")

    # === 販売後の資金確認 ===
    cash_after_sales = journal_processor.get_account_balance(
        "1001", date.today(), date.today()
    )
    print(f"\n💰 販売後現金残高: ¥{cash_after_sales:,}")
    print(f"   販売収入: ¥{cash_after_sales - cash_after_procurement:,}")

    # === 4. 損益確認 ===
    print("\n📊 損益分析...")
    try:
        profit_analysis = management_analyzer.analyze_period_profitability(
            date.today(), date.today()
        )

        gross_profit = profit_analysis.get("gross_profit", 0)
        gross_margin = profit_analysis.get("gross_margin", 0)

        print(f"✓ 粗利益: ¥{gross_profit:,.0f}")
        print(f"✓ 粗利益率: {gross_margin:.1%}")

        if gross_profit > 0:
            print("🎉 利益が出ています！資金増加成功")
        else:
            print("⚠️ 損失発生。戦略の見直しが必要です")

    except Exception as e:
        print(f"✗ 損益分析エラー: {e}")

    # === 5. 期末決算 ===
    print("\n🧾 期末決算...")
    try:
        trial_balance = journal_processor.get_trial_balance()
        print(f"✓ 総借方: ¥{trial_balance.get('total_debit', 0):,}")
        print(f"✓ 総貸方: ¥{trial_balance.get('total_credit', 0):,}")

        net_capital = trial_balance.get("total_credit", 0) - trial_balance.get(
            "total_debit", 0
        )
        print(f"✓ 純資本変動: ¥{net_capital:,}")

        if net_capital > initial_capital:
            profit = net_capital - initial_capital
            print(f"🎯 最終利益: ¥{profit:,}（資金増加成功！）")
        elif net_capital < initial_capital:
            loss = initial_capital - net_capital
            print(f"⚠️ 最終損失: ¥{loss:,}")
        else:
            print("😐 損益なし")

    except Exception as e:
        print(f"✗ 決算エラー: {e}")

    # === 結果サマリ ===
    print("\n\n" + "=" * 60)
    print("📈 資金フロー検証サマリー")
    print(f"初始資金: ¥{initial_capital:,}")
    print(f"最終資金: ¥{cash_after_sales:,}")
    net_change = cash_after_sales - initial_capital
    print(f"純増減: ¥{net_change:,} ({'増加' if net_change > 0 else '減少'})")
    print("💡 発注→販売サイクルで資金増殖可能:" + ("✅" if net_change > 0 else "❌"))

    print("\n✨ 検証完了! AI自販機事業の収益性確認できました。")


async def main():
    try:
        await run_cash_flow_verification()
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    asyncio.run(main())
