#!/usr/bin/env python3
"""
STORAGEも含めた包括的な在庫状況可視化テスト

在庫切れ問題の根本原因を特定するためのテスト
- VENDING_MACHINE在庫状況
- STORAGE在庫状況
- 補充処理の動作確認
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.agents.management_agent.agent import management_agent
from src.agents.management_agent.models import ManagementState
from src.application.services.inventory_service import inventory_service


async def comprehensive_inventory_check():
    """STORAGEも含めた包括的な在庫状況チェック"""

    print("🔍 STORAGE + VENDING_MACHINE 総合在庫状況分析")
    print("=" * 60)

    # 全在庫データを取得
    all_inventory = inventory_service.get_inventory_by_location()
    vending_inventory = all_inventory.get("vending_machine", [])
    storage_inventory = all_inventory.get("storage", [])

    print("📦 在庫状況全体:")
    print(f"  VENDING MACHINE: {len(vending_inventory)}スロット")
    print(f"  STORAGE: {len(storage_inventory)}スロット")
    print(f"  総スロット数: {len(vending_inventory) + len(storage_inventory)}")

    # 商品別集計
    product_summary = {}

    for slot in vending_inventory + storage_inventory:
        product_name = slot.product_name
        location = "VENDING" if slot in vending_inventory else "STORAGE"

        if product_name not in product_summary:
            product_summary[product_name] = {
                "product_id": slot.product_id,
                "vending_qty": 0,
                "storage_qty": 0,
                "total_qty": 0,
                "slots_vending": 0,
                "slots_storage": 0,
            }

        if location == "VENDING":
            product_summary[product_name]["vending_qty"] += slot.current_quantity
            product_summary[product_name]["slots_vending"] += 1
        else:  # STORAGE
            product_summary[product_name]["storage_qty"] += slot.current_quantity
            product_summary[product_name]["slots_storage"] += 1

        product_summary[product_name]["total_qty"] = (
            product_summary[product_name]["vending_qty"]
            + product_summary[product_name]["storage_qty"]
        )

    print("\n📋 商品別詳細在庫:")
    print("-" * 80)
    print(
        f"{'商品名':<15} {'VENDING':>8} {'STORAGE':>8} {'合計':>8} {'スロット':<10} {'ステータス'}"
    )
    print("-" * 80)

    critical_items = []
    low_stock_items = []
    storage_only_items = []

    for product_name, data in product_summary.items():
        vending_qty = data["vending_qty"]
        storage_qty = data["storage_qty"]
        total_qty = data["total_qty"]

        # ステータス判定
        if vending_qty == 0 and storage_qty == 0:
            status = "❌ 完全欠品"
            critical_items.append(product_name)
        elif vending_qty == 0 and storage_qty > 0:
            status = "⚠️  STORAGEのみ"
            storage_only_items.append(product_name)
        elif vending_qty < 15:  # 在庫が少ない閾値
            status = "🟡 在庫不足"
            low_stock_items.append(product_name)
        else:
            status = "✅ 正常"

        vending_slots = data["slots_vending"]
        storage_slots = data["slots_storage"]
        slot_info = f"V:{vending_slots}/S:{storage_slots}"

        print(
            f"{product_name:<15} {vending_qty:>8} {storage_qty:>8} {total_qty:>8} {slot_info:<10} {status}"
        )

    print("\n🚨 問題状況サマリー:")
    print(f"  - 完全欠品商品: {len(critical_items)}個")
    print(f"  - 在庫不足商品: {len(low_stock_items)}個")
    print(f"  - STORAGEのみ商品: {len(storage_only_items)}個")
    print(f"  - 全商品数: {len(product_summary)}個")

    if critical_items:
        print(f"  ❌ 完全欠品: {', '.join(critical_items)}")
    if storage_only_items:
        print(f"  ⚠️  STORAGE未払出: {', '.join(storage_only_items)}")
    if low_stock_items:
        print(f"  🟡 在庫不足: {', '.join(low_stock_items)}")

    # 補充処理分析
    print("\n🔄 補充処理分析:")

    total_vending_stock = sum(data["vending_qty"] for data in product_summary.values())
    total_storage_stock = sum(data["storage_qty"] for data in product_summary.values())

    print(f"  自販機総在庫量: {total_vending_stock}")
    print(f"  STORAGE総在庫量: {total_storage_stock}")
    if total_vending_stock > 0:
        ratio = (total_storage_stock / total_vending_stock) * 100
        print(f"  STORAGE/VENDING比率: {ratio:.1f}% (理想: 100-200%)")
    else:
        print("  STORAGE/VENDING比率: 計算不能 (自販機在庫ゼロ)")

    # 補充必要性の評価
    restock_needed = []
    for product_name, data in product_summary.items():
        if data["vending_qty"] < 20 and data["storage_qty"] >= 10:
            needed = 20 - data["vending_qty"]
            available = min(needed, data["storage_qty"])
            restock_needed.append(
                f"{product_name}(需要:{needed}, 移動可能:{available})"
            )

    if restock_needed:
        print(f"\n📦 補充推奨項目 ({len(restock_needed)}件):")
        for item in restock_needed:
            print(f"  - {item}")
    else:
        print("\n📦 補充推奨項目: なし (STORAGEが空または補充不要)")

    print("\n✅ STORAGE包括的可視化完了")

    return {
        "critical_items": critical_items,
        "storage_only_items": storage_only_items,
        "low_stock_items": low_stock_items,
        "product_summary": product_summary,
    }


async def restock_process_test():
    """補充プロセスの動作テスト"""

    print("\n🔧 補充プロセステスト開始")
    print("=" * 40)

    # 現在の在庫状況を取得
    initial_result = await comprehensive_inventory_check()

    # 補充が必要な商品を特定
    items_needing_restock = []
    for product_name, data in initial_result["product_summary"].items():
        if data["vending_qty"] < 20 and data["storage_qty"] >= 10:
            items_needing_restock.append(product_name)

    if not items_needing_restock:
        print("⏭️ 補充が必要な商品なし - テストスキップ")
        return

    print(f"📦 補充対象商品: {', '.join(items_needing_restock)}")

    # restock ノードを直接テスト
    try:
        from src.agents.management_agent.agent import NodeBasedManagementAgent

        agent = NodeBasedManagementAgent()

        # 初期状態作成
        state = ManagementState(
            session_id="inventory_test_001",
            inventory_analysis={"low_stock_items": items_needing_restock},
        )

        print("🚀 restock ノード実行中...")

        # restock ノード実行
        updated_state = await agent.restock_node(state)

        print("✅ restock ノード実行完了")

        # 補充後の結果確認
        print("\n--- 補充後確認 ---")

    except Exception as e:
        print(f"❌ 補充プロセステスト失敗: {e}")
        return

    # 補充後の在庫変化確認
    await comprehensive_inventory_check()

    print("\n🎯 根本原因分析:")
    print("現在の問題:")
    if initial_result["storage_only_items"]:
        print("- STORAGEに在庫があるのに自販機にありません")
        print("- 補充処理がSTORAGEから自販機への移動を完了していない可能性")
    else:
        print("- STORAGE自体に在庫がありません")
        print("- 調達処理がSTORAGEへ商品を届けてない可能性")


async def main():
    """メイン実行関数"""
    logging.basicConfig(level=logging.ERROR)  # エラーログのみ表示

    print("🚀 在庫可視化テスト開始")
    print("=======================================")

    try:
        # 包括的な在庫チェック
        inventory_status = await comprehensive_inventory_check()

        # 連続在庫切れの場合、補充テストを実行
        critical_count = len(inventory_status["critical_items"])
        storage_only_count = len(inventory_status["storage_only_items"])

        if critical_count > 0 or storage_only_count > 0:
            print("\n💡 在庫問題が検出されました - 補充プロセスをテストします")
            await restock_process_test()
        else:
            print("\n✅ 在庫状況は正常です")

    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()

    print("\n=======================================")
    print("🏁 在庫可視化テスト完了")


if __name__ == "__main__":
    asyncio.run(main())
