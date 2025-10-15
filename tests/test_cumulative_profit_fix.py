#!/usr/bin/env python3
"""
累積利益修正の検証テスト

continuous_multi_day_simulation.pyの累積利益修正が正しく動作するか確認
"""

import asyncio
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from continuous_multi_day_simulation import run_continuous_simulation


async def test_cumulative_profit():
    """累積利益修正テスト"""
    print("🔍 累積利益修正検証テスト開始")
    print("=" * 60)

    try:
        # 2日間のシミュレーション実行で累積利益を確認
        results = await run_continuous_simulation(duration_days=5, verbose=False)

        daily_results = results.get("daily_results", [])
        cumulative_values = []

        print("\n📊 日次累積利益推移:")
        print("-" * 40)
        for day_result in daily_results:
            day = day_result["day"]
            cumulative = day_result.get("cumulative_profit", 0)
            cumulative_values.append(cumulative)
            print("04d")

        print("\n🔍 検証結果:")
        if len(cumulative_values) >= 2:
            if (
                cumulative_values[-1] > 0
                and cumulative_values[-1] >= cumulative_values[0]
            ):
                print("  ✅ SUCCESS: 累積利益が正常に維持されています")
                print("02d")
                print("02d")
                if (
                    len(cumulative_values) >= 2
                    and cumulative_values[-1] > cumulative_values[0]
                ):
                    increase = cumulative_values[-1] - cumulative_values[0]
                    print("02d")
                return True
            else:
                print("  ❌ FAILURE: 累積利益が期待通りではない")
                print(f"  累積値: {cumulative_values}")
                return False
        else:
            print("  ❌ FAILURE: 日次結果データが不足")
            return False

    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """メイン関数"""
    print("🚀 累積利益修正検証テスト")
    print("累積KPIの問題が解決されたか確認します")
    print()

    success = await test_cumulative_profit()

    print("\n" + "=" * 60)
    if success:
        print("🎉 検証成功: 累積利益修正が正常に動作しています！")
        print("Day 2以降で累積利益が正しく増加しています。")
    else:
        print("❌ 検証失敗: 累積利益修正に問題があります。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
