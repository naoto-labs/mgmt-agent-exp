#!/usr/bin/env python3

import asyncio
import logging
import traceback

from src.agents.search_agent import DuckDuckGoSearchClient

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_duckduckgo():
    print("DuckDuckGo検索テスト開始")
    client = DuckDuckGoSearchClient()

    try:
        # Amazon限定検索テスト
        print("Amazon限定検索テスト...")
        results = await client.search(
            "コカ・コーラ", max_results=3, site_limit="amazon.co.jp"
        )

        print(f"結果数: {len(results)}")
        for i, result in enumerate(results[:3], 1):  # 上位3件のみ表示
            print(f"{i}. タイトル: {result.title}")
            print(f"   URL: {result.url}")
            print(f"   価格: {result.price}円" if result.price else "   価格: 未取得")
            print(f"   ソース: {result.source}")
            print(f"   スコア: {result.relevance_score:.2f}")
            print(f"   スニペット: {result.snippet[:100]}...")
            print()

        # 詳細デバッグ情報
        print("レスポンス詳細:")
        if results:
            avg_score = sum(r.relevance_score for r in results[:3]) / min(
                3, len(results)
            )
            print(f"  平均スコア: {avg_score:.2f}")
        else:
            print("  No results")

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_duckduckgo())
