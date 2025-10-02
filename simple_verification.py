"""
Management Agent の簡単な検証スクリプト
"""

import asyncio
import sys


async def verify_imports():
    """インポートの確認"""
    print("=" * 60)
    print("1. インポートの確認")
    print("=" * 60)

    try:
        from src.agents.management_agent import SessionBasedManagementAgent

        print("✓ SessionBasedManagementAgent のインポート成功")
    except Exception as e:
        print(f"✗ SessionBasedManagementAgent のインポート失敗: {e}")
        return False

    try:
        from src.agents.recorder_agent import RecorderAgent

        print("✓ RecorderAgent のインポート成功")
    except Exception as e:
        print(f"✗ RecorderAgent のインポート失敗: {e}")
        return False

    return True


async def verify_basic_functionality():
    """基本機能の確認"""
    print("\n" + "=" * 60)
    print("2. 基本機能の確認")
    print("=" * 60)

    try:
        from src.agents.management_agent import SessionBasedManagementAgent

        # Agent初期化
        print("\nAgent初期化中...")
        agent = SessionBasedManagementAgent(provider="anthropic")
        print(f"✓ Agent初期化成功 ({len(agent.tools)}個のツール)")

        # ビジネスメトリクス取得
        print("\nビジネスメトリクス取得中...")
        metrics = agent.get_business_metrics()
        print(f"✓ メトリクス取得成功")
        print(f"  - 売上: ¥{metrics['sales']:,}")
        print(f"  - 利益率: {metrics['profit_margin']:.1%}")
        print(f"  - 在庫: {metrics['inventory_level']}")

        # セッション管理
        print("\nセッション管理のテスト...")
        session_id = await agent.start_management_session("test")
        print(f"✓ セッション開始: {session_id}")

        summary = await agent.end_management_session()
        print(f"✓ セッション終了: {summary['duration']}")

        return True

    except Exception as e:
        print(f"✗ エラー発生: {e}")
        import traceback

        traceback.print_exc()
        return False


async def verify_tools():
    """ツールの確認"""
    print("\n" + "=" * 60)
    print("3. ツールの確認")
    print("=" * 60)

    try:
        from src.agents.management_agent import SessionBasedManagementAgent

        agent = SessionBasedManagementAgent(provider="anthropic")

        print(f"\n登録されているツール数: {len(agent.tools)}")
        print("\nツール一覧:")
        for i, tool in enumerate(agent.tools, 1):
            print(f"  {i}. {tool.name}: {tool.description}")

        return True

    except Exception as e:
        print(f"✗ エラー発生: {e}")
        return False


async def main():
    """メイン実行"""
    print("\n" + "=" * 60)
    print("LangChain Management Agent 検証スクリプト")
    print("=" * 60 + "\n")

    results = []

    # インポート確認
    result = await verify_imports()
    results.append(("インポート", result))

    if not result:
        print("\n✗ インポートに失敗したため、検証を中止します")
        return

    # 基本機能確認
    result = await verify_basic_functionality()
    results.append(("基本機能", result))

    # ツール確認
    result = await verify_tools()
    results.append(("ツール", result))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)

    for name, success in results:
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{name}: {status}")

    all_success = all(r[1] for r in results)

    if all_success:
        print("\n" + "=" * 60)
        print("✓ 全ての検証が成功しました！")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("✗ 一部の検証が失敗しました")
        print("=" * 60 + "\n")

    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
