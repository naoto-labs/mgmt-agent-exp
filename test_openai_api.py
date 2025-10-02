#!/usr/bin/env python3
"""
OpenAI APIテストスクリプト

Azure OpenAIの設定が正しく動作するかテストします。
"""

import asyncio
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai.model_manager import model_manager


async def test_openai_api():
    """OpenAI APIのテスト"""
    print("🤖 OpenAI APIテスト開始")

    try:
        # モデル統計を取得
        stats = model_manager.get_model_stats()
        print(f"   ✅ プライマリモデル: {stats.get('primary_model')}")
        print(f"   ✅ 利用可能モデル数: {len(stats.get('models', {}))}")

        # 各モデルの詳細情報を表示
        models = stats.get("models", {})
        for model_name, model_info in models.items():
            print(f"   📊 {model_name}: {model_info.get('status', '不明')}")

        # モデルヘルスチェック
        print("\n🔍 モデルヘルスチェックを実行中...")
        health = await model_manager.check_all_models_health()
        print("   ✅ モデルヘルスチェック完了")

        for model_name, is_healthy in health.items():
            status = "✅ 正常" if is_healthy else "❌ 異常"
            print(f"      - {model_name}: {status}")

        # シンプルな応答生成テスト
        print("\n💬 AI応答生成テストを実行中...")
        from src.ai.model_manager import AIMessage

        messages = [
            AIMessage(
                role="user",
                content="こんにちは、テストです。あなたの役割を教えてください。",
            )
        ]

        response = await model_manager.generate_response(messages, max_tokens=100)

        if response.success:
            print(f"   ✅ AI応答生成成功")
            print(f"   🤖 使用モデル: {response.model_used}")
            print(f"   ⏱️ 応答時間: {response.response_time:.2f}秒")
            print(f"   📝 トークン使用量: {response.tokens_used}")
            print(f"   💬 応答内容: {response.content}")
        else:
            print(f"   ❌ AI応答生成失敗: {response.error_message}")
            print(f"   🤖 使用モデル: {response.model_used}")

        # Azure OpenAIがプライマリの場合の詳細テスト
        if stats.get("primary_model") == "azure_openai":
            print("\n🔧 Azure OpenAI詳細テストを実行中...")

            # より複雑なプロンプトでテスト
            complex_messages = [
                AIMessage(role="system", content="あなたは親切なAIアシスタントです。"),
                AIMessage(
                    role="user",
                    content="自動販売機の運営についてアドバイスをお願いします。以下のポイントを考慮してください：1)在庫管理、2)顧客満足度、3)コスト効率。",
                ),
            ]

            complex_response = await model_manager.generate_response(
                complex_messages, max_tokens=200
            )

            if complex_response.success:
                print("   ✅ 複雑なクエリ応答成功")
                print(f"   💬 応答内容: {complex_response.content[:100]}...")
            else:
                print(f"   ❌ 複雑なクエリ応答失敗: {complex_response.error_message}")

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        import traceback

        traceback.print_exc()

    print("\n🎉 OpenAI APIテスト完了")


if __name__ == "__main__":
    asyncio.run(test_openai_api())
