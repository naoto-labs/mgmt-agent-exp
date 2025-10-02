#!/usr/bin/env python3
"""
環境変数確認スクリプト

環境変数の設定状況を確認します。
"""

import os

from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()


def test_env_vars():
    """環境変数の確認"""
    print("🔧 環境変数確認:")

    # OpenAI関連
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE")
    openai_deployment = os.getenv("OPENAI_API_DEPLOYMENT")
    openai_version = os.getenv("OPENAI_API_VERSION")

    print(f"   OPENAI_API_KEY: {'設定あり' if openai_key else '未設定'}")
    if openai_key:
        print(f"     キー長: {len(openai_key)}文字")
        print(f"     先頭文字: {openai_key[:10]}...")

    print(f"   OPENAI_API_BASE: {'設定あり' if openai_base else '未設定'}")
    if openai_base:
        print(f"     エンドポイント: {openai_base}")

    print(f"   OPENAI_API_DEPLOYMENT: {'設定あり' if openai_deployment else '未設定'}")
    if openai_deployment:
        print(f"     デプロイメント: {openai_deployment}")

    print(f"   OPENAI_API_VERSION: {'設定あり' if openai_version else '未設定'}")
    if openai_version:
        print(f"     APIバージョン: {openai_version}")

    # Anthropic関連
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"   ANTHROPIC_API_KEY: {'設定あり' if anthropic_key else '未設定'}")
    if anthropic_key:
        print(f"     キー長: {len(anthropic_key)}文字")
        print(f"     先頭文字: {anthropic_key[:10]}...")

    # その他の設定
    encryption_key = os.getenv("ENCRYPTION_KEY")
    print(f"   ENCRYPTION_KEY: {'設定あり' if encryption_key else '未設定'}")

    print("\n📋 設定状況サマリー:")
    settings_status = []

    if openai_key and openai_base and openai_deployment:
        settings_status.append("✅ Azure OpenAI: 設定完了")
    else:
        settings_status.append("❌ Azure OpenAI: 設定不完全")

    if anthropic_key:
        settings_status.append("✅ Anthropic: 設定完了")
    else:
        settings_status.append("❌ Anthropic: 未設定")

    if encryption_key:
        settings_status.append("✅ 暗号化キー: 設定完了")
    else:
        settings_status.append("❌ 暗号化キー: 未設定")

    for status in settings_status:
        print(f"   {status}")


if __name__ == "__main__":
    test_env_vars()
