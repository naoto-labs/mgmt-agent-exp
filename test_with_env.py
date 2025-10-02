#!/usr/bin/env python3
"""
環境変数を読み込んでテストするスクリプト

.envファイルから環境変数を読み込んでからテストを実行します。
"""

import os
import sys

from dotenv import load_dotenv

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .envファイルから環境変数を読み込み
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"✅ .envファイルを読み込みました: {env_file}")
else:
    print(f"❌ .envファイルが見つかりません: {env_file}")


def test_with_env():
    """環境変数を読み込んでテスト"""
    print("\n🔧 環境変数確認（.env読み込み後）:")

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

    # セキュリティ設定テスト
    print("\n🔐 セキュリティ設定テスト:")
    try:
        from src.config.security import secure_config

        print(f"   Azure OpenAI Key: {bool(secure_config.azure_openai_key)}")
        print(f"   Azure Endpoint: {bool(secure_config.azure_endpoint)}")
        print(f"   Azure Deployment: {secure_config.azure_deployment}")
        print(f"   OpenAI Key: {bool(secure_config.openai_api_key)}")
        print(f"   Anthropic Key: {bool(secure_config.anthropic_api_key)}")
    except Exception as e:
        print(f"   ❌ セキュリティ設定エラー: {e}")


if __name__ == "__main__":
    test_with_env()
