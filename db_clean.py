#!/usr/bin/env python3
"""
データベースクリーンアップスクリプト

既存のデータベースをバックアップし、新しいデータベースを作成します。
"""

import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.management_agent.evaluation_metrics import create_benchmarks_table


def backup_existing_database():
    """既存のデータベースをバックアップ"""
    db_path = Path("data/vending_bench.db")
    backup_dir = Path("data/backup")

    if not db_path.exists():
        print("⚠️  既存のデータベースファイルが見つかりません")
        return False

    # バックアップディレクトリの作成
    backup_dir.mkdir(exist_ok=True)

    # タイムスタンプ付きのバックアップファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"vending_bench_backup_{timestamp}.db"

    try:
        shutil.copy2(db_path, backup_path)
        print(f"✅ データベースをバックアップしました: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ バックアップエラー: {e}")
        return False


def create_new_database():
    """新しいデータベースを作成"""
    try:
        # データディレクトリの作成
        Path("data").mkdir(exist_ok=True)

        # 新しいデータベース接続
        conn = sqlite3.connect("data/vending_bench.db")

        # ベンチマークテーブル作成
        create_benchmarks_table(conn)

        # 作成されたテーブルを確認
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("✅ 新しいデータベースを作成しました")
        print(f"📋 作成されたテーブル: {[table[0] for table in tables]}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ データベース作成エラー: {e}")
        return False


def verify_database_structure():
    """データベース構造を確認"""
    db_path = Path("data/vending_bench.db")

    if not db_path.exists():
        print("❌ データベースファイルが見つかりません")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # テーブル一覧取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("\n📊 データベース構造確認:")
        print("=" * 50)

        for table in tables:
            table_name = table[0]
            print(f"\n🏗️  テーブル: {table_name}")

            # カラム情報取得
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            print("  カラム:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                pk_mark = " 🔑" if is_pk else ""
                print(f"    - {col_name} ({col_type}){pk_mark}")

        # データ件数確認
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"\n📈 {table_name}のデータ件数: {count}件")

        conn.close()
        print("\n✅ データベース構造確認完了")
        return True

    except Exception as e:
        print(f"❌ データベース構造確認エラー: {e}")
        return False


def main():
    """メイン実行関数"""
    print("🧹 データベースクリーンアップ開始")
    print("=" * 50)

    # 1. 既存DBのバックアップ
    print("\n1️⃣ 既存データベースのバックアップ...")
    backup_success = backup_existing_database()

    if not backup_success:
        print("⚠️  バックアップをスキップして新しいデータベースを作成します")

    # 2. 新しいDB作成
    print("\n2️⃣ 新しいデータベースの作成...")
    create_success = create_new_database()

    if not create_success:
        print("❌ データベースクリーンアップに失敗しました")
        return False

    # 3. 構造確認
    print("\n3️⃣ データベース構造の確認...")
    verify_success = verify_database_structure()

    if verify_success:
        print("\n🎉 データベースクリーンアップ完了！")
        print("📁 新しいデータベース: data/vending_bench.db")
        print("💾 バックアップ: data/backup/ フォルダ")
    else:
        print(
            "\n⚠️  データベースクリーンアップ完了しましたが、構造確認で問題が発生しました"
        )

    return verify_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
