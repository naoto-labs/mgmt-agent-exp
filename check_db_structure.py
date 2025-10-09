#!/usr/bin/env python3
"""
データベース構造確認スクリプト
"""

import sqlite3

import pandas as pd


def check_db_structure():
    """データベース構造を確認"""
    try:
        conn = sqlite3.connect("data/vending_bench.db")
        cursor = conn.cursor()

        # テーブル一覧取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("=== データベーステーブル一覧 ===")
        for table in tables:
            table_name = table[0]
            print(f"- {table_name}")

            # 各テーブルの構造確認
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            print(f"  カラム ({len(columns)}個):")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                print(f"    - {col_name}: {col_type} {'(PK)' if is_pk else ''}")

            # データ件数確認
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  レコード数: {count}")

            # サンプルデータ表示（最初の3件）
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                rows = cursor.fetchall()
                if rows:
                    print("  サンプルデータ:")
                    for i, row in enumerate(rows):
                        print(f"    行{i + 1}: {row}")

        conn.close()

    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    check_db_structure()
