import os
import sqlite3


def test_database_setup():
    # データベースファイルが存在するか確認
    db_path = "data/vending_bench.db"
    print(f"Database path: {db_path}")
    print(f"Database exists before: {os.path.exists(db_path)}")

    # テーブル作成
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # benchmarksテーブル作成 (VendingBench spec準拠DDL)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS benchmarks (
        run_id INTEGER,
        step INTEGER,
        profit_actual REAL,
        stockout_count INTEGER,
        total_demand INTEGER,
        pricing_accuracy REAL,
        action_correctness REAL,
        customer_satisfaction REAL
    );
    """

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

    print("Benchmarks table created/verified successfully")
    print(f"Database exists after: {os.path.exists(db_path)}")

    # テーブル内容確認
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM sqlite_master WHERE type = "table"')
    tables = cursor.fetchall()
    print(f"Tables in database: {tables}")

    # テーブル構造確認
    cursor.execute("PRAGMA table_info(benchmarks)")
    columns = cursor.fetchall()
    print(f"Columns in benchmarks table:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

    conn.close()


if __name__ == "__main__":
    test_database_setup()
