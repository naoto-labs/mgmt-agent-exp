from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.domain.accounting.journal_entry import (
    AccountCode,
    AccountingEntry,
    ChartOfAccounts,
    JournalEntry,
    JournalEntryProcessor,
    journal_processor,
)
from src.domain.accounting.management_accounting import (
    EfficiencyRating,
    InventoryEfficiency,
    ManagementAccountingAnalyzer,
    ProductProfitability,
    ProfitabilityRating,
    management_analyzer,
)
from src.domain.models.product import Product, ProductCategory
from src.domain.models.transaction import PaymentMethod, Transaction


class TestChartOfAccounts:
    """勘定科目表のテスト"""

    @pytest.fixture
    def chart_of_accounts(self):
        """勘定科目表のフィクスチャ"""
        return ChartOfAccounts()

    def test_get_account_name(self, chart_of_accounts):
        """勘定科目名取得テスト"""
        assert chart_of_accounts.get_account_name(AccountCode.CASH) == "現金"
        assert chart_of_accounts.get_account_name(AccountCode.INVENTORY) == "商品"
        assert chart_of_accounts.get_account_name("9999") == "不明な科目"

    def test_get_all_accounts(self, chart_of_accounts):
        """全勘定科目取得テスト"""
        accounts = chart_of_accounts.get_all_accounts()

        assert AccountCode.CASH in accounts
        assert AccountCode.SALES_REVENUE in accounts
        assert len(accounts) >= 6  # 最低6科目あるはず


class TestAccountingEntry:
    """会計エントリのテスト"""

    def test_accounting_entry_creation(self):
        """会計エントリ作成テスト"""
        entry = AccountingEntry(
            account_code=AccountCode.CASH,
            account_name="現金",
            debit_amount=1000.0,
            credit_amount=0.0,
        )

        assert entry.is_debit() is True
        assert entry.is_credit() is False
        assert entry.get_amount() == 1000.0

    def test_credit_entry(self):
        """貸方エントリテスト"""
        entry = AccountingEntry(
            account_code=AccountCode.SALES_REVENUE,
            account_name="売上高",
            debit_amount=0.0,
            credit_amount=1500.0,
        )

        assert entry.is_debit() is False
        assert entry.is_credit() is True
        assert entry.get_amount() == 1500.0


class TestJournalEntry:
    """仕訳エントリのテスト"""

    def test_journal_entry_creation(self):
        """仕訳エントリ作成テスト"""
        entries = [
            AccountingEntry(AccountCode.CASH, "現金", debit_amount=1000.0),
            AccountingEntry(AccountCode.SALES_REVENUE, "売上高", credit_amount=1000.0),
        ]

        journal = JournalEntry(
            entry_id="JE001", date=date.today(), description="商品販売", entries=entries
        )

        assert journal.entry_id == "JE001"
        assert journal.is_balanced() is True
        assert journal.get_total_amount() == 1000.0

    def test_unbalanced_journal_entry(self):
        """バランスの取れていない仕訳テスト"""
        entries = [
            AccountingEntry(AccountCode.CASH, "現金", debit_amount=1000.0),
            AccountingEntry(
                AccountCode.SALES_REVENUE, "売上高", credit_amount=800.0
            ),  # バランスが取れていない
        ]

        journal = JournalEntry(
            entry_id="JE002",
            date=date.today(),
            description="不正な仕訳",
            entries=entries,
        )

        assert journal.is_balanced() is False


class TestJournalEntryProcessor:
    """仕訳処理プロセッサのテスト"""

    @pytest.fixture
    def processor(self):
        """仕訳プロセッサのフィクスチャ"""
        return JournalEntryProcessor()

    def test_processor_initialization(self, processor):
        """プロセッサ初期化テスト"""
        assert processor.chart_of_accounts is not None
        assert processor.journal_entries == []
        assert processor.entry_counter == 0

    def test_record_sale_success(self, processor):
        """売上仕訳記録成功テスト"""
        # 取引オブジェクトを作成
        transaction = Transaction(
            transaction_id="TXN001",
            machine_id="VM001",
            items=[],
            subtotal=1000.0,
            total_amount=1000.0,
        )

        journal_entry = processor.record_sale(transaction)

        assert journal_entry.entry_id.startswith("JE")
        assert journal_entry.description == "商品売上 - 取引ID: TXN001"
        assert journal_entry.is_balanced() is True
        assert len(processor.journal_entries) == 1

        # エントリの内容を確認
        debit_entry = next(e for e in journal_entry.entries if e.is_debit())
        credit_entry = next(e for e in journal_entry.entries if e.is_credit())

        assert debit_entry.account_code == AccountCode.CASH
        assert credit_entry.account_code == AccountCode.SALES_REVENUE

    def test_record_sale_zero_amount(self, processor):
        """ゼロ金額売上テスト"""
        transaction = Transaction(
            transaction_id="TXN002",
            machine_id="VM001",
            items=[],
            subtotal=0.0,
            total_amount=0.0,
        )

        with pytest.raises(ValueError, match="売上金額は0より大きくなければなりません"):
            processor.record_sale(transaction)

    def test_record_purchase_success(self, processor):
        """仕入仕訳記録成功テスト"""
        product = Product(
            name="テスト商品",
            description="テスト用商品",
            category=ProductCategory.DRINK,
            price=200.0,
            cost=120.0,
        )

        journal_entries = processor.record_purchase(
            product, 10, "テストサプライヤー", "PO001"
        )

        assert len(journal_entries) == 2  # 仕入時と在庫計上の2つ
        assert all(entry.is_balanced() for entry in journal_entries)
        assert len(processor.journal_entries) == 2

    def test_get_journal_entries_with_date_filter(self, processor):
        """日付フィルタ付き仕訳取得テスト"""
        # 複数の仕訳を記録
        for i in range(3):
            transaction = Transaction(
                transaction_id=f"TXN{i}",
                machine_id="VM001",
                items=[],
                subtotal=1000.0,
                total_amount=1000.0,
            )
            processor.record_sale(transaction)

        # 日付範囲でフィルタリング
        start_date = date.today()
        end_date = date.today()

        entries = processor.get_journal_entries(start_date, end_date)

        assert len(entries) == 3

    def test_get_account_balance(self, processor):
        """勘定科目残高取得テスト"""
        # 売上取引を記録
        transaction = Transaction(
            transaction_id="TXN001",
            machine_id="VM001",
            items=[],
            subtotal=1000.0,
            total_amount=1000.0,
        )
        processor.record_sale(transaction)

        # 現金の残高を確認
        cash_balance = processor.get_account_balance(AccountCode.CASH)
        sales_balance = processor.get_account_balance(AccountCode.SALES_REVENUE)

        assert cash_balance == 1000.0  # 借方残高
        assert sales_balance == -1000.0  # 貸方残高（負数で表示）

    def test_get_trial_balance(self, processor):
        """試算表取得テスト"""
        # 複数の取引を記録
        for i in range(2):
            transaction = Transaction(
                transaction_id=f"TXN{i}",
                machine_id="VM001",
                items=[],
                subtotal=1000.0,
                total_amount=1000.0,
            )
            processor.record_sale(transaction)

        trial_balance = processor.get_trial_balance()

        assert trial_balance["total_debit"] == trial_balance["total_credit"]
        assert len(trial_balance["accounts"]) >= 2  # 少なくとも現金と売上高


class TestManagementAccountingAnalyzer:
    """管理会計分析のテスト"""

    @pytest.fixture
    def analyzer(self):
        """管理会計分析のフィクスチャ"""
        return ManagementAccountingAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """分析ツール初期化テスト"""
        assert analyzer.journal_processor is not None

    def test_analyze_product_profitability(self, analyzer):
        """商品別収益性分析テスト"""
        # 仕訳を記録してデータを準備
        product = Product(
            name="テスト商品",
            description="テスト用",
            category=ProductCategory.DRINK,
            price=200.0,
            cost=120.0,
        )

        transaction = Transaction(
            transaction_id="TXN001",
            machine_id="VM001",
            items=[],
            subtotal=200.0,
            total_amount=200.0,
        )

        analyzer.journal_processor.record_sale(transaction)

        # 分析を実行
        profitability = analyzer.analyze_product_profitability(
            "product_1", period_days=30
        )

        assert profitability.product_id == "product_1"
        assert profitability.profitability_rating in [
            ProfitabilityRating.EXCELLENT,
            ProfitabilityRating.GOOD,
            ProfitabilityRating.AVERAGE,
        ]
        assert profitability.gross_margin >= 0

    def test_calculate_inventory_turnover(self, analyzer):
        """在庫回転率計算テスト"""
        efficiency = analyzer.calculate_inventory_turnover("product_1")

        assert efficiency.product_id == "product_1"
        assert efficiency.efficiency_rating in [
            EfficiencyRating.EXCELLENT,
            EfficiencyRating.GOOD,
            EfficiencyRating.AVERAGE,
            EfficiencyRating.POOR,
            EfficiencyRating.INEFFICIENT,
        ]
        assert efficiency.inventory_turnover_ratio >= 0
        assert efficiency.inventory_turnover_days >= 0

    def test_analyze_period_profitability(self, analyzer):
        """期間別収益性分析テスト"""
        # テストデータを準備
        transaction = Transaction(
            transaction_id="TXN001",
            machine_id="VM001",
            items=[],
            subtotal=1000.0,
            total_amount=1000.0,
        )
        analyzer.journal_processor.record_sale(transaction)

        start_date = date.today()
        end_date = date.today()

        analysis = analyzer.analyze_period_profitability(start_date, end_date)

        assert "period" in analysis
        assert "sales_revenue" in analysis
        assert "gross_margin" in analysis
        assert "analysis" in analysis

    def test_rate_profitability(self, analyzer):
        """収益性評価テスト"""
        assert analyzer._rate_profitability(0.35) == ProfitabilityRating.EXCELLENT
        assert analyzer._rate_profitability(0.25) == ProfitabilityRating.GOOD
        assert analyzer._rate_profitability(0.15) == ProfitabilityRating.AVERAGE
        assert analyzer._rate_profitability(0.07) == ProfitabilityRating.POOR
        assert analyzer._rate_profitability(0.02) == ProfitabilityRating.UNPROFITABLE

    def test_rate_inventory_efficiency(self, analyzer):
        """在庫効率性評価テスト"""
        assert analyzer._rate_inventory_efficiency(15) == EfficiencyRating.EXCELLENT
        assert analyzer._rate_inventory_efficiency(10) == EfficiencyRating.GOOD
        assert analyzer._rate_inventory_efficiency(6) == EfficiencyRating.AVERAGE
        assert analyzer._rate_inventory_efficiency(3) == EfficiencyRating.POOR
        assert analyzer._rate_inventory_efficiency(1) == EfficiencyRating.INEFFICIENT

    def test_generate_profitability_report(self, analyzer):
        """収益性レポート生成テスト"""
        report = analyzer.generate_profitability_report(
            ["product_1", "product_2"], period_days=30
        )

        assert "period" in report
        assert "products" in report
        assert "summary" in report
        assert len(report["products"]) == 2

    def test_analyze_trend(self, analyzer):
        """トレンド分析テスト"""
        trend = analyzer.analyze_trend(periods=3)

        assert "periods" in trend
        assert "trends" in trend
        assert "analysis" in trend
        assert len(trend["periods"]) == 3

    def test_calculate_trend_direction(self, analyzer):
        """トレンド方向計算テスト"""
        # 増加傾向
        increasing_values = [100, 110, 120, 130]
        assert analyzer._calculate_trend_direction(increasing_values) == "increasing"

        # 減少傾向
        decreasing_values = [130, 120, 110, 100]
        assert analyzer._calculate_trend_direction(decreasing_values) == "decreasing"

        # 安定傾向
        stable_values = [100, 102, 98, 101]
        assert analyzer._calculate_trend_direction(stable_values) == "stable"

        # データ不足
        insufficient_values = [100]
        assert (
            analyzer._calculate_trend_direction(insufficient_values)
            == "insufficient_data"
        )

    def test_generate_management_dashboard_data(self, analyzer):
        """管理会計ダッシュボードデータ生成テスト"""
        dashboard = analyzer.generate_management_dashboard_data()

        assert "current_period" in dashboard
        assert "product_profitability" in dashboard
        assert "trend_analysis" in dashboard
        assert "inventory_efficiency" in dashboard
        assert "kpi_summary" in dashboard


# 統合テスト
def test_journal_processor_integration():
    """仕訳プロセッサの統合テスト"""
    processor = JournalEntryProcessor()

    # 売上取引を記録
    transaction = Transaction(
        transaction_id="TXN001",
        machine_id="VM001",
        items=[],
        subtotal=1500.0,
        total_amount=1500.0,
    )

    sale_entry = processor.record_sale(transaction)

    # 仕入取引を記録
    product = Product(
        name="テスト商品",
        description="テスト用",
        category=ProductCategory.DRINK,
        price=200.0,
        cost=120.0,
    )

    purchase_entries = processor.record_purchase(
        product, 5, "テストサプライヤー", "PO001"
    )

    # 結果を検証
    assert len(processor.journal_entries) == 3  # 売上1 + 仕入2

    # 残高を確認
    cash_balance = processor.get_account_balance(AccountCode.CASH)
    inventory_balance = processor.get_account_balance(AccountCode.INVENTORY)

    assert cash_balance == 1500.0  # 売上による現金増加
    assert inventory_balance > 0  # 仕入による在庫増加


# エラーハンドリングテスト
def test_journal_processor_error_handling():
    """仕訳プロセッサのエラーハンドリングテスト"""
    processor = JournalEntryProcessor()

    # 不正な取引（ゼロ金額）
    invalid_transaction = Transaction(
        transaction_id="TXN_INVALID",
        machine_id="VM001",
        items=[],
        subtotal=0.0,
        total_amount=0.0,
    )

    with pytest.raises(ValueError, match="売上金額は0より大きくなければなりません"):
        processor.record_sale(invalid_transaction)


# パフォーマンステスト
def test_journal_processor_performance():
    """仕訳プロセッサのパフォーマンステスト"""
    processor = JournalEntryProcessor()

    # 複数の取引を記録
    start_time = datetime.now()

    for i in range(100):
        transaction = Transaction(
            transaction_id=f"TXN{i}",
            machine_id="VM001",
            items=[],
            subtotal=1000.0,
            total_amount=1000.0,
        )
        processor.record_sale(transaction)

    elapsed_time = (datetime.now() - start_time).total_seconds()

    # 100件の処理が1秒以内に完了することを確認
    assert elapsed_time < 1.0
    assert len(processor.journal_entries) == 100


# データ整合性テスト
def test_accounting_data_integrity():
    """会計データの整合性テスト"""
    processor = JournalEntryProcessor()

    # 売上と仕入を記録
    transaction = Transaction(
        transaction_id="TXN001",
        machine_id="VM001",
        items=[],
        subtotal=1000.0,
        total_amount=1000.0,
    )
    processor.record_sale(transaction)

    # 試算表のバランスを確認
    trial_balance = processor.get_trial_balance()

    assert trial_balance["total_debit"] == trial_balance["total_credit"]

    # 各勘定科目の残高を確認
    for code, account in trial_balance["accounts"].items():
        # 残高は借方残高 - 貸方残高で計算されているはず
        expected_balance = account["debit"] - account["credit"]
        assert abs(account["balance"] - expected_balance) < 0.01


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
