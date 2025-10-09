import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.domain.models.product import Product
from src.domain.models.transaction import Transaction, TransactionType
from src.shared.config.settings import settings

logger = logging.getLogger(__name__)


class AccountCode(str, Enum):
    """勘定科目コード"""

    # 資産
    CASH = "1001"  # 現金
    INVENTORY = "1101"  # 商品

    # 負債
    ACCOUNTS_PAYABLE = "2001"  # 買掛金

    # 収益
    SALES_REVENUE = "4001"  # 売上高

    # 費用
    COST_OF_GOODS_SOLD = "5001"  # 仕入高
    OPERATING_EXPENSES = "6001"  # 販売費及び一般管理費


class DebitCredit(str, Enum):
    """借方・貸方"""

    DEBIT = "debit"
    CREDIT = "credit"


@dataclass
class AccountingEntry:
    """会計エントリ"""

    account_code: str
    account_name: str
    debit_amount: float = 0.0
    credit_amount: float = 0.0
    description: Optional[str] = None

    def is_debit(self) -> bool:
        """借方エントリかチェック"""
        return self.debit_amount > 0

    def is_credit(self) -> bool:
        """貸方エントリかチェック"""
        return self.credit_amount > 0

    def get_amount(self) -> float:
        """金額を取得"""
        return max(self.debit_amount, self.credit_amount)


@dataclass
class JournalEntry:
    """仕訳エントリ"""

    entry_id: str
    date: date
    description: str
    entries: List[AccountingEntry]
    reference_id: Optional[str] = None  # 取引IDや発注ID
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def is_balanced(self) -> bool:
        """借貸バランスが取れているかチェック"""
        total_debit = sum(entry.debit_amount for entry in self.entries)
        total_credit = sum(entry.credit_amount for entry in self.entries)
        return abs(total_debit - total_credit) < 0.01  # 浮動小数点誤差を考慮

    def get_total_amount(self) -> float:
        """総金額を取得"""
        return (
            sum(entry.get_amount() for entry in self.entries) / 2
        )  # 借貸同額なので半分


class ChartOfAccounts:
    """勘定科目表"""

    def __init__(self):
        self.accounts = {
            AccountCode.CASH: "現金",
            AccountCode.INVENTORY: "商品",
            AccountCode.ACCOUNTS_PAYABLE: "買掛金",
            AccountCode.SALES_REVENUE: "売上高",
            AccountCode.COST_OF_GOODS_SOLD: "仕入高",
            AccountCode.OPERATING_EXPENSES: "販売費及び一般管理費",
        }

    def get_account_name(self, code: str) -> str:
        """勘定科目名を取得"""
        return self.accounts.get(code, "不明な科目")

    def get_all_accounts(self) -> Dict[str, str]:
        """全勘定科目を取得"""
        return self.accounts.copy()


class JournalEntryProcessor:
    """仕訳処理クラス"""

    def __init__(self):
        self.chart_of_accounts = ChartOfAccounts()
        self.journal_entries: List[JournalEntry] = []
        self.entry_counter = 0

    def _generate_entry_id(self) -> str:
        """仕訳IDを生成"""
        self.entry_counter += 1
        return f"JE{datetime.now().strftime('%Y%m%d')}{self.entry_counter:04d}"

    def record_sale(self, transaction: Transaction) -> JournalEntry:
        """売上仕訳を記録"""
        logger.info(f"売上仕訳記録開始: {transaction.transaction_id}")

        try:
            # 取引金額を取得
            amount = transaction.total_amount

            if amount <= 0:
                raise ValueError("売上金額は0より大きくなければなりません")

            # 売上仕訳エントリを作成
            entries = [
                AccountingEntry(
                    account_code=AccountCode.CASH.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.CASH.value
                    ),
                    debit_amount=amount,
                    credit_amount=0,
                    description=f"商品売上 - {transaction.transaction_id}",
                ),
                AccountingEntry(
                    account_code=AccountCode.SALES_REVENUE.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.SALES_REVENUE.value
                    ),
                    debit_amount=0,
                    credit_amount=amount,
                    description=f"商品売上 - {transaction.transaction_id}",
                ),
            ]

            # created_atがdatetimeなら.date()、dateならそのまま
            entry_date = (
                transaction.created_at.date()
                if hasattr(transaction.created_at, "date")
                and callable(getattr(transaction.created_at, "date"))
                else transaction.created_at
            )

            # 仕訳エントリを作成
            journal_entry = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=entry_date,
                description=f"商品売上 - 取引ID: {transaction.transaction_id}",
                entries=entries,
                reference_id=transaction.transaction_id,
            )

            # バランスチェック
            if not journal_entry.is_balanced():
                raise ValueError("仕訳の借貸バランスが取れていません")

            # 仕訳を保存
            self.journal_entries.append(journal_entry)

            logger.info(f"売上仕訳記録完了: {journal_entry.entry_id}")
            return journal_entry

        except Exception as e:
            logger.error(f"売上仕訳記録エラー: {e}")
            raise

    def record_purchase(
        self, product: Product, quantity: int, supplier_name: str, order_id: str
    ) -> List[JournalEntry]:
        """仕入仕訳を記録"""
        logger.info(f"仕入仕訳記録開始: 商品={product.name}, 数量={quantity}")

        try:
            # 仕入金額を計算
            total_cost = product.cost * quantity

            if total_cost <= 0:
                raise ValueError("仕入金額は0より大きくなければなりません")

            # 1. 仕入時：借方：仕入高、貸方：買掛金
            purchase_entries = [
                AccountingEntry(
                    account_code=AccountCode.COST_OF_GOODS_SOLD.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.COST_OF_GOODS_SOLD.value
                    ),
                    debit_amount=total_cost,
                    credit_amount=0,
                    description=f"商品仕入 - {supplier_name}",
                ),
                AccountingEntry(
                    account_code=AccountCode.ACCOUNTS_PAYABLE.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.ACCOUNTS_PAYABLE.value
                    ),
                    debit_amount=0,
                    credit_amount=total_cost,
                    description=f"商品仕入 - {supplier_name}",
                ),
            ]

            purchase_journal = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=date.today(),
                description=f"商品仕入 - 仕入先: {supplier_name}, 商品: {product.name}",
                entries=purchase_entries,
                reference_id=order_id,
            )

            # 2. 在庫計上：借方：商品、貸方：仕入高
            inventory_entries = [
                AccountingEntry(
                    account_code=AccountCode.INVENTORY.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.INVENTORY.value
                    ),
                    debit_amount=total_cost,
                    credit_amount=0,
                    description=f"在庫計上 - {product.name}",
                ),
                AccountingEntry(
                    account_code=AccountCode.COST_OF_GOODS_SOLD.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.COST_OF_GOODS_SOLD.value
                    ),
                    debit_amount=0,
                    credit_amount=total_cost,
                    description=f"在庫計上 - {product.name}",
                ),
            ]

            inventory_journal = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=date.today(),
                description=f"在庫計上 - 商品: {product.name}, 数量: {quantity}",
                entries=inventory_entries,
                reference_id=order_id,
            )

            # バランスチェック
            for journal in [purchase_journal, inventory_journal]:
                if not journal.is_balanced():
                    raise ValueError(
                        f"仕訳の借貸バランスが取れていません: {journal.entry_id}"
                    )

            # 仕訳を保存
            journal_entries = [purchase_journal, inventory_journal]
            self.journal_entries.extend(journal_entries)

            logger.info(f"仕入仕訳記録完了: {len(journal_entries)}件")
            return journal_entries

        except Exception as e:
            logger.error(f"仕入仕訳記録エラー: {e}")
            raise

    def record_inventory_adjustment(
        self, product: Product, adjustment_quantity: int, reason: str
    ) -> JournalEntry:
        """在庫調整仕訳を記録"""
        logger.info(
            f"在庫調整仕訳記録開始: 商品={product.name}, 調整数量={adjustment_quantity}"
        )

        try:
            # 調整金額を計算
            adjustment_amount = abs(product.cost * adjustment_quantity)

            if adjustment_amount == 0:
                raise ValueError("調整金額は0より大きくなければなりません")

            # 在庫増減に応じてエントリを作成
            if adjustment_quantity > 0:
                # 在庫増加：借方：商品、貸方：仕入高（再振替）
                entries = [
                    AccountingEntry(
                        account_code=AccountCode.INVENTORY.value,
                        account_name=self.chart_of_accounts.get_account_name(
                            AccountCode.INVENTORY.value
                        ),
                        debit_amount=adjustment_amount,
                        credit_amount=0,
                        description=f"在庫増加調整 - {reason}",
                    ),
                    AccountingEntry(
                        account_code=AccountCode.COST_OF_GOODS_SOLD.value,
                        account_name=self.chart_of_accounts.get_account_name(
                            AccountCode.COST_OF_GOODS_SOLD.value
                        ),
                        debit_amount=0,
                        credit_amount=adjustment_amount,
                        description=f"在庫増加調整 - {reason}",
                    ),
                ]
            else:
                # 在庫減少：借方：仕入高（再振替）、貸方：商品
                entries = [
                    AccountingEntry(
                        account_code=AccountCode.COST_OF_GOODS_SOLD.value,
                        account_name=self.chart_of_accounts.get_account_name(
                            AccountCode.COST_OF_GOODS_SOLD.value
                        ),
                        debit_amount=adjustment_amount,
                        credit_amount=0,
                        description=f"在庫減少調整 - {reason}",
                    ),
                    AccountingEntry(
                        account_code=AccountCode.INVENTORY.value,
                        account_name=self.chart_of_accounts.get_account_name(
                            AccountCode.INVENTORY.value
                        ),
                        debit_amount=0,
                        credit_amount=adjustment_amount,
                        description=f"在庫減少調整 - {reason}",
                    ),
                ]

            journal_entry = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=date.today(),
                description=f"在庫調整 - 商品: {product.name}, 理由: {reason}",
                entries=entries,
                reference_id=product.product_id,
            )

            # バランスチェック
            if not journal_entry.is_balanced():
                raise ValueError("仕訳の借貸バランスが取れていません")

            # 仕訳を保存
            self.journal_entries.append(journal_entry)

            logger.info(f"在庫調整仕訳記録完了: {journal_entry.entry_id}")
            return journal_entry

        except Exception as e:
            logger.error(f"在庫調整仕訳記録エラー: {e}")
            raise

    def record_expense(
        self, expense_type: str, amount: float, description: str
    ) -> JournalEntry:
        """費用仕訳を記録"""
        logger.info(f"費用仕訳記録開始: {expense_type}, 金額={amount}")

        try:
            if amount <= 0:
                raise ValueError("費用金額は0より大きくなければなりません")

            # 費用仕訳エントリを作成
            entries = [
                AccountingEntry(
                    account_code=AccountCode.OPERATING_EXPENSES.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.OPERATING_EXPENSES.value
                    ),
                    debit_amount=amount,
                    credit_amount=0,
                    description=f"{expense_type} - {description}",
                ),
                AccountingEntry(
                    account_code=AccountCode.CASH.value,
                    account_name=self.chart_of_accounts.get_account_name(
                        AccountCode.CASH.value
                    ),
                    debit_amount=0,
                    credit_amount=amount,
                    description=f"{expense_type} - {description}",
                ),
            ]

            journal_entry = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=date.today(),
                description=f"{expense_type} - {description}",
                entries=entries,
            )

            # バランスチェック
            if not journal_entry.is_balanced():
                raise ValueError("仕訳の借貸バランスが取れていません")

            # 仕訳を保存
            self.journal_entries.append(journal_entry)

            logger.info(f"費用仕訳記録完了: {journal_entry.entry_id}")
            return journal_entry

        except Exception as e:
            logger.error(f"費用仕訳記録エラー: {e}")
            raise

    def get_journal_entries(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[JournalEntry]:
        """期間指定で仕訳を取得"""
        if not start_date:
            start_date = date.min
        if not end_date:
            end_date = date.today()

        filtered_entries = [
            entry
            for entry in self.journal_entries
            if start_date <= entry.date <= end_date
        ]

        return sorted(filtered_entries, key=lambda x: x.date)

    def get_account_balance(
        self,
        account_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> float:
        """勘定科目の残高を取得"""
        entries = self.get_journal_entries(start_date, end_date)

        balance = 0.0
        for entry in entries:
            for acc_entry in entry.entries:
                if acc_entry.account_code == account_code:
                    balance += acc_entry.debit_amount - acc_entry.credit_amount

        return balance

    def get_trial_balance(
        self, end_date: Optional[date] = None
    ) -> Dict[str, Dict[str, float]]:
        """試算表を取得"""
        if not end_date:
            end_date = date.today()

        entries = self.get_journal_entries(end_date=end_date)

        account_balances = {}

        for entry in entries:
            for acc_entry in entry.entries:
                code = acc_entry.account_code
                if code not in account_balances:
                    account_balances[code] = {
                        "debit": 0.0,
                        "credit": 0.0,
                        "balance": 0.0,
                    }

                account_balances[code]["debit"] += acc_entry.debit_amount
                account_balances[code]["credit"] += acc_entry.credit_amount
                account_balances[code]["balance"] += (
                    acc_entry.debit_amount - acc_entry.credit_amount
                )

        return {
            "accounts": account_balances,
            "total_debit": sum(acc["debit"] for acc in account_balances.values()),
            "total_credit": sum(acc["credit"] for acc in account_balances.values()),
            "as_of_date": end_date.isoformat(),
        }

    def export_journal_entries(self, format: str = "json") -> str:
        """仕訳をエクスポート"""
        if format.lower() == "json":
            entries_data = [
                {
                    "entry_id": entry.entry_id,
                    "date": entry.date.isoformat(),
                    "description": entry.description,
                    "entries": [
                        {
                            "account_code": acc_entry.account_code,
                            "account_name": acc_entry.account_name,
                            "debit_amount": acc_entry.debit_amount,
                            "credit_amount": acc_entry.credit_amount,
                            "description": acc_entry.description,
                        }
                        for acc_entry in entry.entries
                    ],
                    "reference_id": entry.reference_id,
                    "created_at": entry.created_at.isoformat(),
                }
                for entry in self.journal_entries
            ]

            import json

            return json.dumps(
                {
                    "journal_entries": entries_data,
                    "total_entries": len(entries_data),
                    "exported_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )

        else:
            raise ValueError(f"未対応のエクスポート形式: {format}")

    def get_accounting_summary(self) -> Dict[str, Any]:
        """会計サマリを取得"""
        today = date.today()

        # 月次集計
        month_start = today.replace(day=1)
        entries = self.get_journal_entries(month_start, today)

        summary = {
            "period": {"start": month_start.isoformat(), "end": today.isoformat()},
            "total_entries": len(entries),
            "account_balances": {},
            "top_accounts": [],
        }

        # 勘定科目別残高
        for code, name in self.chart_of_accounts.get_all_accounts().items():
            balance = self.get_account_balance(code, month_start, today)
            if balance != 0:
                summary["account_balances"][code] = {"name": name, "balance": balance}

        # 上位勘定科目（残高順）
        balances = [
            {"code": code, "name": info["name"], "balance": abs(info["balance"])}
            for code, info in summary["account_balances"].items()
        ]
        balances.sort(key=lambda x: x["balance"], reverse=True)
        summary["top_accounts"] = balances[:5]

        return summary

    def is_duplicate_entry(self, transaction_id: str) -> bool:
        """トランザクションIDによる重複チェック"""
        for entry in self.journal_entries:
            if entry.reference_id == transaction_id:
                logger.warning(f"重複トランザクション検知: {transaction_id}")
                return True
        return False

    def add_entry(
        self,
        account_number: str,
        date: date,
        amount: float,
        entry_type: str,
        description: str,
        transaction_id: Optional[str] = None,
    ):
        """エントリを追加（単一勘定）- 重複チェック付き"""
        try:
            # トランザクションIDがある場合は重複チェック
            if transaction_id and self.is_duplicate_entry(transaction_id):
                logger.warning(f"重複エントリのためスキップ: {transaction_id}")
                return None

            debit_amount = amount if entry_type == "debit" else 0.0
            credit_amount = amount if entry_type == "credit" else 0.0

            entry = AccountingEntry(
                account_code=account_number,
                account_name=self.chart_of_accounts.get_account_name(account_number),
                debit_amount=debit_amount,
                credit_amount=credit_amount,
                description=description,
            )

            journal_entry = JournalEntry(
                entry_id=self._generate_entry_id(),
                date=date,
                description=description,
                entries=[entry],
                reference_id=transaction_id,  # トランザクションIDを記録
            )

            self.journal_entries.append(journal_entry)
            logger.debug(
                f"エントリ追加完了: {journal_entry.entry_id} (ID: {transaction_id})"
            )
            return journal_entry

        except Exception as e:
            logger.error(f"エントリ追加エラー: {e}")
            raise


# グローバルインスタンス
journal_processor = JournalEntryProcessor()
