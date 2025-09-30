from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from datetime import date
import logging

from src.models.product import Product, SAMPLE_PRODUCTS
from src.models.transaction import Transaction, PaymentMethod, create_sample_transaction
from src.services.payment_service import payment_service
from src.services.inventory_service import inventory_service
from src.accounting.journal_entry import journal_processor
from src.analytics.event_tracker import event_tracker, EventType, EventSeverity
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/products", response_model=List[Dict[str, Any]])
async def get_products(
    category: Optional[str] = Query(None, description="商品カテゴリでフィルタ"),
    available_only: bool = Query(True, description="利用可能な商品のみ表示")
):
    """商品一覧を取得"""
    try:
        products = []

        # サンプル商品を使用（実際の実装ではデータベースから取得）
        for product in SAMPLE_PRODUCTS:
            if category and product.category.value != category:
                continue

            if available_only and not product.is_available():
                continue

            products.append(product.to_dict())

        return products

    except Exception as e:
        logger.error(f"商品一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail="商品一覧の取得に失敗しました")

@router.get("/products/{product_id}", response_model=Dict[str, Any])
async def get_product(product_id: str):
    """商品詳細を取得"""
    try:
        # サンプル商品から検索（実際の実装ではデータベースから取得）
        for product in SAMPLE_PRODUCTS:
            if product.product_id == product_id:
                return product.to_dict()

        raise HTTPException(status_code=404, detail="商品が見つかりません")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"商品詳細取得エラー: {e}")
        raise HTTPException(status_code=500, detail="商品詳細の取得に失敗しました")

@router.post("/purchase", response_model=Dict[str, Any])
async def purchase_product(
    product_id: str,
    quantity: int = Query(1, gt=0),
    payment_method: PaymentMethod = PaymentMethod.CARD
):
    """商品を購入"""
    try:
        # 商品情報を取得
        product = None
        for p in SAMPLE_PRODUCTS:
            if p.product_id == product_id:
                product = p
                break

        if not product:
            raise HTTPException(status_code=404, detail="商品が見つかりません")

        if not product.is_available():
            raise HTTPException(status_code=400, detail="商品が利用できません")

        if quantity > product.stock_quantity:
            raise HTTPException(status_code=400, detail="在庫が不足しています")

        # 総金額を計算
        total_amount = product.price * quantity

        # 決済処理
        payment_result = await payment_service.process_payment(total_amount, payment_method)

        if not payment_result.success:
            # 決済失敗イベントを記録
            event_tracker.track_event(
                EventType.PAYMENT,
                "vending_api",
                f"決済失敗: {payment_result.error_message}",
                EventSeverity.MEDIUM,
                {"product_id": product_id, "amount": total_amount, "method": payment_method.value}
            )
            raise HTTPException(status_code=400, detail=f"決済に失敗しました: {payment_result.error_message}")

        # 在庫から商品を排出
        success, message = inventory_service.dispense_product(product_id, quantity)

        if not success:
            # 在庫エラー時は返金処理を試行
            await payment_service.refund_payment(payment_result.payment_id)
            raise HTTPException(status_code=500, detail="商品の排出に失敗しました")

        # 取引オブジェクトを作成
        transaction = Transaction(
            machine_id=settings.machine_id,
            items=[],  # 簡易版のため空
            subtotal=total_amount,
            total_amount=total_amount,
            payment_details=None  # 実際にはPaymentDetailsオブジェクト
        )

        # 売上仕訳を記録
        journal_entry = journal_processor.record_sale(transaction)

        # イベントを記録
        event_tracker.track_event(
            EventType.SALE,
            "vending_api",
            f"商品販売完了: {product.name} x{quantity}",
            EventSeverity.LOW,
            {
                "product_id": product_id,
                "product_name": product.name,
                "quantity": quantity,
                "amount": total_amount,
                "payment_id": payment_result.payment_id
            }
        )

        return {
            "transaction_id": transaction.transaction_id,
            "product_name": product.name,
            "quantity": quantity,
            "total_amount": total_amount,
            "payment_id": payment_result.payment_id,
            "journal_entry_id": journal_entry.entry_id,
            "message": f"{product.name}を{quantity}個購入しました"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"商品購入エラー: {e}")
        raise HTTPException(status_code=500, detail="購入処理に失敗しました")

@router.get("/inventory", response_model=Dict[str, Any])
async def get_inventory_status():
    """在庫状況を取得"""
    try:
        summary = inventory_service.get_inventory_summary()
        alerts = inventory_service.get_alert_summary()

        return {
            "summary": summary.dict(),
            "alerts": alerts,
            "low_stock_products": [
                slot.product_name for slot in inventory_service.get_low_stock_slots()
            ],
            "out_of_stock_products": [
                slot.product_name for slot in inventory_service.get_out_of_stock_slots()
            ]
        }

    except Exception as e:
        logger.error(f"在庫状況取得エラー: {e}")
        raise HTTPException(status_code=500, detail="在庫状況の取得に失敗しました")

@router.post("/inventory/restock")
async def restock_inventory(
    product_id: str,
    quantity: int = Query(..., gt=0),
    slot_id: Optional[str] = Query(None)
):
    """在庫を補充"""
    try:
        if slot_id:
            # 指定スロットを補充
            success, message = inventory_service.restock_slot(slot_id, quantity)
        else:
            # 商品の全スロットを補充（簡易版）
            success, message = inventory_service.transfer_to_vending_machine(product_id, quantity)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        # イベントを記録
        event_tracker.track_event(
            EventType.RESTOCK,
            "vending_api",
            f"在庫補充完了: {product_id} x{quantity}",
            EventSeverity.LOW,
            {"product_id": product_id, "quantity": quantity, "slot_id": slot_id}
        )

        return {"message": message, "success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"在庫補充エラー: {e}")
        raise HTTPException(status_code=500, detail="在庫補充に失敗しました")

@router.get("/transactions", response_model=List[Dict[str, Any]])
async def get_transactions(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(50, le=100)
):
    """取引履歴を取得"""
    try:
        # 簡易的な実装：イベントから取引情報を抽出
        sales_events = event_tracker.get_events(
            EventType.SALE,
            limit=limit
        )

        transactions = []
        for event in sales_events:
            if start_date and event.timestamp.date() < start_date:
                continue
            if end_date and event.timestamp.date() > end_date:
                continue

            transactions.append({
                "transaction_id": event.data.get("transaction_id", event.event_id),
                "product_name": event.data.get("product_name", "不明"),
                "amount": event.data.get("amount", 0),
                "quantity": event.data.get("quantity", 1),
                "timestamp": event.timestamp.isoformat(),
                "status": "completed"
            })

        return transactions

    except Exception as e:
        logger.error(f"取引履歴取得エラー: {e}")
        raise HTTPException(status_code=500, detail="取引履歴の取得に失敗しました")

@router.get("/sales/summary", response_model=Dict[str, Any])
async def get_sales_summary(
    days: int = Query(7, ge=1, le=90)
):
    """売上サマリを取得"""
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 日別売上データを取得
        daily_sales = event_tracker.get_daily_sales_data(start_date, end_date)

        # サマリ計算
        total_revenue = sum(day["revenue"] for day in daily_sales)
        total_transactions = sum(day["transaction_count"] for day in daily_sales)
        avg_daily_revenue = total_revenue / max(len(daily_sales), 1)

        return {
            "period_days": days,
            "total_revenue": total_revenue,
            "total_transactions": total_transactions,
            "average_daily_revenue": avg_daily_revenue,
            "daily_breakdown": daily_sales,
            "generated_at": end_date.isoformat()
        }

    except Exception as e:
        logger.error(f"売上サマリ取得エラー: {e}")
        raise HTTPException(status_code=500, detail="売上サマリの取得に失敗しました")

@router.get("/health/vending", response_model=Dict[str, Any])
async def get_vending_health():
    """自動販売機の健全性をチェック"""
    try:
        inventory_summary = inventory_service.get_inventory_summary()
        health_score = event_tracker.get_system_health_score()

        # 在庫健全性を評価
        inventory_health = "good"
        if inventory_summary.out_of_stock_slots > inventory_summary.total_slots * 0.3:
            inventory_health = "critical"
        elif inventory_summary.out_of_stock_slots > inventory_summary.total_slots * 0.1:
            inventory_health = "warning"
        elif inventory_summary.low_stock_slots > inventory_summary.total_slots * 0.2:
            inventory_health = "warning"

        return {
            "overall_health_score": health_score,
            "inventory_health": inventory_health,
            "inventory_summary": inventory_summary.dict(),
            "alert_count": len(inventory_service.get_alerts()),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"自動販売機健全性チェックエラー: {e}")
        raise HTTPException(status_code=500, detail="健全性チェックに失敗しました")
