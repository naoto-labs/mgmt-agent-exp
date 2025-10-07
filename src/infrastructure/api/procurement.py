import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

# from src.agents.search_agent import search_agent, SearchResult, PriceComparison  # TODO: search_agent未実装
from src.application.services.inventory_service import inventory_service
from src.domain.analytics.event_tracker import EventSeverity, EventType, event_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/suppliers/search", response_model=List[Dict[str, Any]])
async def search_suppliers(
    product_name: str = Query(..., description="検索する商品名"),
    max_results: int = Query(10, ge=1, le=20),
):
    """仕入れ先を検索"""
    try:
        # TODO: search_agent未実装のためダミーレスポンスを返す
        # 実際にはsearch_agent.find_suppliers(product_name, max_results)の結果を使用
        formatted_results = [
            {
                "title": f"{product_name} - サプライヤー{i}",
                "url": f"https://supplier{i}.example.com/product/{product_name}",
                "price": 120 + i * 10,
                "currency": "JPY",
                "availability": "in_stock",
                "relevance_score": 0.8 - i * 0.1,
                "snippet": f"{product_name}の高品質品を提供。信頼できるサプライヤー。",
                "source": "test_supplier",
                "timestamp": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
            }
            for i in range(min(max_results, 3))
        ]

        # イベントを記録
        event_tracker.track_event(
            EventType.AI_DECISION,
            "procurement_api",
            f"仕入れ先検索実行: {product_name}",
            EventSeverity.LOW,
            {"product_name": product_name, "results_count": len(formatted_results)},
        )

        return formatted_results

    except Exception as e:
        logger.error(f"仕入れ先検索エラー: {e}")
        raise HTTPException(status_code=500, detail="仕入れ先検索に失敗しました")


@router.get("/prices/compare", response_model=Dict[str, Any])
async def compare_prices(
    product_name: str = Query(..., description="価格比較する商品名"),
):
    """価格比較を実行"""
    try:
        # TODO: search_agent未実装のためダミーレスポンスを返す
        # 実際にはawait search_agent.compare_prices(product_name)の結果を使用
        now = (
            __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat()
        )

        # ダミーの価格比較結果を生成
        result = {
            "product_name": product_name,
            "best_price": 110,
            "average_price": 125,
            "price_range": "110-140",
            "recommendation": "最安値のサプライヤーを推奨。最良の条件で発注してください。",
            "confidence": 0.85,
            "search_results": [
                {
                    "title": f"{product_name} 安価サプライヤー{i}",
                    "url": f"https://cheap-supplier{i}.example.com",
                    "price": 110 + i * 10,
                    "availability": "in_stock",
                    "relevance_score": 0.9 - i * 0.1,
                    "snippet": f"{product_name}の価格比較対象商品",
                }
                for i in range(4)
            ],
            "generated_at": now,
        }

        # イベントを記録
        event_tracker.track_event(
            EventType.AI_DECISION,
            "procurement_api",
            f"価格比較実行: {product_name}",
            EventSeverity.LOW,
            {
                "product_name": product_name,
                "best_price": result["best_price"],
                "results_count": len(result["search_results"]),
            },
        )

        return result

    except Exception as e:
        logger.error(f"価格比較エラー: {e}")
        raise HTTPException(status_code=500, detail="価格比較に失敗しました")


@router.get("/inventory/needs", response_model=Dict[str, Any])
async def get_procurement_needs():
    """調達ニーズを取得"""
    try:
        # 在庫サマリを取得
        inventory_summary = inventory_service.get_inventory_summary()

        # 低在庫・在庫切れ商品を取得
        low_stock_slots = inventory_service.get_low_stock_slots()
        out_of_stock_slots = inventory_service.get_out_of_stock_slots()

        # 補充計画を生成
        restock_plan = inventory_service.generate_restock_plan()

        # 調達優先度を計算
        urgent_needs = []
        normal_needs = []

        for item in restock_plan.items:
            if restock_plan.priority in ["urgent", "high"]:
                urgent_needs.append(item)
            else:
                normal_needs.append(item)

        return {
            "inventory_summary": inventory_summary.dict(),
            "low_stock_count": len(low_stock_slots),
            "out_of_stock_count": len(out_of_stock_slots),
            "restock_plan": restock_plan.dict(),
            "urgent_needs": urgent_needs,
            "normal_needs": normal_needs,
            "total_estimated_cost": restock_plan.total_cost,
            "generated_at": None,  # 実際には現在のタイムスタンプを追加
        }

    except Exception as e:
        logger.error(f"調達ニーズ取得エラー: {e}")
        raise HTTPException(status_code=500, detail="調達ニーズの取得に失敗しました")


@router.post("/orders/generate")
async def generate_purchase_order(
    product_id: str = Body(..., embed=True),
    quantity: int = Body(..., gt=0),
    supplier_name: str = Body(..., embed=True),
    supplier_url: str = Body(..., embed=True),
):
    """発注指示書を生成"""
    try:
        # 商品情報を取得（サンプルから）
        from src.domain.models.product import SAMPLE_PRODUCTS

        product = None
        for p in SAMPLE_PRODUCTS:
            if p.product_id == product_id:
                product = p
                break

        if not product:
            raise HTTPException(status_code=404, detail="商品が見つかりません")

        # 発注情報を生成
        order_info = {
            "order_id": f"PO_{product_id}_{int(__import__('time').time())}",
            "product_id": product_id,
            "product_name": product.name,
            "quantity": quantity,
            "unit_price": product.cost,
            "total_amount": product.cost * quantity,
            "supplier_name": supplier_name,
            "supplier_url": supplier_url,
            "order_date": None,  # 実際には現在の日付を追加
            "estimated_delivery": None,  # 実際には計算した日付を追加
            "status": "pending_approval",
        }

        # イベントを記録
        event_tracker.track_event(
            EventType.AI_DECISION,
            "procurement_api",
            f"発注指示書生成: {product.name} x{quantity}",
            EventSeverity.MEDIUM,
            {
                "product_id": product_id,
                "quantity": quantity,
                "supplier": supplier_name,
                "total_amount": order_info["total_amount"],
            },
        )

        return {
            "order": order_info,
            "message": f"{product.name}の発注指示書を生成しました",
            "requires_approval": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"発注指示書生成エラー: {e}")
        raise HTTPException(status_code=500, detail="発注指示書の生成に失敗しました")


@router.get("/analytics/suppliers", response_model=Dict[str, Any])
async def get_supplier_analytics():
    """仕入れ先分析データを取得"""
    try:
        # TODO: search_agent未実装のためダミーレスポンスを返す
        # 実際にはsearch_agent.get_search_stats()を使用
        search_stats = {
            "total_searches": 42,
            "success_rate": 0.95,
            "avg_response_time": 2.3,
            "last_search_time": "2025-10-03T13:57:17Z",
        }

        # 在庫効率性分析を取得（サンプル商品）
        efficiency_data = []
        for i in range(1, 4):
            efficiency = inventory_service._calculate_optimal_stock_level(
                type("MockSlot", (), {"max_quantity": 100})()  # モックスロット
            )
            efficiency_data.append(
                {
                    "product_id": f"product_{i}",
                    "optimal_stock_level": efficiency,
                    "current_trend": "stable",
                }
            )

        return {
            "search_statistics": search_stats,
            "supplier_performance": [
                {
                    "supplier_name": f"サプライヤー{i}",
                    "products_supplied": i * 2,
                    "avg_delivery_time": 3 + i,
                    "quality_rating": 4.5 - i * 0.2,
                    "reliability_score": 0.9 - i * 0.05,
                }
                for i in range(1, 4)
            ],
            "inventory_efficiency": efficiency_data,
            "market_trends": {
                "price_trend": "stable",
                "demand_trend": "increasing",
                "supply_availability": "good",
            },
            "generated_at": None,  # 実際には現在のタイムスタンプを追加
        }

    except Exception as e:
        logger.error(f"仕入れ先分析データ取得エラー: {e}")
        raise HTTPException(status_code=500, detail="分析データの取得に失敗しました")


@router.get("/reports/procurement", response_model=Dict[str, Any])
async def get_procurement_report(days: int = Query(30, ge=1, le=90)):
    """調達レポートを取得"""
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 期間内のイベントを取得
        procurement_events = event_tracker.get_events(
            EventType.RESTOCK, start_date, end_date
        )

        # レポートデータを集計
        total_restock_events = len(procurement_events)
        total_quantity_restocked = sum(
            event.data.get("quantity", 0) for event in procurement_events
        )

        # 商品別集計
        product_restock = {}
        for event in procurement_events:
            product_name = event.data.get("product_name", "不明")
            quantity = event.data.get("quantity", 0)

            if product_name not in product_restock:
                product_restock[product_name] = {"quantity": 0, "events": 0}

            product_restock[product_name]["quantity"] += quantity
            product_restock[product_name]["events"] += 1

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days,
            },
            "summary": {
                "total_restock_events": total_restock_events,
                "total_quantity_restocked": total_quantity_restocked,
                "unique_products_restocked": len(product_restock),
            },
            "product_breakdown": [
                {
                    "product_name": name,
                    "total_quantity": data["quantity"],
                    "restock_events": data["events"],
                    "avg_quantity_per_event": data["quantity"] / data["events"],
                }
                for name, data in product_restock.items()
            ],
            "trends": {
                "restock_frequency": "安定"
                if total_restock_events > days * 0.5
                else "減少",
                "quantity_trend": "増加"
                if total_quantity_restocked > days * 10
                else "安定",
            },
            "generated_at": end_date.isoformat(),
        }

    except Exception as e:
        logger.error(f"調達レポート取得エラー: {e}")
        raise HTTPException(status_code=500, detail="調達レポートの取得に失敗しました")
