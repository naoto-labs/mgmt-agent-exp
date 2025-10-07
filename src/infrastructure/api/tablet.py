import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from src.agents.customer_agent import customer_agent
from src.application.services.conversation_service import conversation_service
from src.application.services.inventory_service import inventory_service
from src.domain.analytics.event_tracker import EventSeverity, EventType, event_tracker
from src.domain.models.product import PRODUCT_CATEGORIES, SAMPLE_PRODUCTS, Product

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/interface/products", response_model=List[Dict[str, Any]])
async def get_tablet_products():
    """タブレット用商品一覧を取得"""
    try:
        products = []

        for product in SAMPLE_PRODUCTS:
            if product.is_available():
                product_data = product.to_dict()
                # タブレット用に追加情報を付与
                product_data.update(
                    {
                        "display_name": f"{product.name} - ¥{product.price}",
                        "image_available": product.image_url is not None,
                        "category_name": PRODUCT_CATEGORIES[product.category].name,
                        "popularity_score": product.sales_count
                        / max(sum(p.sales_count for p in SAMPLE_PRODUCTS), 1),
                    }
                )
                products.append(product_data)

        # 人気順にソート
        products.sort(key=lambda x: x["popularity_score"], reverse=True)

        return products

    except Exception as e:
        logger.error(f"タブレット商品一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail="商品一覧の取得に失敗しました")


@router.get("/interface/categories", response_model=List[Dict[str, Any]])
async def get_categories():
    """商品カテゴリ一覧を取得"""
    try:
        categories = []

        for category_info in PRODUCT_CATEGORIES.values():
            # カテゴリの商品数と利用可能数を計算
            category_products = [
                p for p in SAMPLE_PRODUCTS if p.category == category_info.category
            ]
            available_count = sum(1 for p in category_products if p.is_available())

            categories.append(
                {
                    "category_id": category_info.category.value,
                    "name": category_info.name,
                    "description": category_info.description,
                    "total_products": len(category_products),
                    "available_products": available_count,
                    "icon_url": category_info.icon_url,
                    "display_order": category_info.display_order,
                }
            )

        # 表示順序でソート
        categories.sort(key=lambda x: x["display_order"])

        return categories

    except Exception as e:
        logger.error(f"カテゴリ一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail="カテゴリ一覧の取得に失敗しました")


@router.post("/chat/start", response_model=Dict[str, Any])
async def start_customer_chat(
    customer_id: str = Body(..., embed=True),
    initial_message: Optional[str] = Body(None),
):
    """顧客とのチャットを開始"""
    try:
        # 顧客エンゲージメントを開始
        engagement_result = await customer_agent.engage_customer(customer_id, "VM001")

        if not engagement_result["success"]:
            raise HTTPException(status_code=500, detail="チャット開始に失敗しました")

        # イベントを記録
        event_tracker.track_event(
            EventType.CONVERSATION_START,
            "tablet_api",
            f"顧客チャット開始: {customer_id}",
            EventSeverity.LOW,
            {"customer_id": customer_id, "session_id": engagement_result["session_id"]},
        )

        return {
            "session_id": engagement_result["session_id"],
            "message": engagement_result["message"],
            "suggested_products": engagement_result["suggested_products"],
            "status": "active",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャット開始エラー: {e}")
        raise HTTPException(status_code=500, detail="チャット開始に失敗しました")


@router.post("/chat/message", response_model=Dict[str, Any])
async def send_chat_message(
    session_id: str = Body(..., embed=True), message: str = Body(..., embed=True)
):
    """チャットメッセージを送信"""
    try:
        # 顧客メッセージを処理
        response = await customer_agent.handle_customer_message(session_id, message)

        if not response["success"]:
            raise HTTPException(status_code=500, detail="メッセージ処理に失敗しました")

        # イベントを記録
        event_tracker.track_event(
            EventType.CUSTOMER_INTERACTION,
            "tablet_api",
            f"チャットメッセージ処理: {message[:50]}...",
            EventSeverity.LOW,
            {
                "session_id": session_id,
                "message_length": len(message),
                "has_suggestions": len(response["suggested_products"]) > 0,
            },
        )

        return {
            "message": response["message"],
            "suggested_products": response["suggested_products"],
            "insights": response["insights"],
            "timestamp": None,  # 実際には現在のタイムスタンプを追加
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャットメッセージ処理エラー: {e}")
        raise HTTPException(status_code=500, detail="メッセージ処理に失敗しました")


@router.get("/chat/history/{customer_id}", response_model=List[Dict[str, Any]])
async def get_chat_history(customer_id: str, limit: int = 10):
    """顧客のチャット履歴を取得"""
    try:
        history = await conversation_service.get_conversation_history(
            customer_id, limit
        )

        return [
            {
                "session_id": conv["session_id"],
                "start_time": conv["start_time"].isoformat(),
                "message_count": conv["message_count"],
                "summary": conv["summary"],
                "last_message": conv["summary"].split("最新発言: ")[-1]
                if "最新発言:" in conv["summary"]
                else None,
            }
            for conv in history
        ]

    except Exception as e:
        logger.error(f"チャット履歴取得エラー: {e}")
        raise HTTPException(status_code=500, detail="チャット履歴の取得に失敗しました")


@router.get("/recommendations/{customer_id}", response_model=List[str])
async def get_personalized_recommendations(customer_id: str):
    """パーソナライズされた商品推奨を取得"""
    try:
        recommendations = await customer_agent.generate_personalized_recommendations(
            customer_id
        )

        return recommendations

    except Exception as e:
        logger.error(f"パーソナライズ推奨取得エラー: {e}")
        raise HTTPException(status_code=500, detail="推奨の取得に失敗しました")


@router.get("/satisfaction/{customer_id}", response_model=Dict[str, Any])
async def get_customer_satisfaction(customer_id: str):
    """顧客満足度を取得"""
    try:
        satisfaction = await customer_agent.analyze_customer_satisfaction(customer_id)

        return satisfaction

    except Exception as e:
        logger.error(f"顧客満足度取得エラー: {e}")
        raise HTTPException(status_code=500, detail="満足度分析に失敗しました")


@router.get("/interface/status", response_model=Dict[str, Any])
async def get_tablet_interface_status():
    """タブレットインターフェースの状態を取得"""
    try:
        # 在庫状況を取得
        inventory_summary = inventory_service.get_inventory_summary()

        # システム健全性を取得
        health_score = event_tracker.get_system_health_score()

        # 人気商品を取得（サンプル）
        popular_products = []
        for product in SAMPLE_PRODUCTS:
            if product.is_available():
                popular_products.append(
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "popularity_score": product.sales_count,
                    }
                )

        popular_products.sort(key=lambda x: x["popularity_score"], reverse=True)

        return {
            "system_status": "operational" if health_score > 0.8 else "degraded",
            "health_score": health_score,
            "available_products": inventory_summary.active_slots,
            "total_products": inventory_summary.total_slots,
            "popular_products": popular_products[:5],
            "last_updated": None,  # 実際には現在のタイムスタンプを追加
            "features": {
                "chat_available": True,
                "payment_available": True,
                "recommendations_available": True,
                "multilingual_support": False,  # 将来的に実装
            },
        }

    except Exception as e:
        logger.error(f"タブレットインターフェース状態取得エラー: {e}")
        raise HTTPException(status_code=500, detail="状態取得に失敗しました")
