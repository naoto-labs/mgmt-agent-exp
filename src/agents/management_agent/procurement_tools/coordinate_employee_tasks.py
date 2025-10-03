"""
coordinate_employee_tasks.py - 従業員業務配分ツール

調達・補充関連全業務の進捗管理・調整Tool
"""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def coordinate_employee_tasks() -> Dict[str, Any]:
    """発注/補充が必要な場合に従業員1人にメール通知 + 新商品発注処理"""
    logger.info("Coordinating employee tasks")

    notifications = []
    employees_status = {}

    # === 1. 在庫補充タスク ===
    from src.agents.shared_tools.tools.data_retrieval.check_inventory_status import (
        check_inventory_status,
    )

    inventory_status = await check_inventory_status()
    low_stock_items = inventory_status.get("low_stock_items", [])

    if low_stock_items:
        notification = {
            "recipient": "employee@vending-company.com",
            "subject": "在庫補充依頼",
            "body": f"以下の商品が在庫不足です。補充をお願いします: {', '.join(low_stock_items)}",
            "priority": "normal",
            "timestamp": datetime.now().isoformat(),
            "task_type": "restock",
        }
        notifications.append(notification)
        employees_status["restock"] = low_stock_items
        logger.info(f"在庫補充通知送信: {low_stock_items}")

    # === 2. 新商品発注タスク ===
    # 在庫データを基に新商品検索クエリを生成
    try:
        # 在庫状況から商品カテゴリを把握
        from src.agents.management_agent.management_tools.get_business_metrics import (
            get_business_metrics,
        )

        metrics = get_business_metrics()
        inventory_level = metrics.get("inventory_level", {})
        sales = metrics.get("sales", 0)

        # カテゴリ別在庫を確認
        drink_categories = [
            item
            for item in inventory_level.keys()
            if "コーラ" in item or "飲料" in item or "ジュース" in item
        ]
        food_categories = [
            item
            for item in inventory_level.keys()
            if "チップス" in item or "ヌードル" in item or "お菓子" in item
        ]

        # 売上実績に基づいて検索クエリを決定
        if sales > 1000:  # 売上が良い場合
            search_query = "人気飲料 新商品"
            logger.info("売上好調のため、新商品飲料を検索")
        elif (
            drink_categories
            and min([inventory_level.get(cat, 0) for cat in drink_categories]) < 5
        ):  # 飲料在庫が少ない場合
            search_query = "人気清涼飲料 ボトル飲料"
            logger.info("飲料在庫不足のため、供給安定した飲料を検索")
        elif food_categories:
            search_query = "人気スナック 健康志向"
            logger.info("既存食品を補完する人気スナックを検索")
        else:
            search_query = "人気飲料"
            logger.info("デフォルトで人気飲料を検索")

        logger.info(f"生成された検索クエリ: {search_query}")

        # Shared Toolsから商品検索機能を使用
        from src.agents.shared_tools import shared_registry

        search_tool = shared_registry.get_tool("market_search")
        if search_tool:
            search_results = await search_tool.asearch(query=search_query)
            logger.info(
                f"検索結果取得: {len(search_results) if search_results else 0}件 (クエリ: {search_query})"
            )
            recommended_products = (
                search_results[:2] if search_results else []
            )  # 上位2つ
        else:
            recommended_products = []
            logger.warning("検索ツールが利用できません")

        if recommended_products:
            procurement_tasks = []
            for product in recommended_products[:2]:  # dict形式を想定
                # Procurement AgentからShared Toolsに変更
                procurement_tool = shared_registry.get_tool("procurement_order")
                if procurement_tool:
                    procurement_result = await procurement_tool.aexecute(
                        product_info={
                            "product_name": product.get("name", "") or product,
                            "recommended_quantity": 10,
                        },
                        supplier_info={
                            "name": "Search Supplier",
                            "url": product.get("url", ""),
                            "price": product.get("price", 150),
                        },
                    )

                    if procurement_result.get("success"):
                        order = procurement_result.get("order", {})
                        procurement_tasks.append(
                            {
                                "product": product.get("name", "") or product,
                                "order_id": order.get("order_id", "unknown"),
                            }
                        )

            if procurement_tasks:
                procurement_notification = {
                    "recipient": "employee@vending-company.com",
                    "subject": "新商品発注完了通知",
                    "body": f"以下の新商品を発注しました。入荷管理をお願いします:\n"
                    + "\n".join(
                        [
                            f"- {t['product']} (注文ID: {t['order_id']})"
                            for t in procurement_tasks
                        ]
                    ),
                    "priority": "high",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": "new_procurement",
                    "orders": procurement_tasks,
                }
                notifications.append(procurement_notification)
                employees_status["new_procurement"] = [
                    t["product"] for t in procurement_tasks
                ]
                logger.info(f"新商品発注通知送信: {len(procurement_tasks)}件")

    except Exception as e:
        logger.error(f"新商品発注プロセスエラー: {e}")

    # === 結果返却 ===
    if notifications:
        return {
            "active_tasks": len(notifications),
            "completed_today": 0,
            "pending": len(notifications),
            "notifications_sent": notifications,
            "employees": {"employee@vending-company.com": employees_status},
        }
    else:
        return {
            "active_tasks": 0,
            "completed_today": 0,
            "pending": 0,
            "notifications_sent": [],
            "employees": {"employee@vending-company.com": "特記事項なし"},
        }
