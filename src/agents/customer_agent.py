"""
Customer Agent - 顧客対応専門エージェント

顧客問い合わせ対応、顧客エンゲージメント、レコメンデーション生成を行う
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CustomerAgent:
    """顧客対応エージェント"""

    def __init__(self):
        logger.info("CustomerAgent initialized")
        self.customer_data = {}  # 簡易顧客データベース
        self.sessions = {}  # アクティブな顧客セッション

    async def respond_to_inquiry(
        self, customer_id: str, inquiry: str
    ) -> Dict[str, Any]:
        """顧客問い合わせに対応"""
        logger.info(f"Responding to customer {customer_id} inquiry: {inquiry}")

        # 基本的な応答ロジック
        if "人気" in inquiry or "おすすめ" in inquiry:
            response = "現在人気の商品は、コーラ、ミネラルウォーター、お茶です。"
        elif "価格" in inquiry:
            response = "商品価格は自動販売機の画面で確認できます。税込価格です。"
        else:
            response = "お問い合わせありがとうございます。詳細をご連絡いたします。"

        return {
            "customer_id": customer_id,
            "inquiry": inquiry,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "status": "responded",
        }

    async def engage_customer(self, customer_id: str, device_id: str) -> Dict[str, Any]:
        """顧客エンゲージメントを開始"""
        logger.info(
            f"Starting customer engagement for {customer_id} on device {device_id}"
        )

        # 顧客データを初期化
        if customer_id not in self.customer_data:
            self.customer_data[customer_id] = {
                "visit_count": 0,
                "preferences": [],
                "last_visit": None,
            }

        self.customer_data[customer_id]["visit_count"] += 1
        self.customer_data[customer_id]["last_visit"] = datetime.now().isoformat()

        return {
            "customer_id": customer_id,
            "device_id": device_id,
            "engagement_started": True,
            "visit_count": self.customer_data[customer_id]["visit_count"],
            "welcome_message": f"こんにちは！{customer_id}さん、いらっしゃいませ。今日はどの商品をお探しですか？",
        }

    async def handle_customer_message(
        self, session_id: str, message: str
    ) -> Dict[str, Any]:
        """顧客メッセージを処理"""
        logger.info(f"Handling customer message for session {session_id}: {message}")

        # 簡易な応答生成
        response = "メッセージを受け取りました。ありがとうございます。"

        # 商品名が含まれていたら関連商品を提案
        if any(keyword in message for keyword in ["コーラ", "ジュース", "水", "珈琲"]):
            response += " 同じカテゴリの商品もおすすめです。"

        return {
            "session_id": session_id,
            "original_message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }

    async def generate_personalized_recommendations(
        self, customer_id: str
    ) -> List[Dict[str, Any]]:
        """パーソナライズされた商品レコメンデーションを生成"""
        logger.info(f"Generating recommendations for customer {customer_id}")

        # 顧客データを取得（存在しない場合はデフォルト値）
        customer_info = self.customer_data.get(
            customer_id, {"visit_count": 1, "preferences": ["drink"]}
        )

        recommendations = []

        # 訪問回数に基づいてレコメンデーション
        visit_count = customer_info.get("visit_count", 1)

        if visit_count >= 5:
            # リピーター顧客
            recommendations = [
                {
                    "product_id": "COLA_VARIANT",
                    "name": "新フレーバーコーラ",
                    "reason": "リピーターのお客様に新しい味をおすすめ",
                    "price": 160,
                    "personalized_score": 0.95,
                }
            ]
        else:
            # 新規/ライトユーザー
            recommendations = [
                {
                    "product_id": "COLA_ORIGINAL",
                    "name": "オリジナルコーラ",
                    "reason": "当店一番人気の定番商品",
                    "price": 150,
                    "personalized_score": 0.85,
                }
            ]

        return recommendations

    async def analyze_customer_satisfaction(self, customer_id: str) -> float:
        """顧客満足度を分析"""
        logger.info(f"Analyzing satisfaction for customer {customer_id}")

        # 顧客データを取得
        customer_info = self.customer_data.get(customer_id, {"visit_count": 1})

        visit_count = customer_info.get("visit_count", 1)

        # 訪問回数に基づく簡易満足度計算
        if visit_count >= 10:
            satisfaction = 4.5  # ロイヤルカスタマー
        elif visit_count >= 5:
            satisfaction = 4.0  # リピーター
        elif visit_count >= 2:
            satisfaction = 3.5  # 定期的利用
        else:
            satisfaction = 3.0  # 初回/新規

        return satisfaction


# グローバルインスタンス
customer_agent = CustomerAgent()
