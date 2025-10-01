import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ai.model_manager import AIMessage, ModelManager, model_manager
from src.config.settings import settings
from src.models.product import PRODUCT_CATEGORIES, Product
from src.services.conversation_service import ConversationService, conversation_service
from src.services.inventory_service import InventoryService, inventory_service

logger = logging.getLogger(__name__)


class CustomerAgent:
    """顧客対話エージェント"""

    def __init__(self):
        self.model_manager = model_manager
        self.conversation_service = conversation_service
        self.inventory_service = inventory_service

    async def engage_customer(
        self, customer_id: str, machine_id: str
    ) -> Dict[str, Any]:
        """顧客とのエンゲージメントを開始"""
        logger.info(f"顧客エンゲージメント開始: {customer_id}")

        try:
            # 過去の会話履歴を取得
            conversation_history = (
                await self.conversation_service.get_conversation_history(
                    customer_id, limit=5
                )
            )

            # 新しいセッションを作成
            session_id = await self.conversation_service.create_session(
                customer_id, machine_id
            )

            # AI会話生成
            ai_prompt = self._build_engagement_prompt(conversation_history, customer_id)

            messages = [
                AIMessage(
                    role="system",
                    content="あなたは親しみやすい自動販売機のアシスタントです。顧客のニーズを理解し、適切な商品を提案してください。",
                ),
                AIMessage(role="user", content=ai_prompt),
            ]

            ai_response = await self.model_manager.generate_response(
                messages, max_tokens=500
            )

            if ai_response.success:
                response_content = self._parse_ai_response(ai_response.content)
            else:
                response_content = {
                    "content": "こんにちは！何かお手伝いできることはありますか？",
                    "suggested_products": [],
                    "engagement_type": "greeting",
                }

            # 会話を記録
            await self.conversation_service.add_message(
                session_id,
                "assistant",
                response_content["content"],
                {"engagement_type": response_content.get("engagement_type", "general")},
            )

            return {
                "session_id": session_id,
                "message": response_content["content"],
                "suggested_products": response_content.get("suggested_products", []),
                "conversation_context": conversation_history,
                "success": True,
            }

        except Exception as e:
            logger.error(f"顧客エンゲージメントエラー: {e}")
            return {
                "session_id": None,
                "message": "申し訳ありませんが、現在サービスが利用できません。",
                "suggested_products": [],
                "success": False,
                "error": str(e),
            }

    def _build_engagement_prompt(
        self, conversation_history: List[Dict[str, Any]], customer_id: str
    ) -> str:
        """エンゲージメントプロンプトを構築"""
        history_summary = ""
        if conversation_history:
            history_summary = "過去の会話履歴:\n"
            for conv in conversation_history:
                history_summary += f"- {conv['start_time']}: {conv['summary']}\n"

        # 現在の在庫状況を取得
        inventory_summary = self.inventory_service.get_inventory_summary()
        available_products = (
            inventory_summary.total_slots - inventory_summary.out_of_stock_slots
        )

        return f"""
        顧客ID: {customer_id}

        {history_summary}

        現在の状況:
        - 利用可能な商品スロット: {available_products}個
        - 在庫切れスロット: {inventory_summary.out_of_stock_slots}個

        この顧客との自然で継続性のある会話を行ってください。
        以下の要素を考慮：
        1. 過去の購入パターンや嗜好（会話履歴から推測）
        2. 前回の会話内容との連続性
        3. 現在の在庫状況に基づく商品提案
        4. 顧客満足度の向上

        応答は以下のJSON形式で：
        {{
            "content": "会話メッセージ（日本語で親しみやすく）",
            "engagement_type": "greeting|product_recommendation|inquiry|farewell",
            "suggested_products": ["商品名1", "商品名2"],
            "insights": {{
                "customer_mood": "推定される顧客の気分（positive|neutral|negative）",
                "engagement_level": "エンゲージメントレベル（high|medium|low）",
                "potential_needs": ["ニーズ1", "ニーズ2"]
            }}
        }}
        """

    def _parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """AI応答をパース"""
        try:
            import json

            # JSON部分を抽出（簡易的な方法）
            json_start = ai_response.find("{")
            json_end = ai_response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                parsed = json.loads(json_str)

                return {
                    "content": parsed.get("content", ai_response),
                    "engagement_type": parsed.get("engagement_type", "general"),
                    "suggested_products": parsed.get("suggested_products", []),
                    "insights": parsed.get("insights", {}),
                }
            else:
                # JSONが見つからない場合はデフォルト応答
                return {
                    "content": ai_response,
                    "engagement_type": "general",
                    "suggested_products": [],
                    "insights": {},
                }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"AI応答のパースエラー: {e}")
            return {
                "content": ai_response,
                "engagement_type": "general",
                "suggested_products": [],
                "insights": {},
            }

    async def handle_customer_message(
        self, session_id: str, message: str
    ) -> Dict[str, Any]:
        """顧客メッセージを処理"""
        logger.info(f"顧客メッセージ処理: {session_id}")

        try:
            # 会話履歴を取得
            conversation_data = (
                await self.conversation_service.get_conversation_for_ai_agent(
                    session_id
                )
            )

            if not conversation_data:
                return {
                    "message": "申し訳ありませんが、会話セッションが見つかりません。",
                    "success": False,
                }

            # AIプロンプトを構築
            prompt = self._build_message_handling_prompt(message, conversation_data)

            # 会話履歴にユーザーメッセージを追加
            messages = [
                AIMessage(
                    role="system",
                    content="あなたは親しみやすい自動販売機のアシスタントです。顧客の質問に丁寧に答え、適切な商品を提案してください。",
                )
            ]

            # 過去の会話を含める
            for msg in conversation_data["message_history"]:
                messages.append(AIMessage(role=msg["role"], content=msg["content"]))

            messages.append(AIMessage(role="user", content=prompt))

            # AI応答を生成
            ai_response = await self.model_manager.generate_response(
                messages, max_tokens=500
            )

            if ai_response.success:
                response_content = self._parse_ai_response(ai_response.content)
            else:
                response_content = {
                    "content": "申し訳ありませんが、応答を生成できませんでした。",
                    "engagement_type": "error",
                }

            # 会話を記録
            await self.conversation_service.add_message(
                session_id, "user", message, {"type": "customer_inquiry"}
            )

            await self.conversation_service.add_message(
                session_id,
                "assistant",
                response_content["content"],
                {
                    "engagement_type": response_content.get(
                        "engagement_type", "response"
                    )
                },
            )

            return {
                "message": response_content["content"],
                "suggested_products": response_content.get("suggested_products", []),
                "insights": response_content.get("insights", {}),
                "success": True,
            }

        except Exception as e:
            logger.error(f"顧客メッセージ処理エラー: {e}")
            return {
                "message": "申し訳ありませんが、メッセージを処理できませんでした。",
                "success": False,
                "error": str(e),
            }

    def _build_message_handling_prompt(
        self, message: str, conversation_data: Dict[str, Any]
    ) -> str:
        """メッセージ処理プロンプトを構築"""
        context = conversation_data.get("customer_context", {})
        previous_insights = conversation_data.get("previous_insights", {})

        return f"""
        顧客メッセージ: {message}

        顧客コンテキスト: {context}
        過去の分析結果: {previous_insights}

        このメッセージに対して適切に応答してください。
        以下の点を考慮：
        1. 商品に関する質問には具体的な情報を提供
        2. 購入意欲を示すメッセージには積極的に提案
        3. 不満や問題には謝罪と解決策を提示
        4. 一般的な会話には親しみを持って対応

        応答は以下のJSON形式で：
        {{
            "content": "応答メッセージ（日本語で丁寧に）",
            "engagement_type": "product_info|purchase_intent|complaint|casual_talk",
            "suggested_products": ["関連商品1", "関連商品2"],
            "action_required": "在庫確認|決済支援|情報提供",
            "insights": {{
                "intent": "顧客の意図（information|purchase|complaint|casual）",
                "satisfaction": "満足度（high|medium|low）",
                "next_best_action": "次に推奨されるアクション"
            }}
        }}
        """

    async def generate_personalized_recommendations(
        self, customer_id: str
    ) -> List[str]:
        """パーソナライズされた商品推奨を生成"""
        try:
            # 会話履歴から嗜好を分析
            history = await self.conversation_service.get_conversation_history(
                customer_id, limit=10
            )

            # 嗜好キーワードを抽出
            preferences = self._analyze_customer_preferences(history)

            # 在庫状況を考慮した推奨
            recommendations = []

            # カテゴリ別の推奨
            for category in PRODUCT_CATEGORIES.values():
                if category.name in preferences:
                    # このカテゴリの商品を在庫から取得
                    available_products = self._get_available_products_by_category(
                        category.category
                    )
                    recommendations.extend(available_products[:2])  # 上位2つ

            return recommendations[:5]  # 最大5つまで

        except Exception as e:
            logger.error(f"パーソナライズ推奨生成エラー: {e}")
            return []

    def _analyze_customer_preferences(self, history: List[Dict[str, Any]]) -> List[str]:
        """顧客の嗜好を分析"""
        preferences = []

        for conv in history:
            summary = conv.get("summary", "")
            context = conv.get("context", {})

            # 会話内容からキーワードを抽出
            if "コーヒー" in summary or "コーヒー" in str(context):
                preferences.append("コーヒー")
            if "ジュース" in summary or "ジュース" in str(context):
                preferences.append("ジュース")
            if "スナック" in summary or "スナック" in str(context):
                preferences.append("スナック")
            if "健康" in summary or "健康" in str(context):
                preferences.append("健康飲料")

        return list(set(preferences))  # 重複を除去

    def _get_available_products_by_category(self, category: str) -> List[str]:
        """カテゴリの利用可能商品を取得"""
        # 簡易的な実装：在庫サービスから取得すべき
        category_products = {
            "drink": ["コカ・コーラ", "お茶", "コーヒー", "ジュース"],
            "snack": ["ポテトチップス", "チョコレート", "キャンディー"],
            "food": ["カップヌードル", "サンドイッチ"],
        }

        return category_products.get(category, [])

    async def analyze_customer_satisfaction(self, customer_id: str) -> Dict[str, Any]:
        """顧客満足度を分析"""
        try:
            history = await self.conversation_service.get_conversation_history(
                customer_id, limit=20
            )

            if not history:
                return {"satisfaction_score": 0.5, "analysis": "会話履歴なし"}

            # 簡易的な満足度分析
            positive_keywords = ["ありがとう", "満足", "美味しい", "良い", "助かる"]
            negative_keywords = ["不満", "悪い", "残念", "問題", "困る"]

            total_messages = sum(conv["message_count"] for conv in history)
            positive_count = 0
            negative_count = 0

            for conv in history:
                summary = conv.get("summary", "")
                for keyword in positive_keywords:
                    if keyword in summary:
                        positive_count += 1
                for keyword in negative_keywords:
                    if keyword in summary:
                        negative_count += 1

            # 満足度スコアを計算（0-1）
            if positive_count + negative_count > 0:
                satisfaction_score = positive_count / (positive_count + negative_count)
            else:
                satisfaction_score = 0.5  # デフォルト

            return {
                "satisfaction_score": satisfaction_score,
                "total_conversations": len(history),
                "total_messages": total_messages,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "analysis": f"満足度スコア: {satisfaction_score:.2f} ({positive_count}件の肯定的指標, {negative_count}件の否定的指標)",
            }

        except Exception as e:
            logger.error(f"顧客満足度分析エラー: {e}")
            return {"satisfaction_score": 0.5, "analysis": f"分析エラー: {str(e)}"}

    async def generate_engagement_report(self, customer_id: str) -> Dict[str, Any]:
        """顧客エンゲージメントレポートを生成"""
        try:
            # 会話履歴を取得
            history = await self.conversation_service.get_conversation_history(
                customer_id, limit=50
            )

            # 満足度分析
            satisfaction = await self.analyze_customer_satisfaction(customer_id)

            # 嗜好分析
            preferences = self._analyze_customer_preferences(history)

            # エンゲージメントレベルを計算
            engagement_level = self._calculate_engagement_level(history)

            return {
                "customer_id": customer_id,
                "total_conversations": len(history),
                "satisfaction_analysis": satisfaction,
                "preferences": preferences,
                "engagement_level": engagement_level,
                "recommendations": await self.generate_personalized_recommendations(
                    customer_id
                ),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"エンゲージメントレポート生成エラー: {e}")
            return {"error": str(e)}

    def _calculate_engagement_level(self, history: List[Dict[str, Any]]) -> str:
        """エンゲージメントレベルを計算"""
        if not history:
            return "none"

        # 会話頻度とメッセージ数で評価
        total_conversations = len(history)
        total_messages = sum(conv["message_count"] for conv in history)

        avg_messages_per_conversation = total_messages / total_conversations

        if total_conversations >= 10 and avg_messages_per_conversation >= 5:
            return "high"
        elif total_conversations >= 5 and avg_messages_per_conversation >= 3:
            return "medium"
        else:
            return "low"


# グローバルインスタンス
customer_agent = CustomerAgent()
