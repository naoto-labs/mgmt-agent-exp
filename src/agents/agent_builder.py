"""
Agent Builder - 依存関係注入による安全なAgent構築

循環インポートを避けつつ、必要な依存関係を解決してAgentを構築します。
ModelManager自体は変更せず、注入パターンでクリーンに扱います。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Agentビルダー - 依存関係注入による安全なAgent構築

    このビルダーは以下の問題を解決します：
    1. 循環インポート (Management Agent → Settings → Infrastructure → Model Manager)
    2. 初期化順序の依存関係
    3. Model Manager - Agent間の疎結合

    使用法:
        agent = await AgentBuilder.build_management_agent()
    """

    @staticmethod
    async def build_management_agent() -> "NodeBasedManagementAgent":
        """
        Management Agentを構築 (依存関係解決済)

        Returns:
            完全な依存関係で初期化されたManagement Agent
        """
        logger.info("Agent Builder: 开始构建Management Agent...")

        # Phase 1: 依存関係の安全な準備 (循環インポート回避)
        llm_manager = await AgentBuilder._prepare_llm_manager()
        config = AgentBuilder._prepare_config()

        # Phase 2: Agent構築 with 依存注入
        from .management_agent import NodeBasedManagementAgent

        agent = NodeBasedManagementAgent(
            llm_manager=llm_manager,
            agent_objectives=config["objectives"],
            provider=config["provider"],
        )

        logger.info("Agent Builder: Management Agent構築完了")

        # Phase 3: 追加初期化 (必要に応じて)
        await AgentBuilder._post_initialize(agent)

        return agent

    @staticmethod
    async def _prepare_llm_manager() -> Optional[object]:
        """
        LLM Manager を安全に準備

        Model Managerは外部projectの仕様変更を避けるため、
        完全に既存APIのみで動作させます。
        """
        try:
            logger.debug("Agent Builder: LLM Manager準備中...")

            # 安全なインポート - importレベルでの循環を避ける
            from src.infrastructure.ai import model_manager

            # Model Managerの健康状態を確認 (オプション)
            if hasattr(model_manager, "check_all_models_health"):
                health_check = await model_manager.check_all_models_health()
                healthy_count = sum(1 for healthy in health_check.values() if healthy)
                logger.info(
                    f"Agent Builder: {healthy_count}/{len(health_check)} LLMモデルが利用可能"
                )
            else:
                logger.warning(
                    "Agent Builder: モデルヘルスチェック非対応のModel Manager"
                )

            # 設定が正しく読み込まれているか確認
            if hasattr(model_manager, "primary_model"):
                primary = model_manager.primary_model
                logger.info(f"Agent Builder: Primary Model: {primary}")
            else:
                logger.warning("Agent Builder: Primary model情報なし")

            return model_manager

        except Exception as e:
            logger.error(f"Agent Builder: LLM Manager準備エラー: {e}")
            # エラーハンドリング - Noneを返してAgentで適切に処理させる
            return None

    @staticmethod
    def _prepare_config() -> dict:
        """
        Agent設定を準備

        Settingsから必要な設定のみを抽出して依存関係を最小化
        """
        try:
            logger.debug("Agent Builder: Agent設定準備中...")

            from src.shared.config.settings import settings

            config = {
                "objectives": getattr(settings, "agent_objectives", {}),
                "provider": getattr(settings, "llm_provider", "openai"),
            }

            logger.debug(
                f"Agent Builder: 設定準備完了 - provider: {config['provider']}"
            )
            return config

        except Exception as e:
            logger.error(f"Agent Builder: 設定準備エラー: {e}")
            # フォールバック設定
            return {"objectives": {}, "provider": "openai"}

    @staticmethod
    async def _post_initialize(agent: "NodeBasedManagementAgent") -> None:
        """
        Agentの後初期化処理

        必要に応じてStateGraphのコンパイル等を行う
        """
        try:
            logger.debug("Agent Builder: Agent後初期化処理...")

            # StateGraphが正しく構築されているか確認
            if hasattr(agent, "state_graph") and agent.state_graph is None:
                logger.info("Agent Builder: StateGraph再構築を試行")
                # 必要に応じて再初期化処理を追加可能

            # 他のpost-initialization処理をここに追加可能

            logger.debug("Agent Builder: 後初期化処理完了")

        except Exception as e:
            logger.error(f"Agent Builder: 後初期化処理エラー: {e}")
            # 後初期化失敗はAgent構築自体を失敗させない
