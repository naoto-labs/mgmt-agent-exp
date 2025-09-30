import asyncio
import logging
import signal
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from src.config.settings import settings, validate_startup_settings
from src.config.security import secure_config, setup_secure_logging
from src.ai.model_manager import model_manager
from src.services.payment_service import payment_service
from src.services.inventory_service import inventory_service
from src.services.conversation_service import conversation_service
from src.agents.search_agent import search_agent
from src.agents.customer_agent import customer_agent
from src.accounting.journal_entry import journal_processor
from src.accounting.management_accounting import management_analyzer
from src.analytics.event_tracker import event_tracker
from src.api.vending import router as vending_router
from src.api.tablet import router as tablet_router
from src.api.procurement import router as procurement_router

logger = logging.getLogger(__name__)

class SystemHealthStatus:
    """システム健全性ステータス"""

    def __init__(self):
        self.overall_status: str = "unknown"
        self.components: Dict[str, Dict[str, Any]] = {}
        self.last_check: Optional[datetime] = None
        self.issues: List[str] = []

    def is_healthy(self) -> bool:
        """システム全体が健全かチェック"""
        return self.overall_status == "healthy"

    def get_critical_issues(self) -> List[str]:
        """重大な問題を取得"""
        return [issue for issue in self.issues if "critical" in issue.lower()]

class SystemOrchestrator:
    """システムオーケストレーター"""

    def __init__(self):
        self.health_status = SystemHealthStatus()
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_requested = False

        # シャットダウンイベントの設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"シグナル {signum} を受信しました。シャットダウンを開始します。")
        self.shutdown_requested = True

    async def initialize_system(self) -> bool:
        """システムを初期化"""
        logger.info("システム初期化開始")

        try:
            # 設定検証
            if not validate_startup_settings():
                logger.error("設定検証に失敗しました")
                return False

            # セキュアログのセットアップ
            setup_secure_logging()

            # 各コンポーネントの初期化
            initialization_steps = [
                ("セキュリティ設定", self._initialize_security),
                ("AIモデルマネージャー", self._initialize_ai_models),
                ("データベース接続", self._initialize_databases),
                ("サービス層", self._initialize_services),
                ("エージェント層", self._initialize_agents),
                ("イベント追跡", self._initialize_event_tracking),
                ("APIルーター", self._initialize_api_routers),
            ]

            for step_name, step_func in initialization_steps:
                logger.info(f"初期化ステップ: {step_name}")
                if not await step_func():
                    logger.error(f"初期化ステップ失敗: {step_name}")
                    return False

            # システム健全性チェック
            await self._check_system_health()

            self.startup_time = datetime.now()
            self.is_running = True

            logger.info("システム初期化完了")
            return True

        except Exception as e:
            logger.error(f"システム初期化エラー: {e}")
            return False

    async def _initialize_security(self) -> bool:
        """セキュリティ設定を初期化"""
        try:
            # APIキー検証
            key_validation = secure_config.validate_api_keys()
            logger.info(f"APIキー検証結果: {key_validation}")

            # 暗号化機能のテスト
            test_data = "テストデータ"
            encrypted = secure_config.encrypt_data(test_data)
            decrypted = secure_config.decrypt_data(encrypted)

            if decrypted != test_data:
                raise ValueError("暗号化/復号化テスト失敗")

            logger.info("セキュリティ設定初期化完了")
            return True

        except Exception as e:
            logger.error(f"セキュリティ設定初期化エラー: {e}")
            return False

    async def _initialize_ai_models(self) -> bool:
        """AIモデルマネージャーを初期化"""
        try:
            # モデルマネージャーは既に初期化されているはず
            # ヘルスチェックを実行
            health_results = await model_manager.check_all_models_health()

            available_models = sum(1 for healthy in health_results.values() if healthy)
            total_models = len(health_results)

            logger.info(f"AIモデルヘルスチェック: {available_models}/{total_models} が利用可能")

            if available_models == 0:
                logger.warning("利用可能なAIモデルがありません")

            return True

        except Exception as e:
            logger.error(f"AIモデル初期化エラー: {e}")
            return False

    async def _initialize_databases(self) -> bool:
        """データベース接続を初期化"""
        try:
            # 会話サービスがストレージを初期化
            if hasattr(conversation_service, 'load_sessions_from_storage'):
                conversation_service.load_sessions_from_storage()

            # イベント追跡システムがストレージを準備
            if hasattr(event_tracker, '_ensure_storage_dir'):
                event_tracker._ensure_storage_dir()

            logger.info("データベース接続初期化完了")
            return True

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            return False

    async def _initialize_services(self) -> bool:
        """サービス層を初期化"""
        try:
            # 各サービスの初期化状態を確認
            services = [
                ("決済サービス", payment_service),
                ("在庫サービス", inventory_service),
                ("会話サービス", conversation_service),
            ]

            for service_name, service in services:
                # 各サービスが適切に初期化されているかチェック
                if hasattr(service, 'get_payment_stats'):  # 決済サービスの場合
                    stats = service.get_payment_stats()
                    logger.info(f"{service_name} 統計: {stats}")
                elif hasattr(service, 'get_inventory_summary'):  # 在庫サービスの場合
                    summary = service.get_inventory_summary()
                    logger.info(f"{service_name} サマリ: {summary.total_slots}スロット")
                elif hasattr(service, 'get_conversation_stats'):  # 会話サービスの場合
                    stats = service.get_conversation_stats()
                    logger.info(f"{service_name} 統計: {stats}")

            logger.info("サービス層初期化完了")
            return True

        except Exception as e:
            logger.error(f"サービス層初期化エラー: {e}")
            return False

    async def _initialize_agents(self) -> bool:
        """エージェント層を初期化"""
        try:
            # エージェントの初期化状態を確認
            agents = [
                ("検索エージェント", search_agent),
                ("顧客エージェント", customer_agent),
            ]

            for agent_name, agent in agents:
                # 各エージェントが適切に初期化されているかチェック
                if hasattr(agent, 'get_search_stats'):  # 検索エージェントの場合
                    stats = agent.get_search_stats()
                    logger.info(f"{agent_name} 統計: {stats}")
                elif hasattr(agent, 'model_manager'):  # 顧客エージェントの場合
                    logger.info(f"{agent_name} AIモデル連携確認済み")

            logger.info("エージェント層初期化完了")
            return True

        except Exception as e:
            logger.error(f"エージェント層初期化エラー: {e}")
            return False

    async def _initialize_event_tracking(self) -> bool:
        """イベント追跡システムを初期化"""
        try:
            # イベント追跡システムの状態を確認
            stats = event_tracker.get_event_stats()
            logger.info(f"イベント追跡統計: {stats}")

            # システム起動イベントを記録
            event_tracker.track_event(
                EventType.SYSTEM_STARTUP,
                "orchestrator",
                "システム起動完了",
                EventSeverity.LOW,
                {"startup_time": datetime.now().isoformat()}
            )

            logger.info("イベント追跡システム初期化完了")
            return True

        except Exception as e:
            logger.error(f"イベント追跡初期化エラー: {e}")
            return False

    async def _initialize_api_routers(self) -> bool:
        """APIルーターを初期化"""
        try:
            # APIルーターが適切に設定されているかチェック
            routers = [
                ("販売API", vending_router),
                ("タブレットAPI", tablet_router),
                ("調達API", procurement_router),
            ]

            for router_name, router in routers:
                if hasattr(router, 'routes'):
                    route_count = len(router.routes)
                    logger.info(f"{router_name}: {route_count}エンドポイント")

            logger.info("APIルーター初期化完了")
            return True

        except Exception as e:
            logger.error(f"APIルーター初期化エラー: {e}")
            return False

    async def check_system_health(self) -> SystemHealthStatus:
        """システム健全性をチェック"""
        logger.info("システム健全性チェック開始")

        self.health_status = SystemHealthStatus()
        self.health_status.last_check = datetime.now()
        self.health_status.issues = []

        try:
            # 各コンポーネントの健全性をチェック
            health_checks = [
                ("AIモデル", self._check_ai_models_health),
                ("サービス層", self._check_services_health),
                ("データベース", self._check_databases_health),
                ("イベント追跡", self._check_event_tracking_health),
            ]

            component_results = {}

            for component_name, check_func in health_checks:
                try:
                    result = await check_func()
                    component_results[component_name] = result
                except Exception as e:
                    logger.error(f"{component_name}健全性チェックエラー: {e}")
                    component_results[component_name] = {
                        "status": "error",
                        "error": str(e)
                    }

            # 全体ステータスの決定
            self.health_status.components = component_results

            # 重大な問題をチェック
            critical_issues = []
            for component, result in component_results.items():
                if result.get("status") in ["error", "critical"]:
                    critical_issues.append(f"{component}: {result.get('error', '不明なエラー')}")

            if critical_issues:
                self.health_status.overall_status = "critical"
                self.health_status.issues = critical_issues
            elif any(result.get("status") == "warning" for result in component_results.values()):
                self.health_status.overall_status = "warning"
            else:
                self.health_status.overall_status = "healthy"

            logger.info(f"システム健全性チェック完了: {self.health_status.overall_status}")
            return self.health_status

        except Exception as e:
            logger.error(f"システム健全性チェックエラー: {e}")
            self.health_status.overall_status = "error"
            self.health_status.issues = [str(e)]
            return self.health_status

    async def _check_ai_models_health(self) -> Dict[str, Any]:
        """AIモデルの健全性をチェック"""
        try:
            health_results = await model_manager.check_all_models_health()
            stats = model_manager.get_model_stats()

            available_models = sum(1 for healthy in health_results.values() if healthy)

            if available_models == 0:
                return {"status": "critical", "error": "利用可能なAIモデルがありません"}

            return {
                "status": "healthy",
                "available_models": available_models,
                "total_models": len(health_results),
                "primary_model": stats.get("primary_model")
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_services_health(self) -> Dict[str, Any]:
        """サービス層の健全性をチェック"""
        try:
            issues = []

            # 決済サービスチェック
            try:
                payment_stats = payment_service.get_payment_stats()
                if payment_stats.get("success_rate", 1.0) < 0.8:
                    issues.append("決済成功率が低い")
            except Exception as e:
                issues.append(f"決済サービスエラー: {e}")

            # 在庫サービスチェック
            try:
                inventory_summary = inventory_service.get_inventory_summary()
                if inventory_summary.out_of_stock_slots > inventory_summary.total_slots * 0.5:
                    issues.append("在庫切れ商品が多すぎる")
            except Exception as e:
                issues.append(f"在庫サービスエラー: {e}")

            if issues:
                return {"status": "warning", "issues": issues}

            return {"status": "healthy"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_databases_health(self) -> Dict[str, Any]:
        """データベースの健全性をチェック"""
        try:
            # 会話サービスチェック
            try:
                conversation_stats = conversation_service.get_conversation_stats()
            except Exception as e:
                return {"status": "warning", "error": f"会話データベースエラー: {e}"}

            # イベント追跡チェック
            try:
                event_stats = event_tracker.get_event_stats()
            except Exception as e:
                return {"status": "warning", "error": f"イベント追跡エラー: {e}"}

            return {"status": "healthy", "details": {"conversations": conversation_stats, "events": event_stats}}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_event_tracking_health(self) -> Dict[str, Any]:
        """イベント追跡の健全性をチェック"""
        try:
            stats = event_tracker.get_event_stats()
            health_score = event_tracker.get_system_health_score()

            if health_score < 0.8:
                return {"status": "warning", "message": f"システム健全性スコアが低い: {health_score}"}

            return {"status": "healthy", "health_score": health_score}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def run_system_monitor(self, interval_seconds: int = 60):
        """システム監視を実行"""
        logger.info(f"システム監視を開始（間隔: {interval_seconds}秒）")

        while self.is_running and not self.shutdown_requested:
            try:
                # システム健全性チェック
                health_status = await self.check_system_health()

                # 重大な問題がある場合はログに記録
                if health_status.get_critical_issues():
                    logger.critical(f"重大なシステム問題検出: {health_status.get_critical_issues()}")

                # 監視間隔待機
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("システム監視がキャンセルされました")
                break
            except Exception as e:
                logger.error(f"システム監視エラー: {e}")
                await asyncio.sleep(interval_seconds)

    async def shutdown_system(self) -> bool:
        """システムをシャットダウン"""
        logger.info("システムシャットダウン開始")

        try:
            self.is_running = False

            # シャットダウンイベントを記録
            event_tracker.track_event(
                EventType.SYSTEM_SHUTDOWN,
                "orchestrator",
                "システムシャットダウン開始",
                EventSeverity.LOW,
                {"shutdown_time": datetime.now().isoformat()}
            )

            # 各コンポーネントのクリーンアップ
            cleanup_steps = [
                ("イベント追跡", self._cleanup_event_tracking),
                ("サービス層", self._cleanup_services),
                ("データベース接続", self._cleanup_databases),
                ("AIモデル", self._cleanup_ai_models),
            ]

            for step_name, step_func in cleanup_steps:
                logger.info(f"クリーンアップステップ: {step_name}")
                try:
                    await step_func()
                except Exception as e:
                    logger.warning(f"クリーンアップステップ失敗: {step_name} - {e}")

            # 最終的なシステム状態を記録
            uptime = datetime.now() - self.startup_time if self.startup_time else timedelta(0)
            logger.info(f"システムシャットダウン完了（稼働時間: {uptime})")

            return True

        except Exception as e:
            logger.error(f"システムシャットダウンエラー: {e}")
            return False

    async def _cleanup_event_tracking(self):
        """イベント追跡システムをクリーンアップ"""
        try:
            # 古いイベントを削除
            removed_count = event_tracker.clear_old_events(days_to_keep=7)
            logger.info(f"古いイベントを削除: {removed_count}件")
        except Exception as e:
            logger.error(f"イベント追跡クリーンアップエラー: {e}")

    async def _cleanup_services(self):
        """サービス層をクリーンアップ"""
        try:
            # 各サービスのクリーンアップ処理（必要に応じて実装）
            pass
        except Exception as e:
            logger.error(f"サービス層クリーンアップエラー: {e}")

    async def _cleanup_databases(self):
        """データベース接続をクリーンアップ"""
        try:
            # 各サービスのクリーンアップ処理（必要に応じて実装）
            pass
        except Exception as e:
            logger.error(f"データベースクリーンアップエラー: {e}")

    async def _cleanup_ai_models(self):
        """AIモデルをクリーンアップ"""
        try:
            # AIモデルのクリーンアップ処理（必要に応じて実装）
            pass
        except Exception as e:
            logger.error(f"AIモデルクリーンアップエラー: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        uptime = datetime.now() - self.startup_time if self.startup_time else timedelta(0)

        return {
            "is_running": self.is_running,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": uptime.total_seconds(),
            "shutdown_requested": self.shutdown_requested,
            "health_status": {
                "overall": self.health_status.overall_status,
                "last_check": self.health_status.last_check.isoformat() if self.health_status.last_check else None,
                "issues": self.health_status.issues
            }
        }

    async def run_diagnostics(self) -> Dict[str, Any]:
        """システム診断を実行"""
        logger.info("システム診断開始")

        try:
            diagnostics = {
                "timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_status(),
                "component_health": self.health_status.components,
                "performance_metrics": await self._get_performance_metrics(),
                "recommendations": []
            }

            # 診断に基づく推奨事項を生成
            if not self.health_status.is_healthy():
                diagnostics["recommendations"].append("システム健全性に問題があります。詳細なログを確認してください。")

            if self.health_status.get_critical_issues():
                diagnostics["recommendations"].append("重大な問題が検出されました。即時対応が必要です。")

            logger.info("システム診断完了")
            return diagnostics

        except Exception as e:
            logger.error(f"システム診断エラー: {e}")
            return {"error": str(e)}

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標を取得"""
        try:
            return {
                "event_count_24h": len(event_tracker.get_recent_events(24)),
                "conversation_count": len(conversation_service.sessions),
                "inventory_slots": len(inventory_service.vending_machine_slots) + len(inventory_service.storage_slots),
                "journal_entries": len(journal_processor.journal_entries),
                "search_history": len(search_agent.search_history)
            }

        except Exception as e:
            logger.error(f"パフォーマンス指標取得エラー: {e}")
            return {}

# グローバルインスタンス
orchestrator = SystemOrchestrator()

# 便利な関数
async def initialize_and_run():
    """システムを初期化して実行"""
    if await orchestrator.initialize_system():
        logger.info("システムが正常に起動しました")

        # システム監視を開始
        monitor_task = asyncio.create_task(orchestrator.run_system_monitor())

        try:
            # メイン処理（ここでは待機）
            while orchestrator.is_running and not orchestrator.shutdown_requested:
                await asyncio.sleep(1)

        finally:
            # 監視タスクをキャンセル
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # システムシャットダウン
            await orchestrator.shutdown_system()
    else:
        logger.error("システム起動に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    # システム起動
    asyncio.run(initialize_and_run())
