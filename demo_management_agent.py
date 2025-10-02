"""
Management Agentのデモスクリプト

SessionBasedManagementAgentとRecorderAgentの基本動作を検証
"""

import asyncio
import logging
from datetime import datetime

from src.agents.management_agent import SessionBasedManagementAgent
from src.agents.recorder_agent import (
    BusinessOutcomeRecord,
    ManagementActionRecord,
    RecorderAgent,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_management_agent_basic():
    """Management Agentの基本機能デモ"""
    print("\n" + "=" * 60)
    print("Management Agent 基本機能デモ")
    print("=" * 60 + "\n")

    # AIモデルの接続確認
    print("1. AIモデルの接続を確認...")
    from src.ai import model_manager

    health_results = await model_manager.check_all_models_health()
    available_models = [name for name, healthy in health_results.items() if healthy]
    print(f"   ✅ 利用可能モデル: {', '.join(available_models)}")

    # Agentの初期化
    print("2. Management Agentを初期化...")
    agent = SessionBasedManagementAgent(provider="openai")
    print(f"   ✓ 初期化完了: {len(agent.tools)}個のツールを利用可能\n")

    # ビジネスメトリクス取得
    print("2. ビジネスメトリクスを取得...")
    metrics = agent.get_business_metrics()
    print(f"   ✓ メトリクス取得完了")
    print(f"   📊 売上: ¥{metrics['sales']:,}")
    print(f"   💰 利益率: {metrics['profit_margin']:.1%}")
    print(f"   📦 在庫レベル: {metrics['inventory_level']}")
    print(f"   😊 顧客満足度: {metrics['customer_satisfaction']}/5.0")
    print(f"   🕐 取得時刻: {metrics['timestamp']}\n")

    # 財務分析
    print("3. 財務パフォーマンスを分析...")
    analysis = await agent.analyze_financial_performance()
    print(f"   ✓ 分析完了")
    print(f"   📈 分析結果: {analysis['analysis']}")
    print(f"   💡 推奨事項:")
    for i, rec in enumerate(analysis["recommendations"], 1):
        print(f"      {i}. {rec}")
    print()

    # 在庫状況確認
    print("4. 在庫状況を確認...")
    inventory = await agent.check_inventory_status()
    print(f"   ✓ 在庫確認完了")
    print(f"   📊 在庫ステータス: {inventory['status']}")
    print(f"   ⚠️  低在庫商品: {inventory['low_stock_items']}")
    print(f"   🔄 再注文必要: {inventory['reorder_needed']}")
    if inventory.get("estimated_stockout"):
        print(f"   ⏰ 在庫切れ予測: {inventory['estimated_stockout']}")
    print()


async def demo_session_flow():
    """セッション型実行のデモ"""
    print("\n" + "=" * 60)
    print("セッション型実行デモ")
    print("=" * 60 + "\n")

    agent = SessionBasedManagementAgent(provider="openai")

    # セッション開始
    print("1. 朝のルーチンセッションを開始...")
    session_id = await agent.start_management_session("morning_routine")
    print(f"   ✓ セッション開始: {session_id}\n")

    # ビジネスデータ取得
    print("2. ビジネスデータを取得...")
    metrics = agent.get_business_metrics()
    print(f"   ✓ データ取得完了\n")

    # 戦略的意思決定
    print("3. 戦略的意思決定を実行...")
    context = f"""
    現在の状況:
    - 売上: ¥{metrics["sales"]:,}
    - 在庫: {metrics["inventory_level"]}
    
    本日の重点課題を決定してください。
    """
    decision = await agent.make_strategic_decision(context)
    print(f"   決定: {decision['decision']}")
    print(f"   根拠: {decision['rationale']}")
    print(f"   アクション: {', '.join(decision['actions'])}\n")

    # セッション終了
    print("4. セッションを終了...")
    summary = await agent.end_management_session()
    print(f"   ✓ セッション終了")
    print(f"   セッションID: {summary['session_id']}")
    print(f"   セッションタイプ: {summary['session_type']}")
    print(f"   期間: {summary['duration']}")
    print(f"   意思決定数: {summary['decisions_count']}\n")


async def demo_daily_routine():
    """一日の業務フローデモ"""
    print("\n" + "=" * 60)
    print("一日の業務フローデモ")
    print("=" * 60 + "\n")

    agent = SessionBasedManagementAgent(provider="openai")

    # 朝のルーチン
    print("【朝のルーチン (9:00)】")
    morning_result = await agent.morning_routine()
    print(f"✓ 完了: {morning_result['session_type']}")
    print(f"  決定事項: {morning_result['decisions']['decision']}\n")

    # 昼のチェック
    print("【昼のチェック (12:00)】")
    midday_result = await agent.midday_check()
    print(f"✓ 完了: {midday_result['session_type']}")
    print(f"  決定事項: {midday_result['decisions']['decision']}\n")

    # 夕方の総括
    print("【夕方の総括 (17:00)】")
    evening_result = await agent.evening_summary()
    print(f"✓ 完了: {evening_result['session_type']}")
    print(f"  教訓: {', '.join(evening_result['lessons_learned'])}\n")


async def demo_human_collaboration():
    """人間協働機能のデモ"""
    print("\n" + "=" * 60)
    print("人間協働機能デモ")
    print("=" * 60 + "\n")

    agent = SessionBasedManagementAgent(provider="oepnai")

    # 補充タスク
    print("1. 補充タスクを割り当て...")
    task = agent.assign_restocking_task(["water", "juice"], urgency="urgent")
    print(f"   タスクID: {task['task_id']}")
    print(f"   対象商品: {', '.join(task['products'])}")
    print(f"   緊急度: {task['urgency']}")
    print(f"   期限: {task['deadline']}\n")

    # 調達依頼
    print("2. 調達を依頼...")
    order = agent.request_procurement(["cola", "water"], {"cola": 100, "water": 150})
    print(f"   発注ID: {order['order_id']}")
    print(f"   商品: {', '.join(order['products'])}")
    print(f"   数量: {order['quantity']}")
    print(f"   配送予定: {order['estimated_delivery']}\n")

    # 従業員タスク調整
    print("3. 従業員タスクを調整...")
    coordination = agent.coordinate_employee_tasks()
    print(f"   アクティブタスク: {coordination['active_tasks']}件")
    print(f"   本日完了: {coordination['completed_today']}件")
    print(f"   担当者: {coordination['employees']}\n")


async def demo_customer_service():
    """顧客対応機能のデモ"""
    print("\n" + "=" * 60)
    print("顧客対応機能デモ")
    print("=" * 60 + "\n")

    agent = SessionBasedManagementAgent(provider="oepnai")

    # 顧客問い合わせ
    print("1. 顧客問い合わせに対応...")
    inquiry = agent.respond_to_customer_inquiry(
        "C001", "商品の賞味期限について教えてください"
    )
    print(f"   顧客ID: {inquiry['customer_id']}")
    print(f"   問い合わせ: {inquiry['inquiry']}")
    print(f"   回答: {inquiry['response']}\n")

    # 顧客苦情処理
    print("2. 顧客苦情を処理...")
    complaint = agent.handle_customer_complaint("C002", "商品が出てこなかった")
    print(f"   顧客ID: {complaint['customer_id']}")
    print(f"   苦情: {complaint['complaint']}")
    print(f"   解決策: {complaint['resolution']}")
    print(f"   補償: {complaint['compensation']}\n")

    # フィードバック収集
    print("3. 顧客フィードバックを収集...")
    feedback = agent.collect_customer_feedback()
    print(f"   フィードバック数: {feedback['feedback_count']}件")
    print(f"   平均評価: {feedback['average_rating']}/5.0")
    print(f"   主な要望: {', '.join(feedback['top_requests'])}")
    print(f"   トレンド: {feedback['trends']}\n")


async def demo_recorder_agent():
    """Recorder Agentのデモ"""
    print("\n" + "=" * 60)
    print("Recorder Agent デモ")
    print("=" * 60 + "\n")

    recorder = RecorderAgent(persist_directory="./data/demo_vector_store")

    # 行動記録
    print("1. 行動を記録...")
    action_record = ManagementActionRecord(
        record_id="demo_action_001",
        session_id="demo_session_001",
        timestamp=datetime.now(),
        action_type="decision",
        context={"sales": 150000, "inventory_low": True},
        decision_process="在庫が少ないため、緊急補充を決定",
        executed_action="従業員に緊急補充タスクを割り当て",
        expected_outcome="2時間以内に在庫補充完了",
    )

    result = await recorder.record_action(action_record)
    if result["success"]:
        storage_type = result.get("storage", "unknown")
        print(f"   ✓ 行動記録完了: {result['record_id']}")
        print(f"   📝 ストレージ: {storage_type}\n")
    else:
        print(f"   ✗ 記録失敗: {result.get('error', result.get('reason', 'N/A'))}\n")

    # 結果記録
    print("2. 結果を記録...")
    outcome_record = BusinessOutcomeRecord(
        record_id="demo_outcome_001",
        session_id="demo_session_001",
        related_action_id="demo_action_001",
        timestamp=datetime.now(),
        outcome_type="inventory_management",
        metrics={"completion_time": 1.5, "inventory_restored": 100},
        success_level="excellent",
        lessons_learned=["迅速な対応が功を奏した", "事前の在庫監視が重要"],
    )

    result = await recorder.record_outcome(outcome_record)
    if result["success"]:
        storage_type = result.get("storage", "unknown")
        print(f"   ✓ 結果記録完了: {result['record_id']}")
        print(f"   📝 ストレージ: {storage_type}\n")
    else:
        print(f"   ✗ 記録失敗: {result.get('error', result.get('reason', 'N/A'))}\n")


async def main():
    """メインデモ実行"""
    print("\n" + "=" * 60)
    print("LangChain Management Agent デモ")
    print("=" * 60)

    try:
        # 基本機能デモ
        await demo_management_agent_basic()

        # セッションフローデモ
        await demo_session_flow()

        # 一日の業務フローデモ
        await demo_daily_routine()

        # 人間協働機能デモ
        await demo_human_collaboration()

        # 顧客対応機能デモ
        await demo_customer_service()

        # Recorder Agentデモ
        await demo_recorder_agent()

        print("\n" + "=" * 60)
        print("✓ 全てのデモが正常に完了しました")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生しました: {e}", exc_info=True)
        print(f"\n✗ エラー: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
