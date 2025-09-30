#!/usr/bin/env python3
"""
販売シミュレーションモデルテストスクリプト
"""

from src.services.payment_service import payment_service

def test_sales_simulation():
    print("🧪 販売シミュレーションモデルテスト開始...")

    # 現実的な販売シミュレーション
    sale_result = payment_service.simulate_realistic_sale('drink_cola', quantity=2)
    print("✅ 販売シミュレーション結果:")
    print(f"   商品ID: {sale_result['product_id']}")
    print(f"   予測需要: {sale_result['predicted_demand']".2f"}")
    print(f"   実際数量: {sale_result['actual_quantity']}")
    print(f"   単価: ¥{sale_result['unit_price']}")
    print(f"   総額: ¥{sale_result['total_amount']}")
    print(f"   顧客満足度: {sale_result['customer_satisfaction']".2f"}")

    # 需要予測
    forecast = payment_service.get_demand_forecast('drink_cola', days=2)
    print("
📊 需要予測結果:"    print(f"   予測期間: {forecast['forecast_period_days']}日間")
    print(f"   総予測需要: {forecast['summary']['total_predicted_demand']".2f"}")
    print(f"   ピーク時間帯: {forecast['summary']['peak_demand_hour']}時")

    # 市場シナリオシミュレーション
    scenario = payment_service.simulate_market_scenario('economic_boom')
    print("
📈 市場シナリオ結果:"    print(f"   シナリオ: {scenario['scenario']}")
    print(f"   推奨事項数: {len(scenario['recommendations'])}")

    print("\n🎉 販売シミュレーションモデルテスト完了！")

if __name__ == "__main__":
    test_sales_simulation()
