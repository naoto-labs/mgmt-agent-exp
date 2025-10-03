import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
import asyncio


async def test_case_a():
    # Import through agents package (analytics disabled)
    from agents.management_agent import management_agent

    print("=== Case A Node-Based Management Agent Test ===")
    print(f"Agent provider: {management_agent.provider}")
    print(f"Number of nodes: {len(management_agent.nodes)}")
    print(f"Node names: {list(management_agent.nodes.keys())}")

    # Test inventory check node
    print("\n--- Testing inventory_check_node ---")
    result = await management_agent.inventory_check_node({})
    print(f"Node: {result['node']}")
    print(f"Status: {result['status']}")
    print(f"Metrics sales: {result.get('metrics', {}).get('sales', 'N/A')}")

    # Test pricing node
    print("\n--- Testing pricing_node ---")
    pricing_result = await management_agent.pricing_node(result)
    print(f"Node: {pricing_result['node']}")
    print(f"Action: {pricing_result.get('action', 'N/A')}")
    print(f"Status: {pricing_result['status']}")

    # Test feedback node
    print("\n--- Testing feedback_node ---")
    feedback_result = await management_agent.feedback_node(result)
    print(f"Node: {feedback_result['node']}")
    print(
        f"Average rating: {feedback_result.get('feedback', {}).get('average_rating', 'N/A')}"
    )
    print(f"Status: {feedback_result['status']}")

    print("\n=== Case A Node Implementation Successful ===")


if __name__ == "__main__":
    asyncio.run(test_case_a())
