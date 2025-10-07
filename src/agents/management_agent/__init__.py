"""
management_agent package

このパッケージは経営管理エージェントとそのツール群を含みます。
"""

# State/Modelクラスを直接import (models.pyからimportすることで循環インポート回避)
# management_agentインスタンスを直接import
from .agent import management_agent
from .models import BusinessMetrics, ManagementState, SessionInfo
from .tools.tool_registry import create_tool_registry


# management_agentインスタンスも遅延インポート (循環インポート回避)
# 直接 absoluted import pathを使うことでモジュール解決問題を回避
def get_management_agent():
    import os
    import sys

    # srcディレクトリをパスに追加済みの前提で、絶対インポート
    from .agent import management_agent

    return management_agent


# 他のサブモジュールを明示的に公開（必要に応じて）
__all__ = [
    "BusinessMetrics",
    "ManagementState",
    "SessionInfo",
    "create_tool_registry",
    "get_management_agent",
    "management_agent",
]
