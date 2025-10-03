import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from store_simulation_scenarios import scenario_runner

from src.infrastructure import (
    procurement_router,
    tablet_router,
    vending_router,
)
from src.shared import (
    secure_config,
    settings,
    setup_secure_logging,
    validate_startup_settings,
)

# ロガーのセットアップ
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# セキュアログのセットアップ
setup_secure_logging()

# グローバル変数（後で適切な依存注入に変更）
vending_simulator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    logger.info(f"Starting {settings.app_name} (Machine ID: {settings.machine_id})")

    # 設定検証（オプション）
    try:
        settings_valid = validate_startup_settings()
        if not settings_valid:
            logger.warning("設定検証で問題が発生しましたが、続行します。")
    except Exception as e:
        logger.warning(f"設定検証エラーですが、続行します: {e}")

    # APIキー検証ログ（マスキング済み）
    key_validation = secure_config.validate_api_keys()
    logger.info(f"APIキー検証結果: {key_validation}")

    # シミュレーター初期化（後で実装）
    # await initialize_simulator()

    logger.info("アプリケーション起動完了")
    yield

    # 終了時の処理
    logger.info("アプリケーション終了処理開始")
    # await cleanup_simulator()
    logger.info("アプリケーション終了完了")


# FastAPIアプリケーションの作成
app = FastAPI(
    title=settings.app_name,
    description="Anthropic PJ Vend Simulator for research and development",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 静的ファイルのマウント
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# APIルーターの登録
app.include_router(
    vending_router,
    prefix="/api/v1/vending",
    tags=["vending"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    tablet_router,
    prefix="/api/v1/tablet",
    tags=["tablet"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    procurement_router,
    prefix="/api/v1/procurement",
    tags=["procurement"],
    responses={404: {"description": "Not found"}},
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],  # 開発環境用
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# セキュリティヘッダー設定
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # セキュリティヘッダーの追加
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


# ルートエンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": settings.app_name,
        "status": "operational",
        "version": "1.0.0",
        "machine_id": settings.machine_id,
        "debug_mode": settings.debug,
    }


# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        # 基本的なシステムチェック
        health_status = {
            "status": "healthy",
            "machine_id": settings.machine_id,
            "timestamp": None,  # 後で適切なタイムスタンプを追加
            "services": {
                "api": "operational",
                "database": "pending",  # データベース接続後に更新
                "ai_models": "pending",  # AIモデルチェック後に更新
            },
        }

        # データベースチェック（後で実装）
        # health_status["services"]["database"] = await check_database()

        # AIモデルチェック（後で実装）
        # health_status["services"]["ai_models"] = await check_ai_models()

        return JSONResponse(status_code=200, content=health_status)

    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        raise HTTPException(status_code=503, detail="ヘルスチェックに失敗しました")


# 設定情報エンドポイント（デバッグ用）
@app.get("/api/v1/config", include_in_schema=not settings.debug)
async def get_config():
    """設定情報取得（デバッグモードのみ公開）"""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="エンドポイントが見つかりません")

    return {
        "app_name": settings.app_name,
        "machine_id": settings.machine_id,
        "debug": settings.debug,
        "ai_safety_threshold": settings.ai_safety_threshold,
        "enable_guardrails": settings.enable_guardrails,
        "allowed_actions": settings.allowed_actions,
        "forbidden_patterns": settings.forbidden_patterns,
    }


# シナリオ実行エンドポイント
@app.get("/api/v1/scenarios")
async def get_scenarios():
    """利用可能なシナリオ一覧を取得"""
    try:
        scenarios = scenario_runner.get_scenario_info()
        return {
            "scenarios": scenarios,
            "total_count": len(scenarios),
        }
    except Exception as e:
        logger.error(f"シナリオ一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail="シナリオ一覧の取得に失敗しました")


@app.post("/api/v1/scenarios/{scenario_name}/run")
async def run_scenario(scenario_name: str):
    """シナリオを実行"""
    try:
        result = await scenario_runner.run_scenario(scenario_name)
        return result
    except Exception as e:
        logger.error(f"シナリオ実行エラー: {e}")
        raise HTTPException(status_code=500, detail="シナリオの実行に失敗しました")


@app.post("/api/v1/scenarios/run-all")
async def run_all_scenarios():
    """すべてのシナリオを実行"""
    try:
        result = await scenario_runner.run_all_scenarios()
        return result
    except Exception as e:
        logger.error(f"全シナリオ実行エラー: {e}")
        raise HTTPException(status_code=500, detail="全シナリオの実行に失敗しました")


# エラーハンドリング
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラー"""
    logger.error(f"未処理の例外: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "内部サーバーエラー",
            "message": "システムエラーが発生しました"
            if not settings.debug
            else str(exc),
            "path": str(request.url.path),
        },
    )


# 開発サーバー起動関数
def run_development_server():
    """開発サーバー起動"""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    run_development_server()
