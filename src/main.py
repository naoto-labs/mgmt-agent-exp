from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from src.config.settings import settings, validate_startup_settings
from src.config.security import setup_secure_logging, secure_config

# ロガーのセットアップ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    # 設定検証
    if not validate_startup_settings():
        logger.error("設定検証に失敗しました。アプリケーションを終了します。")
        raise RuntimeError("設定検証エラー")

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
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # 開発環境用
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
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
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
        "debug_mode": settings.debug
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
                "ai_models": "pending"  # AIモデルチェック後に更新
            }
        }

        # データベースチェック（後で実装）
        # health_status["services"]["database"] = await check_database()

        # AIモデルチェック（後で実装）
        # health_status["services"]["ai_models"] = await check_ai_models()

        return JSONResponse(
            status_code=200,
            content=health_status
        )

    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        raise HTTPException(
            status_code=503,
            detail="ヘルスチェックに失敗しました"
        )

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
        "forbidden_patterns": settings.forbidden_patterns
    }

# エラーハンドリング
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラー"""
    logger.error(f"未処理の例外: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "内部サーバーエラー",
            "message": "システムエラーが発生しました" if not settings.debug else str(exc),
            "path": str(request.url.path)
        }
    )

# 開発サーバー起動関数
def run_development_server():
    """開発サーバー起動"""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )

if __name__ == "__main__":
    run_development_server()
