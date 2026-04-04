import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.core.dependencies import get_app_state
from backend.core.db import init_db
from backend.api.router import router as api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化数据库
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # 初始化应用全局状态
    state = get_app_state()
    try:
        state.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize AppState: {e}")
    yield
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    app = FastAPI(title="Library Smart Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    # Mount frontend directory for static file serving
    if os.path.exists("frontend"):
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()
