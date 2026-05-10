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


# ── 生命周期管理 ──────────────────────────────────────────────────────────────


# 定义应用生命周期管理器 负责挂载和卸载全局资源
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 尝试初始化核心关系型数据库组件
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # 获取单例应用状态并执行模型权重和知识库索引的装载
    state = get_app_state()
    try:
        state.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize AppState: {e}")

    yield
    logger.info("Shutting down application...")


# ── 应用装配 ──────────────────────────────────────────────────────────────────


# 工厂函数 创建并装配 FastAPI 主应用实例
def create_app() -> FastAPI:
    app = FastAPI(title="Library Smart Assistant", lifespan=lifespan)

    # 注入跨域资源共享中间件 开放各端调用权限
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载核心业务接口路由
    app.include_router(api_router, prefix="/api")

    # 若存在打包好的前端页面目录则一并托管 以提供开箱即用的前端体验
    if os.path.exists("frontend/dist"):
        app.mount(
            "/", StaticFiles(directory="frontend/dist", html=True), name="frontend"
        )

    return app


app = create_app()
