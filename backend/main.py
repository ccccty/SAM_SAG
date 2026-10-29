"""SAM_SAG FastAPI 应用入口.

运行方式:
    python -m backend.main          # 推荐(也可是 start.bat 一键启动)
    uvicorn backend.main:app        # 等价手动方式

行为:
- /api/* 走业务路由, 统一 JSON 契约;
- 其余路径回退到 frontend/ 静态单页(html=True, / 即 index.html);
- 所有 ApiError → {ok:false, error:{code,message}} + 对应 HTTP 状态码;
- 未捕获异常 → 500 + INTERNAL_ERROR, 绝不把堆栈暴露给浏览器。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import ApiError, router
from .core import config


def create_app() -> FastAPI:
    config.ensure_dirs()

    app = FastAPI(title=config.APP_TITLE, version=__version__)

    # 本地工具默认放开跨域, 便于调试期前端独立端口开发
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"ok": False, "error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {"code": "INTERNAL_ERROR", "message": f"服务内部错误: {exc}"},
            },
        )

    # 业务 API 优先于静态文件匹配
    app.include_router(router, prefix="/api")

    # 前端单页: 挂载在根路径(html=True → / 返回 index.html)
    app.mount(
        "/",
        StaticFiles(directory=str(config.FRONTEND_DIR), html=True),
        name="frontend",
    )
    return app


app = create_app()


def main() -> None:
    import uvicorn

    print(f"[SAM_SAG] {config.APP_TITLE} v{__version__}")
    print(f"[SAM_SAG] 服务地址: http://{config.HOST}:{config.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    main()
