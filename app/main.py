"""FastAPI entrypoint:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.api import http, ws
from app.config import Settings, get_settings
from app.container import Container, build_container


def create_app(container: Optional[Container] = None, settings: Optional[Settings] = None) -> FastAPI:
    """App factory. Tests pass a pre-built container; production builds one from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or get_settings()
        logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        app.state.container = container or build_container(cfg)
        await app.state.container.startup()
        try:
            yield
        finally:
            await app.state.container.shutdown()

    app = FastAPI(title="Hatchi - personalized AI chat", version="0.1.0", lifespan=lifespan)
    if container is not None:
        app.state.container = container  # available even when lifespan is not run (e.g. ASGITransport)
    app.include_router(http.router)
    app.include_router(ws.router)
    return app


app = create_app()
