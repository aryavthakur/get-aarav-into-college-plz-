"""
CatalystLens FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging_config import audit_event, logger

# Allowed origins — override via CORS_ORIGINS env var (comma-separated) in production
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    audit_event("startup", version="0.1.0", cors_origins=_CORS_ORIGINS)
    yield
    audit_event("shutdown")


app = FastAPI(
    title="CatalystLens",
    description=(
        "Probabilistic biotech capital-to-catalyst audit engine. "
        "Evaluates whether a company's financial runway can survive long enough "
        "to reach its next meaningful clinical or scientific catalyst. "
        "All outputs are model estimates. This is NOT investment advice."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("domain_validation_error path=%s error=%s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "domain_validation_error"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("unhandled_error path=%s error=%r", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal engine error. See server logs.", "error_type": "engine_error"},
    )


app.include_router(router)
