"""
CatalystLens FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
