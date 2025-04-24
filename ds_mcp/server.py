"""Main server module for the Data Science MCP server."""

import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from loguru import logger
from pydantic import BaseModel, Field

from ds_mcp.core.config import settings
from ds_mcp.routers import data_prep, feature_engineering, modeling, evaluation


# Initialize FastAPI app
app = FastAPI(
    title="Data Science MCP Server",
    description="MCP server for data science projects",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_prep.router, prefix="/api/data-preparation", tags=["Data Preparation"])
app.include_router(
    feature_engineering.router, prefix="/api/feature-engineering", tags=["Feature Engineering"]
)
app.include_router(modeling.router, prefix="/api/modeling", tags=["Modeling"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


# Custom OpenAPI schema
def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Data Science MCP API",
        version="0.1.0",
        description="API for the Data Science MCP server",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {settings.PORT}")
    uvicorn.run("ds_mcp.server:app", host=settings.HOST, port=settings.PORT, reload=True)
