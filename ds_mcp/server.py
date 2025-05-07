"""Main server module for the Data Science MCP server."""

import sys
from typing import Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from loguru import logger
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from ds_mcp.core.config import settings
from ds_mcp.routers import (
    data_prep,
    feature_engineering,
    modeling,
    evaluation,
    workflow,
)


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
app.include_router(
    data_prep.router, prefix="/api/data-preparation", tags=["Data Preparation"]
)
app.include_router(
    feature_engineering.router,
    prefix="/api/feature-engineering",
    tags=["Feature Engineering"],
)
app.include_router(modeling.router, prefix="/api/modeling", tags=["Modeling"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(workflow.router, prefix="/api/workflow", tags=["Workflow"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


# MCP Protocol required endpoints
class MCPInitRequest(BaseModel):
    """MCP initialization request model."""

    clientVersion: str = Field(..., description="Client version")
    capabilities: List[str] = Field(
        default_factory=list, description="Client capabilities"
    )


class MCPInitResponse(BaseModel):
    """MCP initialization response model."""

    serverInfo: Dict[str, Any] = Field(..., description="Server information")
    tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Available tools"
    )


@app.post("/mcp", tags=["MCP"])
async def mcp_init(request: MCPInitRequest) -> MCPInitResponse:
    """MCP initialization endpoint."""
    logger.info(f"MCP initialization called with request: {request}")
    logger.info(
        f"MCP initialization request from client version: {request.clientVersion}"
    )

    return MCPInitResponse(
        serverInfo={
            "name": "Data Science MCP Server",
            "version": "0.1.0",
            "description": "MCP server for data science best practices",
        },
        tools=[
            {
                "name": "get-workflow-guidance",
                "description": "Get guidance on best practices of writing a data science / machine learning task",
                "parameters": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the data science task",
                    },
                    "data_type": {
                        "type": "string",
                        "description": "Type of data (e.g., tabular, text, image)",
                        "default": "tabular",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the task",
                        "default": "",
                    },
                },
            },
            {
                "name": "get-data-prep-guidance",
                "description": "Get guidance on data preparation best practices",
                "parameters": {
                    "data_type": {
                        "type": "string",
                        "description": "Type of data (e.g., tabular, text, image)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language for examples",
                        "default": "python",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the task",
                        "default": "",
                    },
                },
            },
            {
                "name": "get-feature-engineering-guidance",
                "description": "Get guidance on feature engineering techniques",
                "parameters": {
                    "data_type": {
                        "type": "string",
                        "description": "Type of data (e.g., tabular, text, image)",
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Type of ML task (e.g., classification, regression, clustering)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language for examples",
                        "default": "python",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the task",
                        "default": "",
                    },
                },
            },
            {
                "name": "get-modeling-recommendations",
                "description": "Get recommendations for ML models",
                "parameters": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of ML task (e.g., classification, regression, clustering)",
                    },
                    "data_type": {
                        "type": "string",
                        "description": "Type of data (tabular, text, image, time-series)",
                        "default": "tabular",
                    },
                    "data_size": {
                        "type": "string",
                        "description": "Size of the data (small, medium, large)",
                        "default": "medium",
                    },
                    "constraints": {
                        "type": "string",
                        "description": "Constraints for the model (e.g., interpretability, explainability)",
                        "default": "",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language for examples",
                        "default": "python",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the task",
                        "default": "",
                    },
                },
            },
            {
                "name": "get-evaluation-guidance",
                "description": "Get guidance on evaluation metrics",
                "parameters": {
                    "task_type": {
                        "type": "string",
                        "description": "Type of ML task (e.g., classification, regression, clustering)",
                    }
                },
            },
        ],
    )


class MCPRunRequest(BaseModel):
    """MCP tool execution request model."""

    toolName: str = Field(..., description="Tool name to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )


class MCPRunResponse(BaseModel):
    """MCP tool execution response model."""

    result: Dict[str, Any] = Field(..., description="Tool execution result")


@app.post("/mcp/run", tags=["MCP"])
async def mcp_run(request: MCPRunRequest) -> MCPRunResponse:
    """MCP tool execution endpoint."""
    logger.info(f"MCP tool execution called with request: {request}")
    logger.info(f"MCP tool execution request: {request.toolName}")

    result = {"error": f"Tool {request.toolName} not implemented yet"}

    if request.toolName == "get-workflow-guidance":
        # Convert parameters to match our router endpoint
        task_description = request.parameters.get("task_description", "")
        data_type = request.parameters.get("data_type", "tabular")
        context = request.parameters.get("context", None)

        logger.info(
            f"Processing MCP workflow guidance request: task_description='{task_description}', data_type='{data_type}'"
        )

        # Create a request object for our existing endpoint
        from ds_mcp.routers.workflow import WorkflowRequest

        workflow_request = WorkflowRequest(
            task_description=task_description, data_type=data_type, context=context
        )

        # Call our existing endpoint logic
        from ds_mcp.routers.workflow import get_workflow_guidance

        result = await get_workflow_guidance(workflow_request)
        logger.info("MCP workflow guidance request completed successfully")

    elif request.toolName == "get-data-prep-guidance":
        # Convert parameters to match our router endpoint
        data_type = request.parameters.get("data_type", "tabular")
        language = request.parameters.get("language", "python")
        context = request.parameters.get("context", None)
        logger.info(
            f"Processing MCP data prep guidance request: data_type='{data_type}', language='{language}', context='{context}'"
        )
        # Create a request object for our existing endpoint
        from ds_mcp.routers.data_prep import DataCleaningRequest

        data_prep_request = DataCleaningRequest(
            data_type=data_type, language=language, context=context
        )
        # Call our existing endpoint logic
        from ds_mcp.routers.data_prep import get_data_cleaning_guidance

        result = await get_data_cleaning_guidance(data_prep_request)
        logger.info("MCP data prep guidance request completed successfully")

    elif request.toolName == "get-feature-engineering-guidance":
        data_type = request.parameters.get("data_type", "tabular")
        task_type = request.parameters.get("task_type", "classification")
        language = request.parameters.get("language", "python")
        context = request.parameters.get("context", None)
        logger.info(
            f"Processing MCP feature engineering guidance request: data_type='{data_type}', task_type='{task_type}', language='{language}', context='{context}'"
        )
        # Create a request object for our existing endpoint
        from ds_mcp.routers.feature_engineering import FeatureEngineeringRequest

        feature_request = FeatureEngineeringRequest(
            data_type=data_type, task_type=task_type, language=language, context=context
        )
        # Call our existing endpoint logic
        from ds_mcp.routers.feature_engineering import get_feature_engineering_guidance

        result = await get_feature_engineering_guidance(feature_request)
        logger.info("MCP feature engineering guidance request completed successfully")

    elif request.toolName == "get-modeling-recommendations":
        task_type = request.parameters.get("task_type", "classification")
        data_type = request.parameters.get("data_type", "tabular")
        data_size = request.parameters.get("data_size", "medium")
        constraints = request.parameters.get("constraints", None)
        language = request.parameters.get("language", "python")
        context = request.parameters.get("context", None)
        logger.info(
            f"Processing MCP modeling recommendations request: task_type='{task_type}', data_type='{data_type}', data_size='{data_size}', constraints='{constraints}', language='{language}', context='{context}'"
        )
        # Create a request object for our existing endpoint
        from ds_mcp.routers.modeling import ModelingRequest

        modeling_request = ModelingRequest(
            task_type=task_type,
            data_type=data_type,
            data_size=data_size,
            constraints=constraints,
            language=language,
            context=context,
        )
        # Call our existing endpoint logic
        from ds_mcp.routers.modeling import get_modeling_recommendations

        result = await get_modeling_recommendations(modeling_request)
        logger.info("MCP modeling recommendations request completed successfully")

    elif request.toolName == "get-evaluation-guidance":
        task_type = request.parameters.get("task_type", "classification")

        logger.info(
            f"Processing MCP evaluation guidance request: task_type='{task_type}'"
        )

        # Create a request object for our existing endpoint
        from ds_mcp.routers.evaluation import EvaluationRequest

        eval_request = EvaluationRequest(task_type=task_type)

        # Call our existing endpoint logic
        from ds_mcp.routers.evaluation import get_evaluation_guidance

        result = await get_evaluation_guidance(eval_request)
        logger.info("MCP evaluation guidance request completed successfully")

    else:
        logger.warning(f"MCP tool execution failed: Unknown tool {request.toolName}")

    return MCPRunResponse(result=result)


@app.get("/", tags=["Root"])
async def root(request: Request):
    # If client requests SSE, respond with a minimal event stream
    if request.headers.get("accept") == "text/event-stream":

        async def event_generator():
            yield "data: ok\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    return {"status": "ok", "message": "Data Science MCP server is running"}


@app.get("/mcp", tags=["MCP"])
async def mcp_get_probe():
    return {"status": "ok", "message": "MCP endpoint is ready"}


@app.middleware("http")
async def log_all_requests(request, call_next):
    # Log all incoming requests with method, url, and headers
    logger.info(
        f"Incoming request: {request.method} {request.url} headers={dict(request.headers)}"
    )
    sys.stdout.flush()
    response = await call_next(request)
    logger.info(
        f"Response status: {response.status_code} for {request.method} {request.url}"
    )
    sys.stdout.flush()
    return response


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

    logger.info(f"Starting Data Science MCP Server on port {settings.PORT}")

    # Print integration information
    print("\n" + "=" * 80)
    print("Data Science MCP Server - Data Science Best Practices for AI Agents")
    print("=" * 80)
    print("\nIntegrate with your AI Code Editor:")

    server_url = f"http://{settings.HOST if settings.HOST != '0.0.0.0' else 'localhost'}:{settings.PORT}"

    print("\n🔹 WINDSURF Integration:")
    print("Add to your Windsurf MCP config file:")
    print(
        f"""{{
  "mcpServers": {{
    "ds-mcp": {{
      "uri": "{server_url}"
    }}
  }}
}}"""
    )

    print("\n🔹 CURSOR Integration:")
    print("Add to your ~/.cursor/mcp.json file:")
    print(
        f"""{{
  "mcpServers": {{
    "ds-mcp": {{
      "uri": "{server_url}"
    }}
  }}
}}"""
    )

    print("\n🔹 VS CODE Integration:")
    print("Add to your VS Code MCP config file:")
    print(
        f"""{{
  "servers": {{
    "DS-MCP": {{
      "type": "http",
      "url": "{server_url}"
    }}
  }}
}}"""
    )

    print("\n" + "=" * 80)
    print(f"Server running at {server_url}")
    print(f"API documentation available at {server_url}/docs")
    print("=" * 80 + "\n")

    uvicorn.run(
        "ds_mcp.server:app", host=settings.HOST, port=settings.PORT, reload=True
    )
