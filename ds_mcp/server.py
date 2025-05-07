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
from ds_mcp.routers.workflow import workflow


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
app.include_router(workflow, prefix="/api/workflow", tags=["Workflow"])


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

    return MCPRunResponse(result=result)


@app.get("/", tags=["Root"])
async def root(request: Request):
    # If client requests SSE, respond with a persistent event stream (heartbeat)
    accept_header = request.headers.get("accept", "").lower()
    if "text/event-stream" in accept_header:
        import asyncio
        import json

        # Set up the global message queue
        global mcp_message_queue
        mcp_message_queue = asyncio.Queue()

        logger.info("MCP SSE connection established")

        async def event_generator():
            # Send an initial MCP protocol initialization response
            init_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "serverInfo": {
                        "name": "Data Science MCP Server",
                        "version": "0.1.0",
                        "description": "MCP server for data science best practices",
                    },
                    "capabilities": {
                        "tools": [
                            {
                                "name": "get-workflow-guidance",
                                "description": "Get guidance on best practices of writing a data science / machine learning task",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
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
                                    "required": ["task_description"],
                                },
                            }
                        ]
                    },
                },
            }

            logger.info(f"Sending MCP initialization response: {init_response}")

            # Send the initialization response as an SSE event
            yield f"data: {json.dumps(init_response)}\n\n"

            # Process messages from the queue and send heartbeats
            while True:
                try:
                    # Try to get a message from the queue with a timeout
                    message = await asyncio.wait_for(mcp_message_queue.get(), timeout=5)
                    logger.info(f"Sending message to client: {message}")
                    # Send the message as an SSE event
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # If no message is available, send a heartbeat
                    yield ": keep-alive\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return {"status": "ok", "message": "Data Science MCP server is running"}


# Global message queue for MCP communication
mcp_message_queue = None


@app.get("/mcp", tags=["MCP"])
async def mcp_get_probe():
    return {"status": "ok", "message": "MCP endpoint is ready"}


@app.post("/mcp-message", tags=["MCP"])
async def handle_mcp_message(request: Request):
    global mcp_message_queue

    if mcp_message_queue is None:
        return {"error": "No active MCP connection"}

    data = await request.json()
    logger.info(f"Received MCP message: {data}")

    # Process the message based on JSON-RPC method
    if data.get("method") == "invoke":
        request_id = data.get("id")
        params = data.get("params", {})
        tool_name = params.get("name")
        parameters = params.get("parameters", {})

        result = {"error": f"Tool {tool_name} not implemented yet"}

        if tool_name == "get-workflow-guidance":
            # Convert parameters to match our router endpoint
            task_description = parameters.get("task_description", "")
            data_type = parameters.get("data_type", "tabular")
            context = parameters.get("context", None)

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

        # Send the result back in JSON-RPC format
        response = {"jsonrpc": "2.0", "id": request_id, "result": result}

        await mcp_message_queue.put(response)

    return {"status": "ok"}


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
    print("Add to your VS Code MCP config file (.vscode/mcp.json):")
    print(
        f"""{{
  "servers": {{
    "DS-MCP": {{
      "type": "sse",
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
