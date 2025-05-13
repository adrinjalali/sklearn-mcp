"""Main server module for the Data Science MCP server."""

import sys
from mcp.server.fastmcp import FastMCP
from ds_mcp.routers.workflow import get_workflow_guidance, WorkflowRequest

# Create the MCP server
mcp = FastMCP("Data Science MCP Server")


@mcp.tool(description="Get guidance on best practices for a data science/ML task")
async def get_workflow_guidance_tool(
    task_description: str, data_type: str = "tabular", context: str = None
):
    """Get guidance on data science workflow best practices."""
    req = WorkflowRequest(
        task_description=task_description, data_type=data_type, context=context
    )
    return await get_workflow_guidance(req)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Data Science MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default="stdio",
        help="Transport to use: stdio (default), http, or streamable-http.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for HTTP transport."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport."
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        print("[INFO] Starting MCP server in stdio mode...", file=sys.stderr)
        mcp.run(transport="stdio")
    elif args.transport == "http":
        print(
            f"[INFO] Starting MCP server on http://{args.host}:{args.port} ...",
            file=sys.stderr,
        )
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        print(
            f"[INFO] Starting MCP server (streamable-http) on http://{args.host}:{args.port} ...",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
