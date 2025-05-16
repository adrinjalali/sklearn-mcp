"""Main server module for the Data Science MCP server."""

import sys
from mcp.server.fastmcp import FastMCP
from ds_mcp.routers.workflow import get_workflow_guidance
from ds_mcp.routers.python_general import get_python_general_guidelines

# Create the MCP server
mcp = FastMCP("Data Science MCP Server")


@mcp.tool(description="Get guidance on best practices for a data science/ML task")
async def get_workflow_guidance_tool(
    task_description: str, data_type: str = "tabular", context: str = None
):
    """
    Get guidance on data science workflow best practices for a specific data science or
    machine learning task.

    This tool is intended for use by AI agents or automated systems that need structured
    advice on best practices, workflow stages, and recommended libraries for a given
    data science/ML workflow. Use this tool when you need:
      - General or stage-specific guidance for any step in a data science/ML project
        (e.g., data preparation, feature engineering, modeling, evaluation).
      - Recommendations on libraries, techniques, or pitfalls for a given data type or
        task.
      - To supplement or validate your own workflow planning or code suggestions with
        best practice advice.

    Args:
        task_description (str):
            A short natural language description of the data science or ML task (e.g.,
            'binary classification', 'time series forecasting', 'image segmentation',
            'recommendation system'). This should specify the main goal or problem type.
        data_type (str, optional):
            The type of data being used. Common values: 'tabular', 'text', 'image',
            'time series', etc. Defaults to 'tabular'.
        context (str, optional):
            Additional context, constraints, or details relevant to the task. This may
            include dataset characteristics, business requirements, performance
            constraints, or any other information that could influence workflow
            recommendations.

    Returns:
        str: Markdown content with workflow guidance.

    When to use this tool:
        - When you need up-to-date, structured, and actionable advice for planning,
          reviewing, or improving a data science/ML workflow.
        - When the task or data type is ambiguous, provide as much context as possible
          for tailored guidance.
        - Use this tool before starting a new workflow, when reviewing an existing
          pipeline, or when troubleshooting workflow design issues.
    """
    return await get_workflow_guidance(task_description, data_type, context)


@mcp.tool(description="Get general Python project guidelines for agent projects")
async def get_python_general_guidelines_tool():
    """
    Get general Python guidelines for agent projects.

    This tool returns best practices and conventions for Python projects intended for agent-based systems, as Markdown.
    """
    return await get_python_general_guidelines()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Data Science MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport to use: stdio (default) or streamable-http.",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        print("[INFO] Starting MCP server in stdio mode...", file=sys.stderr)
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        print(
            "[INFO] Starting MCP server (streamable-http, default host/port) ...",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http")
