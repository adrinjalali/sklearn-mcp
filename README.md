# Data Science MCP Server

A Modular Context Provider (MCP) server for data science projects. This server exposes endpoints to provide guidance, rules, and code examples for ML and data science workflows, as well as general Python project best practices for agent-based systems.

---

## Overview

This MCP server provides specialized knowledge and best practices for data science and machine learning projects, including:

- General Python project guidelines for agent development
- Best practices on using scikit-learn
- Guidelines on using additional libraries such as `skops`, `skore`, and `skrub`
- Serialization and deployment

Endpoints return curated Markdown documents with actionable guidance for AI agents and developers.

---

## Project Structure

- `ds_mcp/server.py` – Main FastAPI MCP server, exposes endpoints as MCP tools.
- `ds_mcp/routers/workflow_guidance.md` – Markdown with best practices for DS/ML workflows.
- `ds_mcp/routers/python_general.md` – Markdown with general Python project guidelines.
- `ds_mcp/core/` – Configuration and utilities.
- `tests/` – Test suite.

All endpoints that return static guidance use a shared utility to read Markdown documents from the `routers/` directory.

---

## Getting Started

### Prerequisites

- [pixi](https://pixi.sh) for environment management

### Installation

1. Clone the repository
2. Set up the environment:

    ```bash
    pixi install
    ```

3. Run the server:

    ```bash
    pixi run mcp-server
    ```

---

## Integration with AI Code Editors

### Example: Windsurf

Add this to your Windsurf MCP config file:

```json
{
  "mcpServers": {
    "ds-mcp": {
      "command": "pixi",
      "args": [
        "run",
        "--manifest-path",
        "/path/to/ds-agent/ds-mcp/pixi.toml",
        "mcp-server"
      ]
    }
  }
}
```
Replace `/path/to/ds-agent/ds-mcp` with the actual path to this project on your system.

---

## API Endpoints

- `/get_workflow_guidance_tool`
  Returns workflow guidance for data science/ML tasks as Markdown.
  **Arguments:**
    - `task_description` (str): Description of the DS/ML task
    - `data_type` (str, default: "tabular"): Type of data
    - `context` (str, optional): Additional context

- `/get_python_general_guidelines_tool`
  Returns general Python project guidelines as Markdown.

All endpoints are available via the MCP protocol and are documented for interactive exploration at `/docs` when the server is running.

---

## Development

- Use `ruff` for linting and formatting (88 char line length).
- Pre-commit hooks are configured.
- Tests use `pytest`.

To contribute, please follow the guidelines in `CONTRIBUTING.md`.
