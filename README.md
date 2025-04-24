# Data Science MCP Server

A Modular Context Provider (MCP) server for data science projects. This server exposes endpoints to provide guidance, rules, and code examples for ML and data science workflows.

## Overview

This MCP server provides specialized knowledge about data science best practices, helping AI agents follow specific conventions and practices when working with:

- Data preparation and cleaning
- Feature engineering
- Model selection
- Training workflows
- Evaluation metrics
- Serialization and deployment
- Data visualization

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
pixi run serve
```

### Integrate with your AI Code Editor

#### Option 1: Connect to a Running MCP Server

If you already have the DS-MCP server running (on the default port 8000), you can configure your code editor to connect to it directly:

##### Install in Windsurf

Add this to your Windsurf MCP config file:

```json
{
  "mcpServers": {
    "ds-mcp": {
      "uri": "http://localhost:8000"
    }
  }
}
```

##### Install in Cursor

Go to: Settings -> Cursor Settings -> MCP -> Add new global MCP server

Paste the following configuration into your Cursor `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "ds-mcp": {
      "uri": "http://localhost:8000"
    }
  }
}
```

##### Install in VS Code

Add this to your VS Code MCP config file:

```json
{
  "servers": {
    "DS-MCP": {
      "type": "http",
      "url": "http://localhost:8000"
    }
  }
}
```

#### Option 2: Let the IDE Start the MCP Server

Alternatively, you can configure your IDE to start the MCP server when needed:

##### Install in Windsurf

Add this to your Windsurf MCP config file:

```json
{
  "mcpServers": {
    "ds-mcp": {
      "command": "pixi",
      "args": ["run", "serve"],
      "cwd": "/path/to/ds-agent/ds-mcp"
    }
  }
}
```

##### Install in Cursor

Go to: Settings -> Cursor Settings -> MCP -> Add new global MCP server

Paste the following configuration into your Cursor `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "ds-mcp": {
      "command": "pixi",
      "args": ["run", "serve"],
      "cwd": "/path/to/ds-agent/ds-mcp"
    }
  }
}
```

##### Install in VS Code

Add this to your VS Code MCP config file:

```json
{
  "servers": {
    "DS-MCP": {
      "type": "stdio",
      "command": "pixi",
      "args": ["run", "serve"],
      "cwd": "/path/to/ds-agent/ds-mcp"
    }
  }
}
```

Replace `/path/to/ds-agent/ds-mcp` with the actual path to this project on your system.

## API Endpoints

Documentation for the API endpoints is available at `/docs` when the server is running.

## Development

To contribute to this project, please follow the guidelines in the CONTRIBUTING.md file.
