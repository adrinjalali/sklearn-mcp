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

## API Endpoints

Documentation for the API endpoints is available at `/docs` when the server is running.

## Development

To contribute to this project, please follow the guidelines in the CONTRIBUTING.md file.
