# Contributing to the Data Science MCP Server

Thank you for your interest in contributing to this project! This document provides guidelines and best practices for contributing to the Data Science MCP server.

## Development Environment

This project uses `pixi` for environment management. To set up your development environment:

1. Install pixi if you haven't already (https://pixi.sh)
2. Clone the repository
3. Run `pixi install` to set up the environment
4. Activate the environment with `pixi shell`

## Code Style and Standards

We follow these coding standards:

- Code formatting with `ruff format` (line length of 88 characters)
- Linting with `ruff`
- Documentation with docstrings (following Google style)
- Type hints for all function signatures

Our pre-commit hooks handle most formatting and linting automatically.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Set them up with:

```bash
pixi run pre-commit install
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure they pass: `pixi run test`
5. Run linting checks: `pixi run lint`
6. Push your branch and submit a pull request

## Testing

Write tests for all new features or bug fixes. Run tests with:

```bash
pixi run test
```

## Data Science Best Practices

When contributing to this MCP server, please follow these data science best practices:

1. **Use Polars**: Prefer `polars` over `pandas` for data processing tasks
2. **Document methods thoroughly**: Include example code in all API responses
3. **Focus on reproducibility**: Emphasize version control, seed setting, and environment management
4. **Specify metrics**: Always include appropriate evaluation metrics for models
5. **Include visualizations**: Where applicable, include visualization code examples
6. **Consider scalability**: Provide guidance on scaling solutions for different data sizes
7. **Security awareness**: Avoid suggesting insecure coding practices

## Adding New Endpoints

When adding new endpoints:
1. Create a new file in the `routers` directory if it's a new domain
2. Use Pydantic models for request/response validation
3. Include comprehensive documentation
4. Add the router to the main server file
