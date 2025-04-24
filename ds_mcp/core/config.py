"""Configuration settings for the Data Science MCP server."""

import os
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENV: str = Field(default="development", description="Environment")

    # API settings
    API_PREFIX: str = Field(default="/api", description="API prefix")
    API_VERSION: str = Field(default="v1", description="API version")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Log level")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings: Application settings instance
    """
    return settings
