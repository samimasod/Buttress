"""Configuration settings for the SuperAdmin Monitoring API microservice."""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


_DEV_ENVIRONMENTS = {"local", "test", "dev", "development"}


def load_admin_yaml_config() -> dict:
    """Load settings from config/admin.yaml if present."""
    config_file = Path("config/admin.yaml")
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "admin" in data:
                    return data["admin"]
        except Exception as e:
            print(f"Warning loading config/admin.yaml: {e}")
    return {}


_yaml_admin_config = load_admin_yaml_config()


class AdminSettings(BaseSettings):
    """SuperAdmin API settings loaded from environment variables & config/admin.yaml."""

    environment: str = Field(default="local", alias="DATABASE_ENV")
    debug: bool = Field(default=True, alias="API_DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="ADMIN_API_HOST")
    api_port: int = Field(default=8001, alias="ADMIN_API_PORT")

    # Configurable SuperAdmin authentication toggle
    admin_auth_enabled: bool = Field(
        default=_yaml_admin_config.get("admin_auth_enabled", True),
        alias="ADMIN_AUTH_ENABLED"
    )

    database_env: str = Field(default="local", alias="DATABASE_ENV")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    super_admin_api_key: Optional[str] = Field(
        default=_yaml_admin_config.get("super_admin_api_key"),
        alias="SUPER_ADMIN_API_KEY"
    )
    super_admin_emails_raw: str = Field(default="", alias="SUPER_ADMIN_EMAILS")

    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    storage_provider: str = Field(default="local", alias="STORAGE_PROVIDER")
    cache_backend: str = Field(default="redis", alias="CACHE_BACKEND")
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    cors_origins: List[str] = Field(
        default=["http://localhost:3001", "http://localhost:3002", "http://localhost:5173", "http://localhost:3000"],
        alias="CORS_ORIGINS"
    )

    class Config:
        env_file = (".env", ".env.local")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def validate_admin_security(self):
        if not self.is_development_environment and not self.admin_auth_enabled:
            raise ValueError(
                "ADMIN_AUTH_ENABLED cannot be disabled outside local/test environments"
            )
        return self

    @property
    def is_development_environment(self) -> bool:
        return self.environment.lower() in _DEV_ENVIRONMENTS

    @property
    def super_admin_emails(self) -> List[str]:
        return [
            email.strip().lower()
            for email in self.super_admin_emails_raw.split(",")
            if email.strip()
        ]


admin_settings = AdminSettings()
