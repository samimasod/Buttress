"""
Application configuration management.
Loads settings from environment variables and config files.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


_DEV_ENVIRONMENTS = {"local", "test", "dev", "development"}
_DEFAULT_JWT_SECRET = "skeleton_local_jwt_secret_key_2026"
_INSECURE_JWT_SECRETS = {
    _DEFAULT_JWT_SECRET,
    "replace-with-a-long-random-jwt-secret",
}


def load_yaml_config(file_path: str) -> dict:
    path = Path(file_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    environment: str = Field(default="local", alias="DATABASE_ENV")
    debug: bool = Field(default=True, alias="API_DEBUG")
    sql_echo: bool = Field(default=False, alias="SQL_ECHO")
    jwt_secret: str = Field(default=_DEFAULT_JWT_SECRET, alias="JWT_SECRET")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_log_level: str = Field(default="INFO", alias="API_LOG_LEVEL")

    database_env: str = Field(default="local", alias="DATABASE_ENV")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    cloud_sql_host: Optional[str] = Field(default=None, alias="CLOUD_SQL_HOST")
    db_user: Optional[str] = Field(default=None, alias="DB_USER")
    db_password: Optional[str] = Field(default=None, alias="DB_PASSWORD")

    firebase_project_id: Optional[str] = Field(default=None, alias="FIREBASE_PROJECT_ID")
    firebase_api_key: Optional[str] = Field(default=None, alias="FIREBASE_API_KEY")
    firebase_admin_credentials_json: Optional[str] = Field(
        default=None,
        alias="FIREBASE_ADMIN_CREDENTIALS_JSON",
    )
    firebase_admin_credentials_path: Optional[str] = Field(
        default=None,
        alias="FIREBASE_ADMIN_CREDENTIALS_PATH",
    )
    super_admin_emails_raw: str = Field(default="", alias="SUPER_ADMIN_EMAILS")

    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_model: str = Field(default="google/gemini-3.1-flash-lite-preview", alias="OPENROUTER_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")

    gcp_project_id: Optional[str] = Field(default=None, alias="GCP_PROJECT_ID")
    gcs_bucket_name: Optional[str] = Field(default=None, alias="GCS_BUCKET_NAME")

    storage_provider: str = Field(default="local", alias="STORAGE_PROVIDER")
    storage_local_path: str = Field(default="./data/storage", alias="STORAGE_LOCAL_PATH")
    storage_gcs_prefix: str = Field(default="artifacts", alias="STORAGE_GCS_PREFIX")
    s3_bucket_name: Optional[str] = Field(default=None, alias="S3_BUCKET_NAME")
    s3_region: Optional[str] = Field(default=None, alias="S3_REGION")
    s3_prefix: str = Field(default="artifacts", alias="S3_PREFIX")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_backend: str = Field(default="redis", alias="CACHE_BACKEND")
    cache_default_ttl_seconds: int = Field(default=300, alias="CACHE_DEFAULT_TTL_SECONDS")
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    cors_origins: List[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

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
    def validate_security_defaults(self):
        if (
            self.environment.lower() not in _DEV_ENVIRONMENTS
            and self.jwt_secret in _INSECURE_JWT_SECRETS
        ):
            raise ValueError(
                "JWT_SECRET must be set to a strong unique value outside local/test environments"
            )
        return self

    @property
    def is_development_environment(self) -> bool:
        return self.environment.lower() in _DEV_ENVIRONMENTS

    @property
    def llm_api_key(self) -> Optional[str]:
        if self.llm_provider.lower() == "openai":
            return self.openai_api_key
        return self.openrouter_api_key or self.openai_api_key

    @property
    def llm_model(self) -> str:
        if self.llm_provider.lower() == "openai":
            return self.openai_model
        return self.openrouter_model

    @property
    def super_admin_emails(self) -> List[str]:
        return [
            email.strip().lower()
            for email in self.super_admin_emails_raw.split(",")
            if email.strip()
        ]

    def get_database_url(self) -> str:
        if self.database_url:
            if self.database_url.startswith("postgresql://"):
                return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.database_url

        config = load_yaml_config("config/database.yaml")
        env_config = config.get("environments", {}).get(self.database_env, {})
        driver = env_config.get("driver", "sqlite")

        if driver == "sqlite":
            db_path = env_config.get("database", "./data/skeleton_local.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"

        if driver == "postgresql":
            host = self.cloud_sql_host or env_config.get("host", "localhost")
            port = env_config.get("port", 5432)
            database = env_config.get("database", "skeleton")
            username = self.db_user or env_config.get("username", "postgres")
            password = self.db_password or env_config.get("password", "")
            return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"

        raise ValueError(f"Unsupported database driver: {driver}")

    def get_sync_database_url(self) -> str:
        async_url = self.get_database_url()

        if "sqlite+aiosqlite" in async_url:
            return async_url.replace("sqlite+aiosqlite", "sqlite")
        if "postgresql+asyncpg" in async_url:
            return async_url.replace("postgresql+asyncpg", "postgresql")

        return async_url


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
