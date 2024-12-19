import os
import secrets
from pathlib import Path
from typing import Any, Literal, Union
from urllib.parse import quote_plus
from pydantic import (
    AnyUrl,
    HttpUrl,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding='utf-8',
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True,
    )
    
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30
    FRONTEND_HOST: str = os.getenv("FRONTEND_HOST","http://localhost:3000")
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    
    BACKEND_CORS_ORIGINS: list[Union[AnyUrl, str]] = []
    
    
    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]
    
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "RedArena")
    SENTRY_DSN: HttpUrl | None = None
    
    # MongoDB Settings
    MONGODB_USER: str = os.getenv("MONGODB_USER", "")
    MONGODB_NAME: str = os.getenv("MONGODB_NAME", "Cluster0")
    MONGODB_PASSWORD: str = os.getenv("MONGODB_PASSWORD", "")
    MONGODB_HOST: str = os.getenv("MONGODB_HOST", "cluster0.6dueq.mongodb.net")
    MONGODB_PROTOCOL: str = os.getenv("MONGODB_PROTOCOL","mongodb")  # Can be "mongodb" or "mongodb+srv"
    MONGODB_OPTIONS: str = os.getenv("MONGODB_OPTIONS", "retryWrites=true&w=majority")
    

    # Google OAuth
    GOOGLE_CLIENT_ID : str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_REDIRECT_URI : str = os.getenv("GOOGLE_REDIRECT_URI", "")
    GOOGLE_CLIENT_SECRET : str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    @computed_field
    @property
    def MONGODB_DATABASE_URI(self) -> str:
        """
        Generates MongoDB connection URI with proper escaping of special characters.
        Supports both standard MongoDB and MongoDB+srv protocols.
        """
        try:
            # Validate required fields
            if not all([self.MONGODB_USER, self.MONGODB_PASSWORD, self.MONGODB_HOST]):
                raise ValueError("Missing required MongoDB configuration values")
            
            # Escape username and password
            escaped_user = quote_plus(self.MONGODB_USER)
            escaped_password = quote_plus(self.MONGODB_PASSWORD)
            
            # Construct the connection URI
            if self.MONGODB_PROTOCOL == "mongodb+srv":
                # MongoDB Atlas SRV format
                uri = f"{self.MONGODB_PROTOCOL}://{escaped_user}:{escaped_password}@{self.MONGODB_HOST}"
            else:
                # Standard MongoDB format
                uri = f"{self.MONGODB_PROTOCOL}://{escaped_user}:{escaped_password}@{self.MONGODB_HOST}"
            
            # Add options and app name
            uri = f"{uri}/?{self.MONGODB_OPTIONS}&appName={self.MONGODB_NAME}"
            
            return uri
            
        except Exception as e:
            logger_msg = f"Error generating MongoDB URI: {str(e)}"
            if self.ENVIRONMENT == "local":
                warnings.warn(logger_msg, stacklevel=1)
            raise ValueError(logger_msg)
    
    def check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)


def debug_env_vars():
    """Helper function to debug environment variable loading"""
    env_file_path = ROOT_DIR / ".env"
    if env_file_path.exists():
        print(f"Found .env file at: {env_file_path}")
    else:
        print(f"No .env file found at: {env_file_path}")
            
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

# Initialize settings with debug option
try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {str(e)}")
    debug_env_vars()
    raise