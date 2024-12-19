from __future__ import annotations
from typing import Dict, List, Type, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass
import os
from functools import wraps

T = TypeVar('T')

@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    name: str
    provider: str = "openai"
    api_version: str = "v1"
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

class ModelRegistry:
    """
    A registry pattern for managing OpenAI model configurations and their clients.
    """
    _registry: Dict[str, Dict[str, Type]] = {}
    _configs: Dict[str, ModelConfig] = {}
    
    def __new__(cls):
        """ python singleton pattern """
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, 
                configs: List[Union[Dict[str, Any], ModelConfig]]):
        """
        Class decorator that registers a model with its configurations.
        
        Args:
            configs: List of model configurations
            initialize: Whether to initialize the client immediately
        """
        def decorator(client_class: Type[T]) -> Type[T]:
            @wraps(client_class)
            def wrapper(*args, **kwargs):
                return client_class(*args, **kwargs)
                
            for config in configs:
                
                if isinstance(config, dict):
                    config = ModelConfig(**config)
                
                # Store the configuration
                cls._configs[config.name] = config
                
                # Initialize provider registry if needed
                if config.provider not in cls._registry:
                    cls._registry[config.provider] = {}
                
                # Register the client class
                cls._registry[config.provider][config.name] = client_class
                
                # Initialize client if requested
                instance = client_class(
                            api_key=config.api_key,
                            provider=config.provider,
                            base_url=config.base_url)
                setattr(cls, f"_{config.name}_instance", instance)
            
            return wrapper
        return decorator
    
    @classmethod
    def get_client(cls, model_name: str, **kwargs) -> Optional[Any]:
        """Get a client instance for a specific model."""
        config = cls._configs.get(model_name)
        if not config:
            raise ValueError(f"No configuration found for model: {model_name}")
            
        instance_name = f"_{model_name}_instance"
        instance = getattr(cls, instance_name, None)
        
        if instance is None:
            client_class = cls._registry.get(config.provider, {}).get(model_name)
            if client_class:
                api_key = kwargs.get('api_key') or config.api_key or os.getenv("OPENAI_API_KEY")
                instance = client_class(
                    api_key=api_key,
                    base_url=config.base_url,
                    **kwargs
                )
                setattr(cls, instance_name, instance)
                
        return instance
    
    @classmethod
    def get_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get the configuration for a specific model."""
        return cls._configs.get(model_name)
    
    @classmethod
    def list_models(cls, provider: Optional[str] = None) -> List[str]:
        """List all registered models, optionally filtered by provider."""
        if provider:
            return list(cls._registry.get(provider, {}).keys())
        return list(cls._configs.keys())
    