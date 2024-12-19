from typing import Dict, Type, Any
from .types import CompletionStrategy

class CompletionFactory:
    """Factory for creating appropriate completion strategies"""
    
    _strategies: Dict[str, Type[CompletionStrategy]] = {}
    
    @classmethod
    def register_strategy(cls, provider: str, strategy_class: Type[CompletionStrategy]):
        """Register a completion strategy for a provider"""
        cls._strategies[provider] = strategy_class
    
    @classmethod
    def create_strategy(cls, provider: str, client: Any) -> CompletionStrategy:
        """Create a strategy instance for the given provider"""
        if provider not in cls._strategies:
            raise ValueError(f"No strategy registered for provider: {provider}")
        return cls._strategies[provider](client)