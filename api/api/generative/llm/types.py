
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, Generic, Iterable, List,
    Optional, Protocol, TypeVar, runtime_checkable
)


# Type variables for different completion types
T_Completion = TypeVar('T_Completion')
T_Token = TypeVar('T_Token')
T_Function = TypeVar('T_Function')

class CompletionCapability(Enum):
    """Capabilities that a model completion might support"""
    STREAMING = auto()
    FUNCTION_CALLING = auto()
    TOOL_CALLING = auto()
    SYSTEM_MESSAGES = auto()

@dataclass
class CompletionMetadata:
    """Metadata about a completion response"""
    model : Optional[str]
    provider: str
    created : Optional[int]
    stream: bool
    usage: Optional[Dict[str, int]] = None
    capabilities: List[CompletionCapability] = field(default_factory=list)

@runtime_checkable
class CompletionResponse(Protocol[T_Completion, T_Token]):
    """Protocol for completion responses"""
    @property
    def raw_response(self) -> T_Completion:
        """Get the raw completion response from the model"""
        ...
    
    @property
    def metadata(self) -> CompletionMetadata:
        """Get metadata about the completion"""
        ...
    
    def iter_tokens(self) -> Iterable[T_Token]:
        """Iterate over tokens in the completion"""
        ...
    
    def get_text(self) -> str:
        """Get the full text of the completion"""
        ...

@runtime_checkable
class FunctionCallable(Protocol[T_Function]):
    """Protocol for completions that support function calling"""
    def get_function_call(self) -> Optional[T_Function]:
        """Get function call information if present"""
        ...

class CompletionStrategy(ABC):
    """Base strategy for handling model completions"""
    
    @abstractmethod
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> CompletionResponse:
        """Create a completion from the model"""
        pass
    
    @abstractmethod
    def supports_capability(self, capability: CompletionCapability) -> bool:
        """Check if this completion strategy supports a given capability"""
        pass