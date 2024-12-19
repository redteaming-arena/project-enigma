from __future__ import annotations
from typing import Iterable, Optional, Any, Dict, List
from ..types import (CompletionCapability, 
                     CompletionMetadata,
                     CompletionResponse, 
                     FunctionCallable, 
                     CompletionStrategy)



class AnthropicCompletionResponse(CompletionResponse["ChatCompletion", str], FunctionCallable[List[Dict[str, str]]]):
    """Anthropic-specific completion response implementation"""
    
    def __init__(self, response: "ChatCompletion", metadata: CompletionMetadata):
        self._response = response
        self._metadata = metadata
    
    @property
    def raw_response(self) -> "ChatCompletion":
        return self._response
    
    @property
    def metadata(self) -> CompletionMetadata:
        return self._metadata
    
    def iter_tokens(self) -> Iterable[str]:
        if hasattr(self._response, "choices") and self._response.choices:
            if hasattr(self._response.choices[0], "message") and self._response.choices[0].message:
                if hasattr(self._response.choices[0].message, "content"):
                    yield self._response.choices[0].message.content
    
    def get_text(self) -> str:
        return "".join(self.iter_tokens())
    
    def get_function_call(self) -> Optional[List[Dict[str, str]]]:
        if (hasattr(self._response.choices[0].message, "tool_calls") and 
            self._response.choices[0].message.tool_calls):
            calls = []
            for call in self._response.choices[0].message.tool_calls:
                calls.append({"name": call.function.name, "arguments": call.function.arguments})
            return calls
        return None
        
class AnthropicCompletionStream(CompletionResponse["Stream[ChatCompletionChunk]", str], FunctionCallable[List[Dict[str, str]]]):
    """Anthropic-specific completion response implementation"""
    
    def __init__(self, stream: "Stream[ChatCompletionChunk]", metadata: CompletionMetadata):
        self._stream = stream
        self._metadata = metadata
        self._chunk_response = []
        self._functions = {}
    
    @property
    def raw_response(self) -> "Stream[ChatCompletionChunk]":
        return self._stream
    
    @property
    def metadata(self) -> CompletionMetadata:
        return self._metadata
    
    def iter_tokens(self) -> Iterable[str]:
        for old_chunk in self._chunk_response:
            yield old_chunk
        at_index = len(self._chunk_response)
        call_index = None
        for chunk in self._stream:
            if hasattr(chunk, "delta") or hasattr(chunk, "content_block"):
                if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "text"):
                    self._chunk_response.append(chunk.delta.text)
                    at_index += 1
                    yield chunk.delta.text
                    while at_index < len(self._chunk_response):
                        yield self._chunk_response[at_index]

                if hasattr(chunk, "content_block") and chunk.content_block.type == 'tool_use':
                    if chunk.content_block.id not in self._functions:
                        self._functions[chunk.content_block.id] = {
                                'arguments': '',
                                'name': chunk.content_block.name,
                            }
                        call_index = chunk.content_block.id
                
                if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "type") and chunk.delta.type == "input_json_delta":
                    self._functions[call_index]['arguments'] += chunk.delta.partial_json

    
    def get_text(self) -> str:
        return "".join(self.iter_tokens())
    
    def get_function_call(self) -> Optional[List[Dict[str, str]]]:
        if self._functions:
            return self._functions.values()
        return None
    
class AnthropicCompletionStrategy(CompletionStrategy):
    """Strategy for handling Anthropic completions"""
    
    def __init__(self, client: Any):
        self.client = client
        self._capabilities = {
            CompletionCapability.STREAMING,
            CompletionCapability.FUNCTION_CALLING,
            CompletionCapability.SYSTEM_MESSAGES
        }
    
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> CompletionResponse:
        response = self.client.messages.create(
            messages=messages,
            stream=stream,
            max_tokens=1000,
            **kwargs
        )
        
        metadata = CompletionMetadata(
            model=kwargs.pop("model", None),
            created=response.created if not stream else None,
            provider="anthropic",
            stream=stream,
            capabilities=list(self._capabilities)
        )
        
        return AnthropicCompletionStream(response, metadata) if stream else AnthropicCompletionResponse(response, metadata)
    
    def supports_capability(self, capability: CompletionCapability) -> bool:
        return capability in self._capabilities