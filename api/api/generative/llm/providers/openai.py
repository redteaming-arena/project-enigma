from __future__ import annotations
from typing import Iterable, Optional, Any, Dict, List
from ..types import (CompletionCapability, 
                     CompletionMetadata,
                     CompletionResponse, 
                     FunctionCallable, 
                     CompletionStrategy)



class OpenAICompletionResponse(CompletionResponse["ChatCompletion", str], FunctionCallable[List[Dict[str, str]]]):
    """OpenAI-specific completion response implementation"""
    
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
        
class OpenAICompletionStream(CompletionResponse["Stream[ChatCompletionChunk]", str], FunctionCallable[List[Dict[str, str]]]):
    """OpenAI-specific completion response implementation"""
    
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
        for chunk in self._stream:
            if hasattr(chunk, "choices") and chunk.choices:
                if hasattr(chunk.choices[0], "delta"):
                    if chunk.choices[0].delta.content:
                        self._chunk_response.append(chunk.choices[0].delta.content)
                        at_index += 1
                        yield chunk.choices[0].delta.content
                        while at_index < len(self._chunk_response):
                            yield self._chunk_response[at_index]
                    if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                        for call in chunk.choices[0].delta.tool_calls:
                            if call.index not in self._functions:
                                self._functions[call.index] = {
                                    'arguments': '',
                                    'name': '',
                                }
                            if call.function.arguments is not None:
                                self._functions[call.index]['arguments'] += call.function.arguments
                            if call.function.name is not None:
                                self._functions[call.index]['name'] += call.function.name

    
    def get_text(self) -> str:
        return "".join(self.iter_tokens())
    
    def get_function_call(self) -> Optional[List[Dict[str, str]]]:
        if self._functions:
            return self._functions.values()
        return None
    
class OpenAICompletionStrategy(CompletionStrategy):
    """Strategy for handling OpenAI completions"""
    
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
        
        response = self.client.chat.completions.create(
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        metadata = CompletionMetadata(
            model=kwargs.pop("model", None),
            created=response.created if not stream else None,
            provider="openai",
            stream=stream,
            capabilities=list(self._capabilities)
        )
        
        return OpenAICompletionStream(response, metadata) if stream else OpenAICompletionResponse(response, metadata)
    
    def supports_capability(self, capability: CompletionCapability) -> bool:
        return capability in self._capabilities