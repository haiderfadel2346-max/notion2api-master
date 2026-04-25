import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# ================================
# Tool / Function Calling Schema
# ================================

class FunctionDefinition(BaseModel):
    """OpenAI function definition"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    """OpenAI tool definition"""
    type: str = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    """A function call in a tool_call"""
    name: str
    arguments: str  # JSON string

class ToolCall(BaseModel):
    """A tool call from the assistant"""
    id: str
    type: str = "function"
    function: FunctionCall

# ================================
# 请求相关 Schema (Chat Completion)
# ================================

class ChatMessage(BaseModel):
    """单条对话消息"""
    role: Literal["user", "assistant", "system", "tool", "function"]
    content: Optional[str] = None
    thinking: Optional[str] = None
    # Tool calling fields
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    """
    OpenAI-Compatible 发起完成请求的 Payload。
    保留 `conversation_id` 作为特定的扩展字段，若缺失则视为独立请求。
    """
    model: str = Field(default="claude-opus4.6", description="Requested model.")
    messages: List[ChatMessage]
    stream: bool = Field(default=False, description="Whether to stream the response as SSE.")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature.")
    conversation_id: Optional[str] = Field(default=None, description="Extension for stateful conversation tracking.")
    # Tool calling fields
    tools: Optional[List[ToolDefinition]] = Field(default=None, description="List of tools the model may call.")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Controls tool usage.")

# ================================
# 非流式返回 Schema
# ================================

class ChatMessageResponse(BaseModel):
    """Response message (supports tool_calls)"""
    role: str = "assistant"
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    model_config = {"json_schema_extra": {"exclude_none": True}}

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def dict(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().dict(**kwargs)

class ChatMessageResponseChoice(BaseModel):
    """非流式响应的选项"""
    index: int = 0
    message: Union[ChatMessage, ChatMessageResponse]
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    """
    OpenAI-Compatible 完整返回 Payload。
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatMessageResponseChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    # Standard 模式扩展字段
    search_metadata: Optional[Dict[str, Any]] = Field(default=None)

# ================================
# 流式返回 Schema (供内部组织)
# ================================

class ToolCallChunkFunction(BaseModel):
    """Function info in a streaming tool call chunk"""
    name: Optional[str] = None
    arguments: Optional[str] = None

class ToolCallChunk(BaseModel):
    """Tool call chunk for streaming"""
    index: int = 0
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[ToolCallChunkFunction] = None

class ChatCompletionChunkDelta(BaseModel):
    """SSE Delta Block"""
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[ToolCallChunk]] = None

class ChatCompletionChunkChoice(BaseModel):
    """SSE Choice Block"""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    """
    OpenAI-Compatible 流式 Chunk
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
