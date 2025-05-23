import asyncio
import json
import logging
import warnings
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from autogen_core import CancellationToken, Component, ComponentModel, FunctionCall
from autogen_core.memory import Memory
from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelFamily,
    SystemMessage,
)
from autogen_core.tools import BaseTool, FunctionTool, StaticWorkbench, Workbench
from pydantic import BaseModel
from typing_extensions import Self

from ._assistant_agent import AssistantAgent
from .. import EVENT_LOGGER_NAME
from ..base import Handoff as HandoffBase
from ..base import Response
from ..messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    StructuredMessage,
    StructuredMessageFactory,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from ..state import AssistantAgentState
from ..utils import remove_images
from ._base_chat_agent import BaseChatAgent

class DebateAssistant(AssistantAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        tools: List[BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
        workbench: Workbench | None = None,
        handoffs: List[HandoffBase | str] | None = None,
        model_context: ChatCompletionContext | None = None,
        description: str = "An agent that provides assistance with ability to use tools.",
        system_message: (
            str | None
        ) = "You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed.",
        model_client_stream: bool = False,
        reflect_on_tool_use: bool | None = None,
        tool_call_summary_format: str = "{result}",
        output_content_type: type[BaseModel] | None = None,
        output_content_type_format: str | None = None,
        memory: Sequence[Memory] | None = None,
        metadata: Dict[str, str] | None = None,
    ):
        super().__init__(name, 
            model_client, 
            tools=tools, 
            workbench=workbench, 
            handoffs=handoffs, 
            model_context=model_context, 
            description=description, 
            system_message=system_message, 
            model_client_stream=model_client_stream, 
            reflect_on_tool_use=reflect_on_tool_use, 
            tool_call_summary_format=tool_call_summary_format, 
            output_content_type=output_content_type, 
            output_content_type_format=output_content_type_format, 
            memory=memory, 
            metadata=metadata)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        print("\n\nDebateAssistant on_messages_stream\n\n")
        async for event in super().on_messages_stream(messages, cancellation_token):
            yield event
        
