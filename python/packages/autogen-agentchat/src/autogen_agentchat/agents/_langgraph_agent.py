from dataclasses import dataclass
# from typing import Any, Callable, List, Literal
# from ._assistant_agent import BaseChatAgent
from langchain_core.tools import tool  # pyright: ignore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage as LCBaseMessage 
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langchain_core.messages import ChatMessage as LCChatMessage
from langchain_core.messages import ToolMessage as LCToolMessage
# from typing import (
#     Any,
#     Awaitable,
#     Callable,
#     List,
# )
from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime, message_handler

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
    UserMessage
)
from autogen_core.tools import BaseTool, FunctionTool, StaticWorkbench, Workbench
from pydantic import BaseModel
from typing_extensions import Self

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
from ._assistant_agent import AssistantAgent

@dataclass
class Message:
    content: str

@tool  # pyright: ignore
def get_weather(location: str) -> str:
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

class LangGraphAgent(AssistantAgent):
    def __init__(
    self,
    name: str,
    model_client: ChatCompletionClient,
    state_graph: StateGraph,
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
        self._workflow = state_graph
        self._app = self._workflow.compile()
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        # system_messages = self._system_messages
        # workbench = self._workbench
        # handoff_tools = self._handoff_tools
        # handoffs = self._handoffs
        # model_client = self._model_client
        # model_client_stream = self._model_client_stream
        # reflect_on_tool_use = self._reflect_on_tool_use
        # tool_call_summary_format = self._tool_call_summary_format
        # tool_call_summary_formatter = self._tool_call_summary_formatter
        # output_content_type = self._output_content_type
        # format_string = self._output_content_type_format
        
        # STEP 1: Add new user/handoff messages to the model context
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        for event_msg in await self._update_model_context_with_memory(
            memory=memory,
            model_context=model_context,
            agent_name=agent_name,
        ):
            inner_messages.append(event_msg)
            yield event_msg

        # STEP 3: Run the StateGraph
        all_messages = await model_context.get_messages()
        lang_messages = self.covert_to_langchain_message(all_messages)
        final_state = await self._app.ainvoke(
            {
                "messages": lang_messages
            },
            config={"configurable": {"thread_id": 42}},
        )
        final_response_content = final_state["messages"][-1].content
        yield Response(
            chat_message=TextMessage(content=final_response_content, source=agent_name),
            inner_messages=inner_messages,
        )

    def covert_to_langchain_message(self, all_messages: List[LLMMessage]) -> List[LCBaseMessage]:
        res = []
        for msg in all_messages:
            if isinstance(msg, SystemMessage):
                res.append(LCSystemMessage(content=msg.content))
            elif isinstance(msg, UserMessage):
                res.append(LCHumanMessage(content=msg.content))
            elif isinstance(msg, AssistantMessage):
                res.append(LCChatMessage(content=msg.content, role=msg.source))
            elif isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    res.append(LCToolMessage(
                        content=result.content,
                        name = result.name,
                        tool_call_id = result.call_id,
                        status = "error" if result.is_error else "success"
                    ))
        return res
