import asyncio
import json
import re
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Literal, Mapping, Sequence, Set

from autogen_core import AgentRuntime, CancellationToken, Component, ComponentModel, event, rpc, MessageContext, DefaultTopicId
from autogen_core.models import (
    FunctionExecutionResultMessage,
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)
from typing_extensions import Self

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import ChatAgent, OrTerminationCondition, Response, TerminationCondition
from autogen_agentchat.conditions import StopMessageTermination
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ChatMessage,
    MessageFactory,
    StopMessage,
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    ToolCallSummaryMessage,
    MultiModalMessage,
    UserMessage,
    HandoffMessage,
)
from .._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatReset,
    GroupChatStart,
    GroupChatTermination,   
    SerializableException,
)
from .... import TRACE_LOGGER_NAME
from ._prompts import (
    LEDGER_DIGRAPH_FINAL_ANSWER_PROMPT,
    LEDGER_DIGRAPH_PROGRESS_LEDGER_PROMPT,
    LEDGER_DIGRAPH_TASK_LEDGER_FACTS_PROMPT,
    LEDGER_DIGRAPH_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    LEDGER_DIGRAPH_TASK_LEDGER_FULL_PROMPT,
    LEDGER_DIGRAPH_TASK_LEDGER_PLAN_PROMPT,
    LEDGER_DIGRAPH_TASK_LEDGER_PLAN_UPDATE_PROMPT,
)
from autogen_agentchat.state import BaseGroupChatManagerState
from autogen_agentchat.teams import BaseGroupChat

from ..._group_chat._base_group_chat_manager import BaseGroupChatManager
from ..._group_chat._events import GroupChatTermination

from ._digraph_group_chat import GraphFlow, GraphFlowConfig, DiGraph, GraphFlowManager

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

@dataclass 
class LedgerItem:
    agent_name: str
    task: str


class LedgerGraphFlowManager(GraphFlowManager):
    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        graph: DiGraph,
        planning_agent: ChatAgent,
        model_client: ChatCompletionClient,
        max_task_turns: int
    ) -> None:
        super().__init__(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            max_turns=max_turns,
            message_factory=message_factory,
            graph=graph
        )
        self._planning_agent=planning_agent
        self._model_client = model_client
        self._ledger: deque[LedgerItem] = deque()
        self._team_description = ""
        self._max_stall_turns = max_task_turns
        self._turns_count = 0
        for topic_type, description in zip(self._participant_names, self._participant_descriptions, strict=True):
            self._team_description += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
        self._team_description = self._team_description.strip()
    
    def _get_progress_ledger_prompt(self, task: str, team: str) -> str:
        return LEDGER_DIGRAPH_PROGRESS_LEDGER_PROMPT.format(task=task, team=team)
    
    async def _log_message(self, log_message: str) -> None:
        trace_logger.debug(log_message)
    
    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:
        if self._termination_condition is not None and self._termination_condition.terminated:
            early_stop_message = StopMessage(
                content="The group chat has already terminated.",
                source=self._name,
            )
            # Signal termination to the caller of the team.
            await self._signal_termination(early_stop_message)
            # Stop the group chat.
            return

        # Validate the group state given the start messages
        await self.validate_group_state(message.messages)

        if message.messages is not None:
            # Log all messages at once
            await self.publish_message(
                GroupChatStart(messages=message.messages),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            for msg in message.messages:
                await self._output_message_queue.put(msg)

            # # Relay all messages at once to participants
            # await self.publish_message(
            #     GroupChatStart(messages=message.messages),
            #     topic_id=DefaultTopicId(type=self._group_topic_type),
            #     cancellation_token=ctx.cancellation_token,
            # )
            planner_topic_type = self._participant_name_to_topic_type[self._planning_agent.name]
            await self.publish_message(
                GroupChatStart(messages=message.messages),
                topic_id=DefaultTopicId(type=planner_topic_type),
                cancellation_token=ctx.cancellation_token,
            )

            # Append all messages to thread
            await self.update_message_thread(message.messages)

            # Check termination condition after processing all messages
            if await self._apply_termination_condition(message.messages):
                # Stop the group chat.
                return

        # Select speakers to start/continue the conversation
        await self._next_step(ctx.cancellation_token)

    @event
    async def handle_agent_response(self, message: GroupChatAgentResponse, ctx: MessageContext) -> None:  # type: ignore
        try:
            # Update _message_thread
            delta: List[BaseAgentEvent | BaseChatMessage] = []
            if message.agent_response.inner_messages is not None:
                for inner_message in message.agent_response.inner_messages:
                    delta.append(inner_message)
            delta.append(message.agent_response.chat_message)
            await self.update_message_thread(delta)
            if not message.agent_name == self._planning_agent.name:
                await self.publish_message(
                    GroupChatAgentResponse(agent_response=message.agent_response, agent_name=message.agent_name),
                    topic_id=DefaultTopicId(type=self._participant_name_to_topic_type[self._planning_agent.name]),
                    cancellation_token=ctx.cancellation_token,
                )


            self._active_speakers.remove(message.agent_name)
            if len(self._active_speakers) > 0:
                # If there are still active speakers, return without doing anything.
                return

            if self._termination_condition is not None:
                stop_message = await self._termination_condition(delta)
                if stop_message is not None:
                    # Reset the termination conditions.
                    await self._termination_condition.reset()
                    # Signal termination.
                    await self._signal_termination(stop_message)
                    return
                
            # Update ledger
            if message.agent_name == self._planning_agent.name:
                await self.add_ledger()
                self._turns_count = 0
            else:
                await self.update_ledger()
                self._turns_count += 1
            
            # Select speakers to continue the conversation.
            await self._next_step(ctx.cancellation_token)
        except Exception as e:
            error = SerializableException.from_exception(e)
            await self._signal_termination_with_error(error)
            # Raise the error to the runtime.
            raise
    
    async def _next_step(self, cancellation_token: CancellationToken) -> None:
        speakers: List[str] = []
        # 中断当前流程，强制切换为Planning Agent
        if self._turns_count >= self._max_stall_turns:
            speakers.append(self._planning_agent.name)
            self._ledger.clear()
            task_message = TextMessage(content="Something went wrong during the task. Check the history and replan the task.", source="user")
            await self.publish_message(
                GroupChatAgentResponse(agent_response=Response(chat_message=task_message), agent_name="user"),
                topic_id=DefaultTopicId(type=self._participant_name_to_topic_type[self._planning_agent.name]),
                cancellation_token=cancellation_token,
            )   
        else:
            # 当且仅当ledger为空时，选择Planning Agent
            if self._ledger:
                speakers.append(self._ledger[0].agent_name)
            else:
                speakers.append(self._planning_agent.name)
                # # Drain the ready queue for the next set of speakers.
                # while self._ready:
                #     speaker = self._ready.popleft()
                #     speakers.append(speaker)
                #     # Reset the bookkeeping for the node that were selected.
                #     if self._activation[speaker] == "any":
                #         self._enqueued_any[speaker] = False
                #     else:
                #         self._remaining[speaker] = len(self._parents[speaker])
            
        # speaker_names_future = asyncio.ensure_future(self.select_speaker(self._message_thread))
        # # Link the select speaker future to the cancellation token.
        # cancellation_token.link_future(speaker_names_future)
        # speaker_names = await speaker_names_future

        for speaker_name in speakers:
            if speaker_name not in self._participant_name_to_topic_type:
                raise RuntimeError(f"Speaker {speaker_name} not found in participant names.")
        await self._log_speaker_selection(speakers)

        # Send request to publish message to the next speakers
        for speaker_name in speakers:
            speaker_topic_type = self._participant_name_to_topic_type[speaker_name]
            if self._ledger:
                task_message = TextMessage(content=self._ledger[0].task, source="user")
                await self.publish_message(
                    GroupChatAgentResponse(agent_response=Response(chat_message=task_message), agent_name=self._planning_agent.name),
                    topic_id=DefaultTopicId(type=speaker_topic_type),
                    cancellation_token=cancellation_token,
                )
            # 唤起 next speaker
            await self.publish_message(
                GroupChatRequestPublish(),
                topic_id=DefaultTopicId(type=speaker_topic_type),
                cancellation_token=cancellation_token,
            )
            self._active_speakers.append(speaker_name)
    
    async def add_ledger(self):
        # 只取倒序第一个planning agent的规划更新ledger
        pattern = r"TASK:\s*([^:]+):\s*(.+?)(?=\s*\{TASK_DELIMITER\}|\s*TASK:|$|\s*USER_INPUT:)"
        for message in reversed(self._message_thread):
            if isinstance(message, BaseChatMessage) and message.source == self._planning_agent.name:
                if "USER_INPUT" in message.content:
                    self._ledger.append(LedgerItem("user_proxy", ""))
                    break
                else:
                    match = re.findall(pattern, message.content, re.DOTALL)
                    for agent_name, task in match:
                        self._ledger.append(LedgerItem(agent_name, task))
                    break
    
    async def update_message_thread(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> None:
        self._message_thread.extend(messages)

        # Update the graph
        message = messages[-1]
        if message.source not in self._graph.nodes:
            # Ignore messages from sources outside of the graph.
            return
        assert isinstance(message, BaseChatMessage)
        source = message.source
        content = message.to_model_text()
        # Propagate the update to the children of the node.
        for edge in self._edges[source]:
            if edge.condition and edge.condition not in content:
                continue
            if self._activation[edge.target] == "all":
                self._remaining[edge.target] -= 1
                if self._remaining[edge.target] == 0:
                    # If all parents are done, add to the ready queue.
                    self._ready.append(edge.target)
            else:
                # If activation is any, add to the ready queue if not already enqueued.
                if not self._enqueued_any[edge.target]:
                    self._ready.append(edge.target)
                    self._enqueued_any[edge.target] = True
    
    async def update_ledger(self):
        # 调用model_client，获取task的完成情况
        context = self._thread_to_context()
        ledger_item = self._ledger.popleft()
        if ledger_item.agent_name == "user_proxy":
            return 
        task = ledger_item.agent_name + ": " + ledger_item.task
        progress_ledger_prompt = self._get_progress_ledger_prompt(
            task, self._team_description
        )

        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        # 判断task是否完成
        response = await self._model_client.create(context, json_output=True)
        response_str = response.content
        try:
            response_dict = json.loads(response_str)
            if not isinstance(response_dict["completed"]["answer"], bool):
                raise ValueError("LLM output is not a JSON object.")
            if response_dict["completed"]["answer"]:
                # Task 完成，删除ledger
                pass
            else:
                # Task 未完成，左向置入deque
                self._ledger.appendleft(LedgerItem(ledger_item.agent_name, ledger_item.task))

        except (json.JSONDecodeError, TypeError):
            await self._log_message("Invalid ledger format encountered, retrying...")

    def _thread_to_context(self) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in self._message_thread:
            if isinstance(m, ToolCallExecutionEvent):
                context.append(FunctionExecutionResultMessage(content=m.content, source=m.source))
            elif isinstance(m, ToolCallRequestEvent):
                context.append(AssistantMessage(content=m.content, source=m.source))
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == "user":
                if not isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage)):
                    print(f"\n\nUnexpected message type: {type(m)}\n\n")
                    continue
                context.append(UserMessage(content=m.content, source=m.source))
            else:
                if not isinstance(m, (TextMessage, ToolCallSummaryMessage)):
                    print(f"\n\nUnexpected message type: {type(m)}\n\n")
                    continue
                context.append(AssistantMessage(content=m.content, source=m.source))
        return context

class LedgerGraphFlow(GraphFlow, Component[GraphFlowConfig]):

    component_config_schema = GraphFlowConfig
    component_provider_override = "autogen_agentchat.teams.LedgerGraphFlow"

    def __init__(
        self,
        participants: List[ChatAgent],
        planning_agent: ChatAgent,
        model_client: ChatCompletionClient,
        graph: DiGraph,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
        max_task_turns: int = 3
    ) -> None:
        self._planning_agent = planning_agent
        self._model_client = model_client
        self._max_task_turns = max_task_turns
        super().__init__(
            participants=participants,
            graph=graph,
            group_chat_manager_name="LedgerGraphManager",
            group_chat_manager_class=LedgerGraphFlowManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
            custom_message_types=custom_message_types
        )


    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
    ) -> Callable[[], LedgerGraphFlowManager]:
        
        def _factory() -> LedgerGraphFlowManager:
            return LedgerGraphFlowManager(
                name=name,
                group_topic_type=group_topic_type,
                output_topic_type=output_topic_type,
                participant_topic_types=participant_topic_types,
                participant_names=participant_names,
                participant_descriptions=participant_descriptions,
                output_message_queue=output_message_queue,
                termination_condition=termination_condition,
                max_turns=max_turns,
                message_factory=message_factory,
                graph=self._graph,
                planning_agent=self._planning_agent,
                model_client=self._model_client,
                max_task_turns=self._max_task_turns
            )

        return _factory

    def _to_config(self) -> GraphFlowConfig:
        """Converts the instance into a configuration object."""
        participants = [participant.dump_component() for participant in self._input_participants]
        termination_condition = (
            self._input_termination_condition.dump_component() if self._input_termination_condition else None
        )
        return GraphFlowConfig(
            participants=participants,
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            graph=self._graph,
        )

    @classmethod
    def _from_config(cls, config: GraphFlowConfig) -> Self:
        """Reconstructs an instance from a configuration object."""
        participants = [ChatAgent.load_component(participant) for participant in config.participants]
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition) if config.termination_condition else None
        )
        return cls(
            participants, graph=config.graph, termination_condition=termination_condition, max_turns=config.max_turns
        )

    
