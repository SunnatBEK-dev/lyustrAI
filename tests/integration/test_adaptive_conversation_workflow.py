import json

import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    CapabilityRouter,
    DependencyHandoffCoordinator,
    HandoffOutputFormat,
    HandoffStage,
    MultiAgentCoordinator,
    MultiModelRoute,
    SequentialHandoffCoordinator,
    WorkflowCancelledError,
    WorkflowProgressStatus,
)
from ai_sdk.application.conversation_manager import ConversationManager
from ai_sdk.context.prompt_builder import PromptBuilder
from ai_sdk.core.conversation import Conversation
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.storage.json import JSONConversationRepository
from ai_sdk.tools import ToolExecutor, ToolRegistry

pytestmark = pytest.mark.integration


class StageClient(BaseToolLLMClient):
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def ask(self, messages):
        return self.output

    def stream(self, messages):
        yield self.output

    def complete_tool_turn(self, messages, schemas, events):
        self.prompts.append(messages[0]["content"])
        return AgentModelResponse(
            [
                AgentTextBlock(self.output),
            ]
        )


def stage_worker(name, client, provider):
    return AgentWorker(
        name,
        f"{name} responsibility",
        AgentRunner(
            client,
            ToolExecutor(ToolRegistry()),
        ),
        provider=provider,
    )


def payload(summary, *, facts=(), recommendations=()):
    return json.dumps(
        {
            "summary": summary,
            "facts": list(facts),
            "uncertainties": [],
            "recommendations": list(recommendations),
        }
    )


def test_adaptive_combines_providers_and_persists_final_answer(
    tmp_path,
):
    gemini = StageClient(
        payload(
            "Extracted context",
            facts=["Extracted facts"],
        )
    )
    claude = StageClient(
        payload(
            "Reasoned solution",
            facts=["Extracted facts"],
            recommendations=["Use the solution"],
        )
    )
    openai = StageClient("Final combined answer")
    workers = [
        stage_worker("context", gemini, "gemini"),
        stage_worker("reasoner", claude, "anthropic"),
        stage_worker("writer", openai, "openai"),
    ]
    workflow = DependencyHandoffCoordinator(
        MultiAgentCoordinator(workers),
        [
            HandoffStage(
                "context",
                "context",
                "Extract facts",
                output_format=HandoffOutputFormat.STRUCTURED,
            ),
            HandoffStage(
                "reason",
                "reasoner",
                "Analyze facts",
                output_format=HandoffOutputFormat.STRUCTURED,
                depends_on=("context",),
            ),
            HandoffStage(
                "final",
                "writer",
                "Write answer",
                depends_on=("context", "reason"),
            ),
        ],
    )
    repository = JSONConversationRepository(tmp_path / "adaptive_conversation.json")
    conversation = Conversation()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=MultiModelWorkflowClient(workflow),
        repository=repository,
    )

    answer = manager.send_message("Solve this problem")
    restored = repository.load()

    assert answer == "Final combined answer"
    assert [message.content for message in restored.history()] == [
        "Solve this problem",
        "Final combined answer",
    ]
    assert "Extracted facts" in claude.prompts[0]
    assert "Reasoned solution" in openai.prompts[0]
    assert "Use the solution" in openai.prompts[0]
    assert "Required dependency handoffs" in openai.prompts[0]


def test_adaptive_client_selects_workflow_and_persists_turns(
    tmp_path,
):
    providers = {route: StageClient(route.value) for route in MultiModelRoute}
    workflows = {}
    for route, provider in providers.items():
        route_worker = stage_worker(
            f"{route.value}_worker",
            provider,
            "openai",
        )
        workflows[route] = MultiModelWorkflowClient(
            SequentialHandoffCoordinator(
                MultiAgentCoordinator([route_worker]),
                [
                    HandoffStage(
                        "final",
                        route_worker.name,
                        "Answer",
                    )
                ],
            )
        )
    client = AdaptiveMultiModelClient(
        CapabilityRouter(),
        workflows,
    )
    repository = JSONConversationRepository(tmp_path / "adaptive_conversation.json")
    conversation = Conversation()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
    )

    fast_answer = manager.send_message("Salom")
    reasoning_answer = manager.send_message("Nega bu yechim ishlaydi?")

    assert fast_answer == "fast"
    assert reasoning_answer == "reasoning"
    assert len(providers[MultiModelRoute.FAST].prompts) == 1
    assert len(providers[MultiModelRoute.REASONING].prompts) == 1
    assert providers[MultiModelRoute.CONTEXT].prompts == []
    assert providers[MultiModelRoute.FULL].prompts == []
    assert [message.content for message in repository.load().history()] == [
        "Salom",
        "fast",
        "Nega bu yechim ishlaydi?",
        "reasoning",
    ]


def test_cancelled_adaptive_turn_is_rolled_back(tmp_path):
    providers = {route: StageClient(route.value) for route in MultiModelRoute}
    workflows = {}
    for route, provider in providers.items():
        route_worker = stage_worker(
            f"{route.value}_worker",
            provider,
            "openai",
        )
        workflows[route] = MultiModelWorkflowClient(
            SequentialHandoffCoordinator(
                MultiAgentCoordinator([route_worker]),
                [
                    HandoffStage(
                        "final",
                        route_worker.name,
                        "Answer",
                    )
                ],
            )
        )

    active = {}

    def cancel_after_route(event):
        if event.status is WorkflowProgressStatus.ROUTE_SELECTED:
            assert active["client"].cancel() is True

    client = AdaptiveMultiModelClient(
        CapabilityRouter(),
        workflows,
        progress_handler=cancel_after_route,
    )
    active["client"] = client
    repository = JSONConversationRepository(
        tmp_path / "cancelled_adaptive_conversation.json"
    )
    conversation = Conversation()
    manager = ConversationManager(
        conversation=conversation,
        prompt_builder=PromptBuilder(conversation),
        client=client,
        repository=repository,
    )

    with pytest.raises(WorkflowCancelledError):
        manager.send_message("Salom")

    assert conversation.is_empty() is True
    assert repository.load().is_empty() is True
    assert all(provider.prompts == [] for provider in providers.values())
    assert client.last_progress_event.status is (
        WorkflowProgressStatus.WORKFLOW_CANCELLED
    )
