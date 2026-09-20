import pytest

from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    AICapability,
    CapabilityRouter,
    CoordinationError,
    HandoffStage,
    MultiAgentCoordinator,
    MultiModelRoute,
    RoutingDecision,
    RoutingSignal,
    SequentialHandoffCoordinator,
)
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry


@pytest.mark.parametrize(
    ("text", "expected_route"),
    [
        ("Salom", MultiModelRoute.FAST),
        (
            "Ushbu hujjatdagi manbalarni ko'rsat",
            MultiModelRoute.CONTEXT,
        ),
        ("Nega bu yechim ishlaydi?", MultiModelRoute.REASONING),
        (
            "Manbalarni chuqur tahlil qilib taqqosla",
            MultiModelRoute.FULL,
        ),
        (
            "Retrieved context:\nPython facts",
            MultiModelRoute.CONTEXT,
        ),
        (
            "Birinchi savol? Ikkinchi savol?",
            MultiModelRoute.REASONING,
        ),
        (
            "1. Birinchi vazifa\n2. Ikkinchi vazifa",
            MultiModelRoute.REASONING,
        ),
        ("Natijani tekshirib ber", MultiModelRoute.CONTEXT),
        ("Loyihani tahlilini ber", MultiModelRoute.REASONING),
    ],
)
def test_capability_router_selects_explainable_route(
    text,
    expected_route,
):
    decision = CapabilityRouter().route(text)

    assert decision.route is expected_route
    assert decision.capabilities[-1] is AICapability.SYNTHESIS


def test_capability_router_marks_long_request_as_full():
    decision = CapabilityRouter(
        long_request_chars=10,
        max_analyzed_chars=20,
    ).route("x" * 30)

    assert decision.route is MultiModelRoute.FULL
    assert decision.signals == (RoutingSignal.LONG_REQUEST,)
    assert decision.capabilities == (
        AICapability.CONTEXT,
        AICapability.REASONING,
        AICapability.SYNTHESIS,
    )


def test_capability_router_honors_explicit_maximum_override():
    decision = CapabilityRouter(forced_route=MultiModelRoute.FULL).route("Salom")

    assert decision.route is MultiModelRoute.FULL
    assert decision.signals == (RoutingSignal.USER_OVERRIDE,)
    assert decision.estimated_model_requests == 3


def test_routing_decision_and_router_validate_configuration():
    with pytest.raises(TypeError, match="route"):
        RoutingDecision("fast")
    with pytest.raises(TypeError, match="signals"):
        RoutingDecision(MultiModelRoute.FAST, ("invalid",))
    with pytest.raises(CoordinationError, match="unique"):
        RoutingDecision(
            MultiModelRoute.FAST,
            (
                RoutingSignal.LONG_REQUEST,
                RoutingSignal.LONG_REQUEST,
            ),
        )
    with pytest.raises(ValueError, match="positive"):
        CapabilityRouter(long_request_chars=0)
    with pytest.raises(ValueError, match="cannot be below"):
        CapabilityRouter(
            long_request_chars=10,
            max_analyzed_chars=5,
        )
    with pytest.raises(TypeError, match="Forced"):
        CapabilityRouter(forced_route="full")
    with pytest.raises(ValueError, match="cannot be empty"):
        CapabilityRouter().route(" ")


class RouteClient(BaseToolLLMClient):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def ask(self, messages):
        return self.response

    def stream(self, messages):
        yield self.response

    def complete_tool_turn(self, messages, schemas, events):
        self.prompts.append(messages[0]["content"])
        return AgentModelResponse(
            [
                AgentTextBlock(self.response),
            ]
        )


def workflow(route):
    provider = RouteClient(route.value)
    worker = AgentWorker(
        f"{route.value}_worker",
        "Return route response",
        AgentRunner(
            provider,
            ToolExecutor(ToolRegistry()),
        ),
    )
    client = MultiModelWorkflowClient(
        SequentialHandoffCoordinator(
            MultiAgentCoordinator([worker]),
            [
                HandoffStage(
                    "final",
                    worker.name,
                    "Answer",
                )
            ],
        )
    )
    return client, provider


def adaptive_client():
    clients = {}
    providers = {}
    for route in MultiModelRoute:
        client, provider = workflow(route)
        clients[route] = client
        providers[route] = provider
    return (
        AdaptiveMultiModelClient(CapabilityRouter(), clients),
        providers,
    )


@pytest.mark.parametrize(
    ("text", "expected_route"),
    [
        ("Salom", MultiModelRoute.FAST),
        ("Manbani ko'rsat", MultiModelRoute.CONTEXT),
        ("Nega shunday?", MultiModelRoute.REASONING),
        ("Manbani tahlil qil", MultiModelRoute.FULL),
    ],
)
def test_adaptive_client_executes_only_selected_workflow(
    text,
    expected_route,
):
    client, providers = adaptive_client()

    response = client.ask(
        [
            {
                "role": "user",
                "content": text,
            }
        ]
    )

    assert response == expected_route.value
    assert client.last_decision.route is expected_route
    assert client.last_result.completed is True
    assert providers[expected_route].prompts
    assert all(
        not provider.prompts
        for route, provider in providers.items()
        if route is not expected_route
    )


def test_adaptive_client_validates_workflows_and_messages():
    client, _ = adaptive_client()
    valid_workflows = client.workflows

    with pytest.raises(TypeError, match="router"):
        AdaptiveMultiModelClient(object(), valid_workflows)
    with pytest.raises(TypeError, match="mapping"):
        AdaptiveMultiModelClient(CapabilityRouter(), [])
    invalid_keys = dict(valid_workflows)
    fast_workflow = invalid_keys.pop(MultiModelRoute.FAST)
    invalid_keys["fast"] = fast_workflow
    with pytest.raises(TypeError, match="keys"):
        AdaptiveMultiModelClient(CapabilityRouter(), invalid_keys)
    with pytest.raises(ValueError, match="every route"):
        AdaptiveMultiModelClient(
            CapabilityRouter(),
            {MultiModelRoute.FAST: valid_workflows[MultiModelRoute.FAST]},
        )
    invalid_workflows = dict(valid_workflows)
    invalid_workflows[MultiModelRoute.FAST] = object()
    with pytest.raises(TypeError, match="MultiModelWorkflowClient"):
        AdaptiveMultiModelClient(
            CapabilityRouter(),
            invalid_workflows,
        )
    with pytest.raises(RuntimeError, match="streaming"):
        client.stream([{"role": "user", "content": "Hi"}])
    with pytest.raises(ValueError, match="cannot be empty"):
        client.ask([])
    with pytest.raises(ValueError, match="requires a user"):
        client.ask([{"role": "assistant", "content": "Hi"}])
    with pytest.raises(TypeError, match="content"):
        client.ask([{"role": "user", "content": None}])
