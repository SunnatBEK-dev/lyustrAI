from hashlib import sha256

import pytest

import ai_sdk.application.bootstrap as bootstrap_module
from ai_sdk.agents import (
    AgentModelResponse,
    AgentRunner,
    AgentTextBlock,
    AgentWorker,
    DependencyHandoffCoordinator,
    HandoffOutputFormat,
    MultiModelRoute,
    WorkflowProgressEvent,
    WorkflowProgressStatus,
)
from ai_sdk.application import AssistantMode
from ai_sdk.application.bootstrap import (
    create_adaptive_multi_model_manager,
    create_rag_manager,
    create_single_model_manager,
)
from ai_sdk.llm.adaptive_multi_model import (
    AdaptiveMultiModelClient,
    MultiModelWorkflowClient,
)
from ai_sdk.llm.base import BaseToolLLMClient
from ai_sdk.tools import ToolExecutor, ToolRegistry
from app.cli import (
    load_document,
    print_adaptive_metrics,
    print_adaptive_progress,
    run_cli,
    select_assistant_mode,
    select_single_model_provider,
)


def test_load_document_uses_stable_path_identity(tmp_path):
    file_path = tmp_path / "Python guide.txt"
    file_path.write_text(
        "Python functions",
        encoding="utf-8",
    )

    first = load_document(str(file_path))
    file_path.write_text(
        "Updated Python functions",
        encoding="utf-8",
    )
    second = load_document(str(file_path))

    assert first.id == second.id
    assert first.id.startswith("doc_")
    assert first.content == "Python functions"
    assert second.content == "Updated Python functions"
    assert second.metadata == {
        "source": str(file_path.resolve()),
        "format": "txt",
        "content_hash": sha256(b"Updated Python functions").hexdigest(),
    }


def test_load_document_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document(str(tmp_path / "missing.txt"))


def test_load_document_rejects_non_utf8_file(tmp_path):
    file_path = tmp_path / "binary.txt"
    file_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        load_document(str(file_path))


def test_cli_selects_application_mode_after_invalid_choice(capsys):
    choices = iter(["unknown", "2"])

    mode = select_assistant_mode(lambda _: next(choices))

    assert mode is AssistantMode.ADAPTIVE_MULTI_MODEL
    assert "Invalid mode" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("selection", "provider"),
    [
        ("1", "anthropic"),
        ("Claude", "anthropic"),
        ("2", "openai"),
        ("GPT", "openai"),
        ("3", "gemini"),
        ("Gemini", "gemini"),
    ],
)
def test_cli_selects_single_model_provider(selection, provider):
    assert select_single_model_provider(lambda _: selection) == provider


def test_cli_reprompts_for_invalid_provider(capsys):
    choices = iter(["other", "openai"])

    provider = select_single_model_provider(lambda _: next(choices))

    assert provider == "openai"
    assert "Invalid provider" in capsys.readouterr().out


class NonStreamingManager:
    def __init__(self):
        self.prompts = []
        self.last_citations = ()

    def send_message(self, prompt):
        self.prompts.append(prompt)
        return "Combined answer"


def test_cli_can_run_non_streaming_adaptive_mode(capsys):
    manager = NonStreamingManager()
    commands = iter(["Question", "/exit"])

    run_cli(
        manager,
        input_fn=lambda _: next(commands),
        ingestor=object(),
        title="Adaptive Multi-Model",
        stream_responses=False,
    )

    output = capsys.readouterr().out
    assert "Adaptive Multi-Model" in output
    assert "Assistant: Combined answer" in output
    assert manager.prompts == ["Question"]


def test_cli_explains_metrics_availability(capsys):
    manager = NonStreamingManager()

    print_adaptive_metrics(manager)

    assert "available in Adaptive Multi-Model mode" in capsys.readouterr().out


def test_cli_prints_safe_adaptive_progress(capsys):
    print_adaptive_progress(
        WorkflowProgressEvent(
            1,
            WorkflowProgressStatus.ROUTE_SELECTED,
            "reasoning",
            0,
            2,
        )
    )
    print_adaptive_progress(
        WorkflowProgressEvent(
            2,
            WorkflowProgressStatus.STAGE_STARTED,
            "reasoning",
            0,
            2,
            "reasoning",
        )
    )
    print_adaptive_progress(
        WorkflowProgressEvent(
            3,
            WorkflowProgressStatus.WORKFLOW_COMPLETED,
            "reasoning",
            2,
            2,
        )
    )

    output = capsys.readouterr().out
    assert "REASONING (2 stages)" in output
    assert "reasoning: started" in output
    assert "workflow: completed" in output


def test_manager_builder_rejects_provider_and_explicit_client():
    with pytest.raises(ValueError, match="either"):
        create_rag_manager(provider="openai", client=object())

    with pytest.raises(TypeError, match="BaseLLMClient"):
        create_rag_manager(client=object())


def test_single_model_builder_uses_normalized_provider_history(
    monkeypatch,
    tmp_path,
):
    received = {}
    expected = object()

    def fake_create_rag_manager(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        bootstrap_module,
        "create_rag_manager",
        fake_create_rag_manager,
    )
    conversation_path = tmp_path / "openai.json"

    result = create_single_model_manager(
        " OpenAI ",
        conversation_file=conversation_path,
    )

    assert result is expected
    assert received == {
        "provider": "openai",
        "conversation_file": conversation_path,
        "runtime": None,
    }


class WorkerClient(BaseToolLLMClient):
    def ask(self, messages):
        return "done"

    def stream(self, messages):
        yield "done"

    def complete_tool_turn(self, messages, schemas, events):
        return AgentModelResponse([AgentTextBlock("done")])


def test_adaptive_builder_configures_three_provider_stages(
    monkeypatch,
    tmp_path,
):
    providers = []
    received = {}
    expected = object()

    retry_policies = []

    def fake_worker(
        name,
        description,
        provider,
        *,
        retry_policy,
    ):
        providers.append(provider)
        retry_policies.append(retry_policy)
        return AgentWorker(
            name,
            description,
            AgentRunner(
                WorkerClient(),
                ToolExecutor(ToolRegistry()),
            ),
            provider=provider,
        )

    def fake_create_rag_manager(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        bootstrap_module,
        "create_provider_worker",
        fake_worker,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "create_rag_manager",
        fake_create_rag_manager,
    )
    conversation_path = tmp_path / "adaptive.json"

    result = create_adaptive_multi_model_manager(conversation_file=conversation_path)

    assert result is expected
    assert providers == ["gemini", "anthropic", "openai"]
    assert len({id(policy) for policy in retry_policies}) == 1
    assert retry_policies[0].max_attempts == 3
    assert received["conversation_file"] == conversation_path
    adaptive_client = received["client"]
    assert isinstance(adaptive_client, AdaptiveMultiModelClient)
    assert set(adaptive_client.workflows) == set(MultiModelRoute)
    assert {
        route: len(client.workflow.stages)
        for route, client in adaptive_client.workflows.items()
    } == {
        MultiModelRoute.FAST: 1,
        MultiModelRoute.CONTEXT: 2,
        MultiModelRoute.REASONING: 2,
        MultiModelRoute.FULL: 3,
    }
    full = adaptive_client.workflows[MultiModelRoute.FULL]
    assert isinstance(full, MultiModelWorkflowClient)
    assert isinstance(
        full.workflow,
        DependencyHandoffCoordinator,
    )
    assert [stage.output_format for stage in full.workflow.stages] == [
        HandoffOutputFormat.STRUCTURED,
        HandoffOutputFormat.STRUCTURED,
        HandoffOutputFormat.TEXT,
    ]
    assert [stage.depends_on for stage in full.workflow.stages] == [
        (),
        ("context",),
        ("context", "reasoning"),
    ]
