from typing import Literal

from pydantic import BaseModel, Field, model_validator

ProviderName = Literal["anthropic", "openai", "gemini"]
ChatMode = Literal["single", "adaptive"]
AdaptiveStrategy = Literal["auto", "maximum"]


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    mode: ChatMode
    provider: ProviderName | None = None
    adaptive_strategy: AdaptiveStrategy | None = None
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^chat_[0-9a-f]{12}$",
    )

    @model_validator(mode="after")
    def validate_provider(self) -> "ChatStreamRequest":
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("Message cannot be blank.")
        if self.mode == "single" and self.provider is None:
            raise ValueError("Single Model mode requires a provider.")
        if self.mode == "single" and self.adaptive_strategy is not None:
            raise ValueError("Single Model mode does not accept an adaptive strategy.")
        if self.mode == "adaptive" and self.provider is not None:
            raise ValueError(
                "Adaptive Multi-Model mode selects providers automatically."
            )
        if self.mode == "adaptive" and self.adaptive_strategy is None:
            self.adaptive_strategy = "auto"
        return self


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def normalize_title(self) -> "ConversationRenameRequest":
        self.title = " ".join(self.title.split())
        if not self.title:
            raise ValueError("Conversation title cannot be blank.")
        return self


class ConversationMessageResponse(BaseModel):
    id: str
    role: str
    content: str


class ConversationSummaryResponse(BaseModel):
    schema_version: int
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    last_mode: ChatMode
    last_provider: ProviderName | None
    last_adaptive_strategy: AdaptiveStrategy
    message_count: int


class ConversationDetailResponse(ConversationSummaryResponse):
    messages: list[ConversationMessageResponse]


class ProviderReadinessResponse(BaseModel):
    provider: str
    display_name: str
    ready: bool
    missing_variables: list[str]


class AssistantStatusResponse(BaseModel):
    name: str
    version: str
    adaptive_ready: bool
    providers: list[ProviderReadinessResponse]
    document_count: int
    active_run_id: str | None
    adaptive_metrics: dict[str, object]


class IndexedDocumentResponse(BaseModel):
    document_id: str
    source: str
    format: str
    chunk_count: int
    page_count: int | None = None


class DocumentUploadResponse(BaseModel):
    document: IndexedDocumentResponse


class RunCancellationResponse(BaseModel):
    run_id: str
    accepted: bool
