from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from ai_sdk.config import WEB_HOST, WEB_PORT
from ai_sdk.web.conversations import ConversationNotFoundError
from ai_sdk.web.models import (
    AssistantStatusResponse,
    ChatStreamRequest,
    ConversationDetailResponse,
    ConversationRenameRequest,
    ConversationSummaryResponse,
    DocumentUploadResponse,
    IndexedDocumentResponse,
    RunCancellationResponse,
)
from ai_sdk.web.service import (
    ActiveRunError,
    KnowledgeAssistantService,
    UploadValidationError,
)

WEB_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=WEB_ROOT / "templates")


def create_app(
    service: KnowledgeAssistantService | None = None,
) -> FastAPI:
    application = FastAPI(
        title="LyustrAI",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    application.state.assistant_service = service or KnowledgeAssistantService()
    application.mount(
        "/static",
        StaticFiles(directory=WEB_ROOT / "static"),
        name="static",
    )

    @application.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"upload_limit_mb": 10},
        )

    @application.get("/api/status", response_model=AssistantStatusResponse)
    async def assistant_status(request: Request) -> dict[str, object]:
        return _assistant_service(request).status()

    @application.get(
        "/api/documents",
        response_model=list[IndexedDocumentResponse],
    )
    async def list_documents(request: Request) -> list[dict[str, object]]:
        return _assistant_service(request).documents()

    @application.post(
        "/api/documents",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
    ) -> dict[str, object]:
        assistant_service = _assistant_service(request)
        content = await _read_upload(file, assistant_service.upload_max_bytes)
        try:
            document = await run_in_threadpool(
                assistant_service.index_upload,
                file.filename or "",
                content,
            )
        except UploadValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        finally:
            await file.close()
        return {"document": document}

    @application.delete(
        "/api/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_document(document_id: str, request: Request) -> None:
        deleted = await run_in_threadpool(
            _assistant_service(request).delete_document,
            document_id,
        )
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
        return None

    @application.post("/api/chat/stream")
    async def chat_stream(
        payload: ChatStreamRequest,
        request: Request,
    ) -> StreamingResponse:
        try:
            run = _assistant_service(request).start_chat(payload)
        except ActiveRunError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (KeyError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        return StreamingResponse(
            _sse_events(run.events),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "X-Run-ID": run.run_id,
                "X-Conversation-ID": run.conversation_id,
            },
        )

    @application.get(
        "/api/conversations",
        response_model=list[ConversationSummaryResponse],
    )
    async def list_conversations(request: Request) -> list[dict[str, object]]:
        return _assistant_service(request).list_conversations()

    @application.get(
        "/api/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
    )
    async def get_conversation(
        conversation_id: str,
        request: Request,
    ) -> dict[str, object]:
        try:
            return _assistant_service(request).get_conversation(conversation_id)
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @application.patch(
        "/api/conversations/{conversation_id}",
        response_model=ConversationSummaryResponse,
    )
    async def rename_conversation(
        conversation_id: str,
        payload: ConversationRenameRequest,
        request: Request,
    ) -> dict[str, object]:
        try:
            return await run_in_threadpool(
                _assistant_service(request).rename_conversation,
                conversation_id,
                payload.title,
            )
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ActiveRunError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.delete(
        "/api/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_conversation(
        conversation_id: str,
        request: Request,
    ) -> None:
        try:
            await run_in_threadpool(
                _assistant_service(request).delete_conversation,
                conversation_id,
            )
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ActiveRunError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return None

    @application.post(
        "/api/runs/{run_id}/cancel",
        response_model=RunCancellationResponse,
    )
    async def cancel_run(
        run_id: str,
        request: Request,
    ) -> dict[str, object]:
        accepted = _assistant_service(request).cancel(run_id)
        return {"run_id": run_id, "accepted": accepted}

    return application


def _assistant_service(request: Request) -> KnowledgeAssistantService:
    service = request.app.state.assistant_service
    if not isinstance(service, KnowledgeAssistantService):
        raise RuntimeError("Knowledge assistant service is invalid.")
    return service


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    content = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > limit:
            await file.close()
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded document exceeds 10 MB.",
            )
    return bytes(content)


def _sse_events(
    events: Iterator[dict[str, object]],
) -> Iterator[str]:
    try:
        for item in events:
            event_type = item.get("event")
            data = item.get("data")
            if not isinstance(event_type, str):
                continue
            serialized = json.dumps(
                data,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield f"event: {event_type}\ndata: {serialized}\n\n"
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
