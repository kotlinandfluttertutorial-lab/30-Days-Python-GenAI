# Day 26 — Concepts
## FastAPI AI Backend

---

## 1. Why FastAPI for AI?

```
SYNC frameworks (Django, Flask):
  Thread-per-request → blocked during LLM calls → 10 threads = 10 concurrent requests
  
FASTAPI (async):
  Event loop → non-blocking during LLM calls → 1 thread handles thousands of concurrent requests
  + Pydantic validation (same as LLM structured output)
  + Auto-generated OpenAPI docs
  + Type hints → IDE autocomplete + runtime validation
  + Dependency injection → auth, rate limiting, DB sessions
```

---

## 2. Complete AI Backend Structure

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import AsyncGenerator
import asyncio
import time

app = FastAPI(
    title="AI Engineering API",
    description="Production AI backend — Day 26",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 3. Pydantic Models for AI APIs

```python
from pydantic import BaseModel, Field, field_validator

class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"user", "assistant", "system"}:
            raise ValueError(f"role must be user/assistant/system, got: {v}")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt-4o-mini"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def at_least_one_message(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: float


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True
    filter_category: str | None = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    latency_ms: float
    cached: bool = False
```

---

## 4. Dependency Injection

```python
from fastapi import Depends, HTTPException, Header
from functools import lru_cache

# Auth dependency
async def get_api_key(x_api_key: str = Header(...)) -> str:
    """Validate API key from header."""
    valid_keys = {"secret-key-1", "secret-key-2"}  # In production: database lookup
    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


# Rate limiting dependency
from collections import defaultdict
import time

_rate_limits: dict[str, list[float]] = defaultdict(list)

async def check_rate_limit(
    api_key: str = Depends(get_api_key),
    requests_per_minute: int = 60,
) -> str:
    now = time.time()
    window_start = now - 60
    _rate_limits[api_key] = [t for t in _rate_limits[api_key] if t > window_start]

    if len(_rate_limits[api_key]) >= requests_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

    _rate_limits[api_key].append(now)
    return api_key


# DB session dependency
from contextlib import asynccontextmanager

@lru_cache
def get_settings():
    from dotenv import load_dotenv
    load_dotenv()
    return {"api_key": os.getenv("OPENAI_API_KEY")}

# Usage in endpoint:
# @app.post("/chat")
# async def chat(request: ChatRequest, key: str = Depends(check_rate_limit)):
```

---

## 5. Streaming Responses

```python
from fastapi.responses import StreamingResponse

async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    """Async generator for streaming LLM tokens."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            # Server-Sent Events format
            yield f"data: {delta.content}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(check_rate_limit),
) -> StreamingResponse:
    """Stream chat response token by token."""
    last_user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )

    return StreamingResponse(
        stream_llm_response(last_user_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
```

---

## 6. Complete Endpoint Example

```python
import uuid
import time
import logging
from fastapi import Request

logger = logging.getLogger(__name__)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    api_key: str = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Chat completion endpoint.
    Validates input → calls LLM → returns structured response.
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    logger.info(
        "Chat request",
        extra={
            "request_id": request_id,
            "model": request.model,
            "message_count": len(request.messages),
            "client_ip": http_request.client.host if http_request.client else "unknown",
        }
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        response = await client.chat.completions.create(
            model=request.model,
            messages=[m.dict() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        logger.info(
            "Chat response",
            extra={
                "request_id": request_id,
                "latency_ms": latency_ms,
                "output_tokens": response.usage.completion_tokens,
            }
        )

        return ChatResponse(
            id=request_id,
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",  # Never expose raw error to client
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
```
