"""
Day 26 — AI API Service
========================
Production-quality FastAPI AI backend with:
- Pydantic request/response validation
- API key authentication
- Rate limiting
- Streaming responses (SSE)
- Dependency injection
- Structured logging
- Global error handling
- Health check endpoint

Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_api")


# ─────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"user", "assistant", "system"}:
            raise ValueError(f"role must be user/assistant/system")
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
    def messages_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict[str, int]
    latency_ms: float


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    model: str = "text-embedding-3-small"


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool


# ─────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI API Service...")
    # Check LLM availability
    app.state.llm_available = bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    )
    logger.info(f"LLM available: {app.state.llm_available}")
    yield
    logger.info("Shutting down AI API Service...")


# ─────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Engineering API",
    description="Production AI backend — Day 26 of 30-Day AI Engineer Program",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────────────────

VALID_API_KEYS = {"dev-key-123", "test-key-456"}  # In production: DB lookup

async def authenticate(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate API key. Inject user's key into endpoint."""
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


_rate_store: dict[str, list[float]] = defaultdict(list)

async def rate_limit(api_key: str = Depends(authenticate), rpm: int = 60) -> str:
    """Token bucket rate limiter — 60 requests per minute per API key."""
    now = time.time()
    window_start = now - 60
    recent = [t for t in _rate_store[api_key] if t > window_start]
    _rate_store[api_key] = recent

    if len(recent) >= rpm:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit: {rpm} requests/minute exceeded",
            headers={"Retry-After": "60"},
        )

    _rate_store[api_key].append(now)
    return api_key


# ─────────────────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────────────────

def _get_openai_client():
    """Get OpenAI or Groq client."""
    if groq_key := os.getenv("GROQ_API_KEY"):
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        ), "llama-3.1-8b-instant"
    elif openai_key := os.getenv("OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=openai_key), None
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No LLM API key configured",
    )


# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint — no auth required."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_available=getattr(request.app.state, "llm_available", False),
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_req: Request,
    api_key: str = Depends(rate_limit),
) -> ChatResponse:
    """
    Chat completion endpoint.
    Validates → calls LLM → returns structured response.
    """
    req_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    logger.info(
        f"[{req_id}] Chat | model={request.model} | "
        f"messages={len(request.messages)} | temp={request.temperature}"
    )

    try:
        client, default_model = _get_openai_client()
        model = default_model or request.model

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""

        logger.info(
            f"[{req_id}] Done | "
            f"latency={latency_ms:.0f}ms | "
            f"tokens={response.usage.total_tokens}"
        )

        return ChatResponse(
            id=req_id,
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat completion failed",
        )


@app.post("/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(rate_limit),
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""
    client, default_model = _get_openai_client()
    model = default_model or request.model

    async def generate() -> AsyncGenerator[str, None]:
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {delta.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def embed(
    request: EmbedRequest,
    api_key: str = Depends(rate_limit),
) -> EmbedResponse:
    """Generate embedding vector for input text."""
    start = time.perf_counter()

    if os.getenv("OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.embeddings.create(
            input=request.text,
            model=request.model,
        )
        embedding = response.data[0].embedding
    else:
        # Mock embedding for demo
        import math
        seed = hash(request.text)
        embedding = [math.sin(seed + i * 0.7) / 10 for i in range(1536)]

    latency_ms = (time.perf_counter() - start) * 1000
    return EmbedResponse(
        embedding=embedding,
        dimensions=len(embedding),
        model=request.model,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/v1/models")
async def list_models(api_key: str = Depends(authenticate)) -> dict:
    """List available models."""
    return {
        "models": [
            {"id": "gpt-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "provider": "openai"},
            {"id": "llama-3.1-70b-versatile", "provider": "groq"},
            {"id": "llama-3.1-8b-instant", "provider": "groq"},
        ]
    }


# ─────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ─────────────────────────────────────────────────────────
# STARTUP MESSAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        DAY 26 — AI API SERVICE                           ║")
    print("║                                                          ║")
    print("║  Test with API key: X-API-Key: dev-key-123              ║")
    print("║  Docs: http://localhost:8000/docs                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
