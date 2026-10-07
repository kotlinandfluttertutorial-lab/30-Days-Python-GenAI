# Day 26 — NotebookLM Notes: FastAPI AI Backend

## Why FastAPI for AI

1. **Async**: non-blocking LLM calls → handles thousands concurrent requests with one thread
2. **Pydantic**: same validation as LLM structured output; auto-validates request/response
3. **Auto-docs**: OpenAPI/Swagger at `/docs` — zero extra work
4. **Dependency injection**: auth, rate limiting, DB sessions composable and reusable
5. **Type hints**: IDE autocomplete, runtime errors caught at request boundary

## Core Pattern

```python
@app.post("/endpoint", response_model=ResponseModel)
async def endpoint(
    request: RequestModel,          # Validated automatically
    api_key: str = Depends(auth),   # Dependency injection
) -> ResponseModel:
    try:
        result = await do_async_work(request)
        return ResponseModel(**result)
    except HTTPException:
        raise  # Pass through HTTP errors
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(500, "Internal server error")  # Never expose raw errors
```

## Authentication

```python
# API key in header: X-API-Key: your-key
async def authenticate(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key not in valid_keys:
        raise HTTPException(401, "Invalid API key")
    return x_api_key
```

## Rate Limiting

```python
# Token bucket: N requests per 60 seconds per API key
# Track timestamps per key, reject if too many in window
```

## Streaming (SSE)

```python
# Server-Sent Events: stream tokens as "data: token\n\n"
return StreamingResponse(
    async_generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache"},
)
```

## Interview Facts

1. `@asynccontextmanager lifespan`: startup/shutdown code (load models, connect DBs)
2. `Depends()`: FastAPI's dependency injection — runs before endpoint
3. `response_model=`: validates return value, strips extra fields
4. `HTTPException(422)`: auto-raised by Pydantic validation failures
5. Never expose raw exception messages to clients (info leakage)
6. `X-Accel-Buffering: no` header: disables Nginx buffering for streaming

## Common Mistakes

- Using `def` instead of `async def` for endpoints with async operations
- Not handling `HTTPException` separately (would catch and re-wrap as 500)
- Returning raw Python dicts instead of Pydantic models (no validation)
- No global error handler → stack traces leak to clients
- Not adding `response_model` → no output validation
- Rate limiter state in-memory → resets on restart; use Redis for production
