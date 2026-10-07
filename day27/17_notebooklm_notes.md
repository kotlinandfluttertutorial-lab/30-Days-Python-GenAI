# Day 27 — NotebookLM Notes: Docker + Cloud

## Docker Commands

```bash
docker build -t ai-api:latest .          # Build image
docker run -p 8000:8000 ai-api:latest    # Run container
docker ps                                 # List running
docker logs <container_id>               # View logs
docker exec -it <id> bash                # Shell into container
docker-compose up --build                # Start all services
docker-compose down -v                   # Stop + remove volumes
docker system prune -f                   # Clean unused images/containers
```

## Dockerfile Best Practices

1. **Non-root user**: `useradd`, then `USER appuser` — never run as root
2. **Layer caching**: copy requirements.txt BEFORE app code
3. **slim base**: `python:3.11-slim` not `python:3.11` (smaller image)
4. **no-cache-dir**: `pip install --no-cache-dir` reduces image size
5. **HEALTHCHECK**: Docker knows if container is unhealthy, can restart
6. **Multi-stage builds**: build in one stage, copy only artifacts to final (not needed for Python)

## Docker Compose Service Dependencies

```yaml
depends_on:
  postgres:
    condition: service_healthy   # Wait for healthcheck to pass
  redis:
    condition: service_healthy
```

## Environment Variables Pattern

```bash
# Development: .env file
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...

# Docker Compose: env_file + environment override
env_file: [.env]
environment:
  - DATABASE_URL=postgresql://ai_user:ai_pass@postgres:5432/db

# Production: cloud secret manager (never .env in production)
```

## Interview Facts

1. `docker build` creates an immutable image; `docker run` creates a container from it
2. Layers: each RUN/COPY/ADD creates a layer. Order matters for caching.
3. `depends_on` with `condition: service_healthy` waits for healthcheck (not just startup)
4. Network: services in the same compose file can communicate by service name
5. Volume mounts: persist data outside container lifecycle
6. `restart: unless-stopped`: restart on crash, but not on manual `docker stop`

## Common Mistakes

- Copying entire project before installing deps (breaks layer caching)
- Running as root user (security vulnerability)
- Secrets in Dockerfile or committed .env (use env_file + secret manager)
- `depends_on` without `condition: service_healthy` (DB not ready when app starts)
- `docker run` without naming the container (hard to reference later)
- Not adding healthcheck → Docker can't detect unhealthy container
