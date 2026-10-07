# Day 27 — Concepts
## Docker + Cloud

---

## 1. Dockerfile for AI Applications

```dockerfile
# ── Production Dockerfile for AI FastAPI app ──────────────────
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (security)
RUN useradd --create-home --shell /bin/bash appuser

# Working directory
WORKDIR /app

# Install Python dependencies FIRST (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example .env

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

## 2. Docker Compose for Multi-Service AI App

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ── AI API Backend ─────────────────────────────────────
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://ai_user:ai_pass@postgres:5432/ai_db
      - REDIS_URL=redis://redis:6379
      - CHROMADB_URL=http://chroma:8001
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - ai_network

  # ── PostgreSQL Database ────────────────────────────────
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ai_user
      POSTGRES_PASSWORD: ai_pass
      POSTGRES_DB: ai_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_user -d ai_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ai_network

  # ── Redis Cache ────────────────────────────────────────
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ai_network

  # ── ChromaDB Vector Database ───────────────────────────
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    networks:
      - ai_network

networks:
  ai_network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  chroma_data:
```

---

## 3. Environment Variables and Secrets

```bash
# .env.example (commit this)
OPENAI_API_KEY=your-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO

# .env (never commit — add to .gitignore)
OPENAI_API_KEY=sk-proj-actual-key
DATABASE_URL=postgresql://ai_user:secret_pass@localhost:5432/ai_db

# In production: use secret management
# AWS: AWS Secrets Manager
# GCP: Google Secret Manager
# Azure: Azure Key Vault
# Kubernetes: Secrets mounted as files
```

```python
# In Python: validate all secrets at startup
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DATABASE_URL", "REDIS_URL"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")
```

---

## 4. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy AI API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to cloud
        run: |
          # SSH to server and pull latest image
          # Or use cloud-specific deployment
```

---

## 5. Cloud Deployment Options

```
OPTION 1: VPS (Hetzner, DigitalOcean, Linode)
  Cost: $6-20/month
  Setup: Docker + Nginx + Let's Encrypt SSL
  Best for: small production, full control

OPTION 2: Container services
  AWS ECS / GCP Cloud Run / Azure Container Apps
  Cost: pay-per-request
  Setup: push Docker image → configure → deploy
  Best for: auto-scaling, no server management

OPTION 3: Kubernetes (EKS, GKE, AKS)
  Cost: higher
  Best for: large scale, enterprise
  Overkill for: most AI apps starting out

OPTION 4: Railway / Render / Fly.io
  Cost: free tier available
  Setup: connect GitHub → auto-deploy
  Best for: fast prototypes, small production

FOR AI APPS SPECIFICALLY:
  Modal: serverless GPU functions → model inference
  Replicate: model hosting API
  Hugging Face Spaces: demo deployment
```

---

## 6. Production Nginx Configuration

```nginx
# /etc/nginx/sites-available/ai-api
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # Proxy to FastAPI
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;

        # Request size limit
        client_max_body_size 10M;
    }
}
```
