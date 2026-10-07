# Day 27 — Day Summary: Docker + Cloud

## What You Built
Production Dockerfile (non-root user, layer caching, healthcheck), Docker Compose with PostgreSQL + Redis + API, CI/CD pipeline sketch, cloud deployment options.

## Key Takeaways
1. Layer caching: copy requirements.txt before app code — rebuild only when deps change
2. Non-root user: mandatory security practice in production containers
3. `depends_on` with `service_healthy`: ensures DB/Redis are ready before API starts
4. Environment variables: .env locally, secret manager in production
5. docker-compose: local multi-service setup mirrors production architecture

## Phase 9 Complete (Days 26-27)
FastAPI AI Backend + Docker + Cloud. Production deployment pattern established.

## Tomorrow: Day 28 — LLMOps + Security + Evaluation
Structured logging, metrics, tracing, prompt injection defense, PII handling, guardrails.
