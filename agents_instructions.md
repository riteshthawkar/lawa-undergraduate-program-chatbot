# Agents Instructions

Use this file as the implementation contract for every service repo that is monitored by the platform status dashboard.

The goal is simple:
- every service should expose health in the same way
- every release should be traceable
- fatal user-path failures must be monitorable
- monitoring traffic must not pollute product analytics or persisted data

Do not put `.env` or secret-management details in this file. This file only covers code, endpoints, contracts, deployment metadata, and CI/CD behavior.

## 1. Health Contract

Every service must expose these endpoints where applicable:

### Required
- `GET /health`
  - Fast liveness + high-level readiness.
  - Must return machine-readable JSON.
  - Must include a top-level status field.
- `GET /health/detailed`
  - Full dependency breakdown.
  - Must include per-component status and error context.

### Required for LLM-backed services
- `GET /health/generation`
  - Must execute a safe synthetic generation path.
  - Must validate the actual model/provider path used in production.
  - Must fail if generation, retrieval, reranking, or model selection is broken.

### Status semantics
- `200` means healthy or degraded-but-serving, depending on body contract.
- `503` means the service is not ready for real traffic or a critical path is broken.
- Do not return `200` for a health probe if the primary user path is fatally broken.

### JSON shape
Use a stable shape similar to:

```json
{
  "status": "healthy",
  "service": "lawa-rag",
  "timestamp": "2026-04-07T10:00:00Z",
  "checks": {
    "database": { "status": "healthy" },
    "vector_store": { "status": "healthy" },
    "openai": { "status": "healthy", "model": "gpt-5.4-mini" },
    "generation": { "status": "healthy" }
  }
}
```

Required status values:
- `healthy`
- `degraded`
- `unhealthy`

## 2. Monitoring Probe Behavior

All services must support synthetic monitoring traffic without side effects.

### Probe header
Honor this header:
- `x-health-probe: true`

### Probe rules
When `x-health-probe: true` is present:
- do not persist chat history
- do not persist analytics events
- do not trigger notifications
- do not create user-visible artifacts
- return explicit failures instead of soft user-facing fallback text

For example:
- normal user request may return a friendly fallback message
- health probe request must return a failing status code if generation failed

## 3. LLM/Model Rules

### Required
- all models must be env-driven or config-driven
- never hard-code deprecated or provider-specific model aliases in request paths
- `/health/generation` must validate the actual configured model path
- health checks must detect:
  - missing API key
  - invalid model
  - deprecated model
  - provider rejection
  - retrieval/generation pipeline failure

### For RAG services
Health must cover:
- embedding model availability
- vector DB connectivity
- retrieval path readiness
- reranker readiness if used
- final generation readiness

## 4. Deployment Metadata Contract

Every repo should make deployed versions observable and loggable.

### Required metadata per deployment
- service identifier
- environment
- version
- commit SHA
- deployed by
- deployed at
- notes / release summary

### Required behavior
- deployment scripts or CI must record this metadata
- post-deploy smoke checks should use the live service endpoints
- deployment info should be easy to feed into the dashboard registry

### Strong recommendation
Expose a lightweight version endpoint or include this in `/health/detailed`:

```json
{
  "release": {
    "version": "v1.8.3",
    "commit_sha": "7a48fb5",
    "deployed_at": "2026-04-07T10:00:00Z"
  }
}
```

## 5. Failure Semantics

The dashboard should be able to distinguish between:
- service reachable but degraded
- service up but user path broken
- service fully unavailable
- service in planned maintenance

### Rules
- fatal generation failure on a probe must not look healthy
- dependency outage must be surfaced in `/health/detailed`
- if the app catches exceptions for UX, probe mode must still surface them as failures
- do not rely only on basic liveness text like `"working"`

## 6. Logging and Debuggability

Every service should emit structured logs.

### Required
- timestamp
- level
- request identifier
- endpoint / route
- dependency failure reason
- model/provider error details where safe

### Strong recommendation
- include release version / commit SHA in logs
- include probe mode marker when `x-health-probe: true`

## 7. Ownership and Operations Metadata

Each repo should make it easy to identify:
- service owner
- runbook URL
- dashboard service ID
- primary repository URL
- public base URL

This can live in:
- a small metadata file
- code-level constants
- or documented release metadata

The important part is consistency, not storage location.

## 8. CI/CD Contract

Every service repo should have CI that runs on pull requests.

### Required CI checks
- install dependencies
- lint / typecheck
- tests
- build
- health-contract validation for modified health endpoints

### Deployment flow
Use this release pattern:
1. merge to `main`
2. build from a tagged release or explicit commit SHA
3. take a pre-deploy snapshot or backup
4. deploy
5. run smoke checks
6. record deployment metadata
7. rollback automatically or explicitly if smoke checks fail

## 9. Smoke Tests After Deploy

Every deploy should validate:
- `GET /health`
- `GET /health/detailed`
- `GET /health/generation` for LLM services
- one user-path probe using `x-health-probe: true`

For chatbot services, smoke tests must verify:
- valid response object exists
- failure phrases are absent
- actual generation path works

## 10. Agent Studio / Multi-Service Systems

For multi-backend systems:
- widget/backend health must check the real backend port and URL
- async health handlers must not call sync DB APIs directly
- optional metrics libraries must not hard-fail the whole health endpoint
- direct-IP health checks must be supported where required for monitors

## 11. What To Avoid

Do not:
- hard-code model names in request handlers
- return `200` for synthetic probes when critical generation failed
- let probe traffic create DB rows or analytics noise
- depend only on `/health` text responses
- hide dependency failures behind generic success JSON
- make health depend on optional libraries unless their failure is non-critical and clearly marked as degraded

## 12. Definition of Done for Each Repo

A repo is aligned only when all of these are true:
- `/health` exists and is machine-readable
- `/health/detailed` exists and checks real dependencies
- `/health/generation` exists for LLM-backed services
- probe mode is side-effect free
- fatal user-path failures are observable by monitoring
- model configuration is not hard-coded
- CI runs lint/tests/build
- deploys are versioned and traceable
- post-deploy smoke checks exist
- rollback path is documented

## 13. Dashboard Integration Checklist

When connecting a repo to the monitoring dashboard, ensure:
- service ID matches the dashboard service config
- health endpoints are stable
- generation probe endpoint is added where needed
- release/deployment metadata can be logged
- owner/runbook/repo/base URL metadata is available

If a service cannot support safe public probing, add a dedicated internal health path instead of weakening the contract.
