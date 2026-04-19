from datetime import datetime, timezone
from typing import Any


HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
CONTRACT_VERSION = "monitoring-contract/v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def component(status: str, **fields: Any) -> dict[str, Any]:
    payload = {"status": status}
    payload.update({key: value for key, value in fields.items() if value is not None})
    return payload


def aggregate_status(checks: dict[str, dict[str, Any]]) -> str:
    statuses = [check.get("status", UNHEALTHY) for check in checks.values()]
    if any(status == UNHEALTHY for status in statuses):
        return UNHEALTHY
    if any(status == DEGRADED for status in statuses):
        return DEGRADED
    return HEALTHY


def health_status_code(status: str) -> int:
    return 503 if status == UNHEALTHY else 200


def build_contract_payload(
    *,
    service_id: str,
    service_name: str,
    service_type: str,
    environment: str,
    checks: dict[str, dict[str, Any]] | None = None,
    journey: dict[str, Any] | None = None,
    release: dict[str, Any] | None = None,
    operations: dict[str, Any] | None = None,
    summary: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    payload_status = status or aggregate_status(checks or {})
    payload: dict[str, Any] = {
        "version": CONTRACT_VERSION,
        "service": {
            "id": service_id,
            "name": service_name,
            "type": service_type,
            "environment": environment,
        },
        "status": payload_status,
        "timestamp": utc_timestamp(),
    }
    if summary:
        payload["summary"] = summary
    if checks is not None:
        payload["checks"] = checks
    if journey is not None:
        payload["journey"] = journey
    if release is not None:
        payload["release"] = release
    if operations is not None:
        payload["operations"] = operations
    return payload
