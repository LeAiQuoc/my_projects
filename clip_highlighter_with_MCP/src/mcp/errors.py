from __future__ import annotations

from typing import Any


def ok(data: dict[str, Any]) -> dict[str, Any]:
    """Build a successful MCP tool response envelope."""
    return {"success": True, "data": data, "error": None}


def fail(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a failed MCP tool response envelope."""
    return {
        "success": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }
