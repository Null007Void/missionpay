"""Authenticated MissionPay roles using Argon2 and server-side sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    VOLUNTEER = "volunteer"
    REVIEWER = "reviewer"


class AuthenticationFailed(ValueError):
    pass


class AuthorizationDenied(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AuthenticatedSession:
    token: str
    csrf_token: str
    user_id: str
    role: Role
    expires_at: datetime
