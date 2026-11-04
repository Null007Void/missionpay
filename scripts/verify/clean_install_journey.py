"""Drive the full MissionPay journey against a running server over real HTTP.

This is the clean-install acceptance probe. It uses only the standard library, so it
can be run from any directory against an installed wheel with no access to the
development checkout. It counts assertions and exits non-zero on the first failure.

Usage:
    python clean_install_journey.py --base-url http://127.0.0.1:9877
    python clean_install_journey.py --base-url http://127.0.0.1:9877 --expect genuine

``--expect mock`` asserts that provider operations are refused. ``--expect genuine``
asserts the sandbox surface is present. It never performs a payout itself.
"""

from __future__ import annotations

import argparse
import re
import sys
from http.cookiejar import CookieJar
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    """Leave 3xx responses visible so redirects can be asserted, not followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None

VOLUNTEER = ("volunteer@missionpay.local", "volunteer-demo-passphrase")
REVIEWER = ("reviewer@missionpay.local", "reviewer-demo-passphrase")

CSRF = re.compile(r'name="csrf_token" value="([^"]+)"')
CLAIM = "/claims/river-safety-001"

assertions = 0
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    global assertions
    assertions += 1
    if not condition:
        failures.append(message)


class Browser:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()), _NoRedirect())

    def _request(self, method: str, path: str, data: dict | None = None):
        body = None
        headers = {}
        if data is not None:
            body = urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=20) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def get(self, path: str):
        return self._request("GET", path)

    def post(self, path: str, data: dict):
        return self._request("POST", path, data)

    def csrf_at(self, path: str = "/app") -> str:
        _status, body = self.get(path)
        match = CSRF.search(body)
        if not match:
            raise SystemExit(f"no CSRF token on {path}")
        return match.group(1)

    def review_identity(self, path: str) -> dict[str, str]:
        """The hidden identity a reviewer's browser would submit with approval."""
        _status, body = self.get(path)
        fields: dict[str, str] = {}
        for name in ("interpretation_id", "interpretation_output_sha256"):
            match = re.search(rf'name="{name}"\s+value="([^"]+)"', body, re.S)
            if match:
                fields[name] = match.group(1)
        return fields

    def login(self, credentials) -> None:
        email, password = credentials
        token = self.csrf_at("/login")
        status, _body = self.post(
            "/login", {"email": email, "password": password, "csrf_token": token}
        )
        check(status == 200 or status == 303, f"login for {email} returned {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expect", choices=["mock", "genuine"], default="mock")
    args = parser.parse_args()

    gen = args.expect == "genuine"

    # ---- volunteer intake ------------------------------------------------
    volunteer = Browser(args.base_url)
    status, body = volunteer.get("/login")
    check(status == 200, f"/login returned {status}")
    check("MISSIONPAY" in body, "/login does not identify the application")
    check(
        ("MODE: MOCK" in body) if not gen else ("MODE: PAYPAL SANDBOX" in body),
        "login page does not state the provider mode",
    )

    volunteer.login(VOLUNTEER)
    status, body = volunteer.get("/app")
    check(status == 200, f"/app returned {status}")
    check("Volunteer workspace" in body, "volunteer workspace missing")

    status, body = volunteer.post("/claims/demo", {"csrf_token": volunteer.csrf_at("/app")})
    check(status == 303, f"demo submission returned {status} (expected 303)")
    check(CLAIM in body or status == 303, "demo submission did not redirect to the claim")

    status, detail = volunteer.get(CLAIM)
    check(status == 200, f"claim detail returned {status}")
    check("Expense evidence" in detail, "evidence panel missing")
    check("$78.99" in detail, "amount not shown")
    check(
        "NOT RUN — THE AI STAGE IS INCOMPLETE" in detail,
        "a fresh claim does not declare the AI stage incomplete",
    )

    # The AI stage is explicit and persisted; refreshing must not produce inference.
    for _ in range(2):
        volunteer.get(CLAIM)
    status, _body = volunteer.post(f"{CLAIM}/interpret", {"csrf_token": volunteer.csrf_at("/app")})
    check(status == 303, f"interpretation action returned {status} (expected 303)")

    status, detail = volunteer.get(CLAIM)
    check(status == 200, f"claim detail returned {status} after interpretation")
    check("Observed fact" in detail, "observed facts not rendered")
    check("Inference" in detail, "inferences not rendered")
    check("Uncertainty" in detail, "uncertainty not rendered")
    check("Citation" in detail, "citations not rendered")
    check("interpretation interp-" in detail, "interpretation identity not shown")
    check("Garden Fund" in detail, "Garden Fund decision missing")
    check("Refused" in detail, "the restricted fund did not visibly refuse")
    check("River Cleanup Fund" in detail, "River Cleanup Fund decision missing")
    check("Eligible" in detail, "the eligible fund was not shown as eligible")
    check("No payment intent" in detail, "pre-authorization state is wrong")
    check(
        ("SIMULATED INTERPRETATION FIXTURE" in detail) if not gen else ("LIVE LOCAL MODEL" in detail),
        "interpretation provenance is not labelled",
    )
    check(
        ("SIMULATED — NO PAYPAL PAYMENT EXECUTED" in detail)
        if not gen
        else ("SANDBOX FUNDS, NOT REAL MONEY" in detail),
        "payment mode banner is missing or wrong",
    )

    # A volunteer must not be able to execute.
    status, body = volunteer.post(f"{CLAIM}/pay", {"csrf_token": volunteer.csrf_at("/app")})
    check(status == 403, f"volunteer payment attempt returned {status} (expected 403)")

    # A duplicate submission must be refused, not silently accepted.
    status, body = volunteer.post("/claims/demo", {"csrf_token": volunteer.csrf_at("/app")})
    check(status == 409, f"duplicate submission returned {status} (expected 409)")

    # ---- reviewer authority ---------------------------------------------
    reviewer = Browser(args.base_url)
    reviewer.login(REVIEWER)
    status, queue = reviewer.get("/review")
    check(status == 200, f"/review returned {status}")
    check("river-safety-001" in queue, "claim missing from the review queue")

    status, body = reviewer.post(
        f"{CLAIM}/approve",
        {"csrf_token": reviewer.csrf_at("/app"), "revision": "1", "amount_cents": "1"},
    )
    check(status == 409, f"tampered amount returned {status} (expected 409)")
    check("FROZEN_AMOUNT_MISMATCH" in body, "tampered amount was not named")

    status, body = reviewer.post(
        f"{CLAIM}/approve",
        {"csrf_token": reviewer.csrf_at("/app"), "revision": "99", "amount_cents": "7899"},
    )
    check(status == 409, f"stale revision returned {status} (expected 409)")
    check("FROZEN_REVISION_MISMATCH" in body, "stale revision was not named")

    status, body = reviewer.post(
        f"{CLAIM}/approve",
        {"csrf_token": "forged-token", "revision": "1", "amount_cents": "7899"},
    )
    check(status == 403, f"forged CSRF returned {status} (expected 403)")

    # Approval must name the interpretation the reviewer saw.
    status, body = reviewer.post(
        f"{CLAIM}/approve",
        {"csrf_token": reviewer.csrf_at("/app"), "revision": "1", "amount_cents": "7899"},
    )
    check(status == 409, f"approval without a reviewed identity returned {status} (expected 409)")
    check(
        "INTERPRETATION_NOT_FOUND" in body or "INTERPRETATION_OUTPUT_MISMATCH" in body,
        "unnamed approval was not refused by name",
    )

    identity = reviewer.review_identity(CLAIM)
    check(bool(identity.get("interpretation_id")), "approval form carries no interpretation id")
    check(
        len(identity.get("interpretation_output_sha256", "")) == 64,
        "approval form carries no output digest",
    )

    status, _body = reviewer.post(
        f"{CLAIM}/approve",
        {
            "csrf_token": reviewer.csrf_at("/app"),
            "revision": "1",
            "amount_cents": "7899",
            **identity,
        },
    )
    check(status == 303, f"authorization returned {status} (expected 303)")

    status, approved = reviewer.get(CLAIM)
    check(status == 200, f"authorized claim returned {status}")
    check("Human authorization" in approved, "authorization panel missing")
    check("Exact authority frozen" in approved, "authority was not frozen")
    check("Not attempted" in approved, "pre-dispatch state is wrong")
    check("missionpay:" in approved, "operation key not visible")

    # ---- payment boundary ----------------------------------------------
    if gen:
        check("Execute one sandbox payout" in approved, "execution control missing in sandbox mode")
    else:
        status, body = reviewer.post(f"{CLAIM}/pay", {"csrf_token": reviewer.csrf_at("/app")})
        check(status == 409, f"mock execution returned {status} (expected 409)")
        check(
            "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS" in body,
            "mock mode did not refuse provider execution by name",
        )

    status, body = reviewer.post(f"{CLAIM}/reconcile", {"csrf_token": reviewer.csrf_at("/app")})
    check(status == 409, f"mock reconciliation returned {status} (expected 409)")

    # Re-authorizing the same revision must not create a second operation.
    status, body = reviewer.post(
        f"{CLAIM}/approve",
        {"csrf_token": reviewer.csrf_at("/app"), "revision": "1", "amount_cents": "7899"},
    )
    check(status == 409, f"repeated approval returned {status} (expected 409)")
    check("DUPLICATE_REIMBURSEMENT_BLOCKED" in body, "duplicate approval was not named")

    # ---- report ---------------------------------------------------------
    print(f"assertions={assertions} failures={len(failures)}")
    for failure in failures:
        print(f"  FAIL: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
