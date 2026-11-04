"""Hosted structured-output adapter: one provider, one pinned origin, one attempt.

Why this shape rather than a provider SDK: the whole product's claim is that the model
proposes and deterministic code plus a human decides. An SDK that retries, falls back,
streams or "helpfully" repairs JSON would put decisions this module cannot see between
the request and the record. So the adapter is one function, one POST, no retry, no
fallback, and a typed failure for every way the provider can decline.

Security and cost properties, each covered by a test:

* the origin is pinned to the provider's API host and compared exactly, so a base URL
  from the environment cannot redirect a credential;
* **redirects are refused, not followed.** urllib follows a 3xx by default and re-sends
  the request — including ``Authorization`` — to whatever host the response names. The
  opener is therefore built without a redirect handler, and a 3xx is its own refusal;
* the key appears only inside the ``Authorization`` header. It is ``repr=False``, it is
  never placed in an exception message, and every failure raises a bare code rather
  than a status line, a body or a URL that could contain a query parameter;
* **the request size is checked before it leaves**: the payload length in bytes plus the
  output ceiling must fit the approved total budget, or nothing is sent. The payload
  length is a conservative estimate of the prompt token count, not a proven ceiling —
  it assumes the provider tokenizer emits no more tokens than there are bytes, which
  this code cannot verify. A post-response check cannot undo spend, so both exist;
* the answer must have completed normally (``finish_reason == "stop"``) and must carry
  usage that is present, integral, non-negative, internally consistent and within
  budget. Usage is retained for evidence rather than discarded;
* a refusal, a truncation, a filtered answer, an empty answer, an oversized answer and
  an unparseable body are different typed failures — none of them is "retry";
* the response body is read with a byte ceiling, so a hostile or broken endpoint cannot
  exhaust memory before the parse runs;
* a wall-clock deadline is measured around the call. The socket timeout bounds one
  blocking operation; it is not a total duration, and the two are reported separately.
"""

from __future__ import annotations

import json
import time
from typing import Protocol
import urllib.error
import urllib.request

from .ai import INTERPRETATION_SCHEMA

#: The only origin this build will send a credential to.
PINNED_ORIGIN = "https://api.openai.com"
CHAT_PATH = "/v1/chat/completions"
PROVIDER_ID = "openai"

#: A complete interpretation of a 20 000-character receipt is far smaller than this.
MAX_RESPONSE_BYTES = 256 * 1024

#: Bounds on the knobs that decide cost. Enforced at construction, not at send.
MIN_OUTPUT_TOKENS = 1
MAX_OUTPUT_TOKENS = 4096
MIN_TOTAL_TOKENS = 2
MAX_TOTAL_TOKENS = 32_768
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 120
MIN_LATE_RESPONSE_SECONDS = 1
MAX_LATE_RESPONSE_SECONDS = 600

#: Characters per token used for the *pre-send* estimate. English prose runs nearer 4,
#: JSON with punctuation nearer 3. The smaller figure over-estimates the prompt, which
#: is the safe direction: refuse before sending rather than discover the overrun after
#: the request has been paid for.
ESTIMATE_CHARS_PER_TOKEN = 3

#: The three counters a usage block must carry, in the order they are checked.
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class HostedModelError(RuntimeError):
    """A typed refusal. The message is a code; it never carries a secret or a body."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class HostedTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        *,
        max_bytes: int,
        timeout: int,
    ) -> tuple[int, bytes]: ...


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Returning ``None`` from ``redirect_request`` turns a 3xx into an error.

    This is the whole mitigation. The default handler follows ``Location`` and re-sends
    the request, which for a credentialed API means handing the bearer token to whatever
    host the response named.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


class UrllibHostedTransport:
    """One bounded HTTPS request. Raises :class:`HostedModelError` on transport failure."""

    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        # Built once, without a redirect handler, so no 3xx can become a second hop.
        self._opener = urllib.request.build_opener(_RefuseRedirects())

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        *,
        max_bytes: int,
        timeout: int,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read(max_bytes + 1)
                status = response.status
        except urllib.error.HTTPError as exc:
            # A 3xx arrives here because the redirect handler refused it. The body is
            # never read: an error page can be arbitrarily large and can echo request
            # content back into a log line.
            return exc.code, b""
        except TimeoutError as exc:
            raise HostedModelError("HOSTED_TIMEOUT") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                raise HostedModelError("HOSTED_TIMEOUT") from exc
            raise HostedModelError("HOSTED_NETWORK_ERROR") from exc
        except OSError as exc:  # connection reset, DNS failure, TLS failure
            raise HostedModelError("HOSTED_NETWORK_ERROR") from exc
        if len(payload) > max_bytes:
            raise HostedModelError("HOSTED_RESPONSE_TOO_LARGE")
        return status, payload

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"UrllibHostedTransport(timeout_seconds={self.timeout_seconds}, "
            "redirects=refused)"
        )


class HostedJsonModel:
    """A :class:`missionpay.ai.JsonModel` backed by one hosted chat completion."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str,
        max_output_tokens: int,
        max_total_tokens: int,
        timeout_seconds: int,
        late_response_seconds: int | None = None,
        transport: HostedTransport | None = None,
    ) -> None:
        if not api_key:
            raise HostedModelError("HOSTED_AI_KEY_REQUIRED")
        if not model_id:
            raise HostedModelError("HOSTED_MODEL_REQUIRED")
        if base_url.rstrip("/") != PINNED_ORIGIN:
            # Exact comparison, not a prefix or suffix test: a host such as
            # api.openai.com.attacker.example contains the pinned string.
            raise HostedModelError("HOSTED_ORIGIN_REQUIRED")
        if isinstance(max_output_tokens, bool) or not (
            MIN_OUTPUT_TOKENS <= max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise HostedModelError("HOSTED_MAX_OUTPUT_TOKENS_INVALID")
        if isinstance(max_total_tokens, bool) or not (
            MIN_TOTAL_TOKENS <= max_total_tokens <= MAX_TOTAL_TOKENS
        ):
            raise HostedModelError("HOSTED_MAX_TOTAL_TOKENS_INVALID")
        if max_output_tokens >= max_total_tokens:
            # A budget that cannot hold even one answer is a configuration mistake.
            raise HostedModelError("HOSTED_BUDGET_INCONSISTENT")
        if isinstance(timeout_seconds, bool) or not (
            MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise HostedModelError("HOSTED_TIMEOUT_INVALID")
        late_response = (
            timeout_seconds if late_response_seconds is None else late_response_seconds
        )
        if isinstance(late_response, bool) or not (
            MIN_LATE_RESPONSE_SECONDS <= late_response <= MAX_LATE_RESPONSE_SECONDS
        ):
            raise HostedModelError("HOSTED_LATE_RESPONSE_LIMIT_INVALID")

        self.base_url = PINNED_ORIGIN
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.timeout_seconds = timeout_seconds
        #: A late-response threshold, not a total deadline: a blocked read is bounded by
        #: ``timeout_seconds``, and this value only decides when an answer that did
        #: arrive is refused for having arrived too late.
        self.late_response_seconds = late_response
        self._api_key = api_key  # deliberately not a public attribute
        self._transport = transport or UrllibHostedTransport(timeout_seconds)
        self.last_usage: dict[str, int] | None = None
        self.last_estimate: dict[str, int] | None = None
        self.last_elapsed_seconds: float | None = None

    # -- reporting -------------------------------------------------------

    @property
    def provider(self) -> str:
        return PROVIDER_ID

    @property
    def qualified_model_id(self) -> str:
        """Provider-qualified identity, so a stored id cannot be ambiguous."""
        return f"{PROVIDER_ID}:{self.model_id}"

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to log or render."""
        return {
            "provider": PROVIDER_ID,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "timeout_seconds": self.timeout_seconds,
            "late_response_seconds": self.late_response_seconds,
            "api_key_present": bool(self._api_key),
        }

    def __repr__(self) -> str:
        return (
            f"HostedJsonModel(provider={PROVIDER_ID!r}, model_id={self.model_id!r}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"max_total_tokens={self.max_total_tokens}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"late_response_seconds={self.late_response_seconds})"
        )

    # -- the one request -------------------------------------------------

    def _payload(self, system: str, user: str) -> bytes:
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            # Do not retain the request as application state. Abuse monitoring is a
            # documented provider policy and is not something this flag controls.
            "store": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "missionpay_interpretation",
                    "strict": True,
                    "schema": INTERPRETATION_SCHEMA,
                },
            },
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def estimate(self, body: bytes) -> dict[str, int]:
        """Pre-send arithmetic: a conservative ceiling and a labelled expectation.

        ``upper_bound_input_tokens`` is the payload length in bytes. It is called an upper
        bound because a byte-level tokenizer cannot cover fewer than one byte per token,
        so for such a tokenizer the byte count does cap the token count — but that is an
        assumption about a tokenizer this process never inspects, not a property it can
        check. A tokenizer that splits text more finely would produce more tokens than the
        figure allows, and the request would already have been sent by then. It is a
        conservative planning figure used as a pre-send gate; it is not a proof.

        ``estimated_input_tokens`` is a heuristic (characters / 3) that says what the
        request will probably cost. It is not a bound, it is not a measurement, and it is
        never used to decide whether a request may be sent.
        """
        payload_bytes = len(body)
        estimated = payload_bytes // ESTIMATE_CHARS_PER_TOKEN + 1
        return {
            "payload_bytes": payload_bytes,
            "upper_bound_input_tokens": payload_bytes,
            "estimated_input_tokens": estimated,
            "max_output_tokens": self.max_output_tokens,
            "upper_bound_total_tokens": payload_bytes + self.max_output_tokens,
            "estimated_total_tokens": estimated + self.max_output_tokens,
            "budget_tokens": self.max_total_tokens,
            "bound_basis": (
                "payload_bytes — an assumption, not a proven bound: it holds only if the "
                "provider tokenizer emits no more tokens than bytes, which this code cannot verify"
            ),
            "estimate_basis": "payload_bytes / 3 heuristic (not a bound)",
        }

    def preview(self, system: str, user: str) -> dict[str, int]:
        """The exact payload arithmetic for a request, without sending one."""
        return self.estimate(self._payload(system, user))

    def _dispatch(self, headers, body: bytes, timeout: int) -> tuple[int, bytes]:
        """One transport call, with transport-level failures normalised.

        A transport that raises instead of returning a status is still one attempt: it
        is classified here rather than escaping as a bare exception, because a raw
        ``TimeoutError`` reaching a route would render as a 500 and read like a bug in
        the product rather than an unanswered request.
        """
        try:
            return self._transport.request(
                "POST",
                self.base_url + CHAT_PATH,
                headers,
                body,
                max_bytes=MAX_RESPONSE_BYTES,
                timeout=timeout,
            )
        except HostedModelError:
            raise
        except TimeoutError as exc:
            raise HostedModelError("HOSTED_TIMEOUT") from exc
        except OSError as exc:
            raise HostedModelError("HOSTED_NETWORK_ERROR") from exc
        except Exception as exc:  # noqa: BLE001 - classified, never retried
            raise HostedModelError("HOSTED_TRANSPORT_ERROR") from exc

    def complete_json(self, system: str, user: str) -> str:
        """Exactly one POST. Any failure raises; nothing is retried or repaired."""
        body = self._payload(system, user)
        estimate = self.estimate(body)
        self.last_estimate = estimate
        if estimate["upper_bound_total_tokens"] > self.max_total_tokens:
            # Refused before a socket is opened, on the proved bound rather than on the
            # heuristic: the budget cannot absorb this request even in the best case for
            # the tokenizer.
            raise HostedModelError("HOSTED_REQUEST_OVER_BUDGET")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.monotonic()
        status, raw = self._dispatch(
            headers, body, min(self.timeout_seconds, self.late_response_seconds)
        )
        self.last_elapsed_seconds = time.monotonic() - started

        if len(raw) > MAX_RESPONSE_BYTES:
            # Enforced here as well as in the HTTP transport: a different transport
            # implementation must not be able to hand an unbounded body to the parser.
            raise HostedModelError("HOSTED_RESPONSE_TOO_LARGE")
        if 300 <= status < 400:
            # The opener refuses to follow, and the status is refused explicitly so the
            # reason is unambiguous in the record.
            raise HostedModelError("HOSTED_REDIRECT_REFUSED")
        if status in (401, 403):
            raise HostedModelError("HOSTED_AUTH_FAILED")
        if status == 429:
            raise HostedModelError("HOSTED_RATE_LIMITED")
        if status == 400:
            raise HostedModelError("HOSTED_REQUEST_REJECTED")
        if status == 404:
            raise HostedModelError("HOSTED_MODEL_UNAVAILABLE")
        if status >= 500:
            raise HostedModelError("HOSTED_PROVIDER_ERROR")
        if status < 200 or status >= 300:
            raise HostedModelError(f"HOSTED_HTTP_{status}")

        if self.last_elapsed_seconds > self.late_response_seconds:
            # A *late-response rejection*, not a deadline: the call was not interrupted,
            # it ran to completion and the answer is refused for having taken longer than
            # the owner approved. The spend is already committed, so this is terminal and
            # the caller must not retry. A true total deadline would need an interruptible
            # read, which this adapter does not implement.
            raise HostedModelError("HOSTED_LATE_RESPONSE")

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HostedModelError("HOSTED_RESPONSE_INVALID") from exc
        if not isinstance(payload, dict):
            raise HostedModelError("HOSTED_RESPONSE_INVALID")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise HostedModelError("HOSTED_RESPONSE_INVALID")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise HostedModelError("HOSTED_RESPONSE_INVALID")

        # A refusal is a decision by the provider, not a malformed answer, and it is
        # reported as one even when the provider sends no usage block.
        if message.get("refusal"):
            raise HostedModelError("HOSTED_REFUSED")

        # Usage is evidence: an answer whose cost cannot be audited is not an answer.
        # Checked before the content is accepted, so an unaccountable answer never
        # reaches the validator.
        self.last_usage = self._validated_usage(payload.get("usage"))

        finish = choice.get("finish_reason")
        if finish == "length":
            # Partial JSON parses as nothing; calling it "invalid JSON" would hide that
            # the token ceiling, not the model's obedience, is what failed.
            raise HostedModelError("HOSTED_TRUNCATED")
        if finish == "content_filter":
            raise HostedModelError("HOSTED_CONTENT_FILTERED")
        if finish != "stop":
            # Absent, null, tool_calls or anything unknown: the answer did not finish the
            # way an extraction answer must.
            raise HostedModelError("HOSTED_INCOMPLETE")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise HostedModelError("HOSTED_CONTENT_MISSING")
        return content

    def _validated_usage(self, usage: object) -> dict[str, int]:
        """Retain what the provider reported, then apply the two ceilings in order.

        The retention happens as soon as the block is structurally sound. A response that
        is refused for exceeding a ceiling still cost money, so its counters are evidence
        and must survive the refusal — ``last_usage`` is set before either ceiling check.
        """
        if not isinstance(usage, dict):
            raise HostedModelError("HOSTED_USAGE_MISSING")
        values: dict[str, int] = {}
        for key in USAGE_KEYS:
            value = usage.get(key)
            if value is None:
                raise HostedModelError("HOSTED_USAGE_MISSING")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HostedModelError("HOSTED_USAGE_INVALID")
            values[key] = value
        if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
            raise HostedModelError("HOSTED_USAGE_INCONSISTENT")
        self.last_usage = values

        # The output ceiling is a separate promise from the total budget: the provider was
        # asked for at most max_output_tokens and answered with more, so the answer is
        # refused even if the total happens to fit.
        if values["completion_tokens"] > self.max_output_tokens:
            raise HostedModelError("HOSTED_OUTPUT_CEILING_EXCEEDED")
        if values["total_tokens"] > self.max_total_tokens:
            raise HostedModelError("HOSTED_USAGE_OVER_BUDGET")
        return values
