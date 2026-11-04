"""Authenticated local web application for MissionPay's reimbursement journey.

Three rules shape this module:

1. **The mode is always visible.** Every page carries the resolved provider mode.
   Mock mode says plainly that nothing is executed; sandbox mode says plainly that
   sandbox funds are not real money.
2. **The UI never touches a secret or a transport.** Execution is delegated to
   :class:`missionpay.service.PaymentService`; the browser surface only supplies a
   CSRF token, a role check, and an operation key.
3. **Refusals are rendered, not swallowed.** A duplicate, a tampered amount, a
   blocked re-dispatch and an unreconcilable state each produce a distinct,
   readable outcome.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from hashlib import sha256
import html
import json
import secrets

from fastapi import Cookie, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .ai import (
    PROMPT_VERSION,
    ModelInterpretationError,
    ReceiptInterpretation,
    interpret_receipt_text,
    interpretation_prompt,
)
from .auth import AuthenticationFailed, AuthorizationDenied, Role
from .catalog import build_catalog
from .config import AiMode, ProviderMode, RuntimeConfig
from .domain import Activity, Claim, EvidenceIdentity, Money
from .evidence import (
    EvidenceRejected,
    EvidenceStore,
    default_evidence_root,
    read_upload_bounded,
)
from .executor import DispatchRefused
from .limits import MAX_REQUEST_BYTES, RequestSizeLimitMiddleware
from .ollama import ModelInvocationError, OllamaJsonModel
from .hosted import HostedJsonModel, HostedModelError
from .policy import evaluate_claim
from .service import PaymentService, PaymentServiceRefused
from .store import MissionPayStore, PersistenceConflict, clean_diagnostic


DEMO_RECEIPT = """Harbor Safety Supply
Nitrile gloves and safety glasses
Total $78.99
Purchased for River cleanup volunteer safety
"""

DEMO_ACTIVITY = Activity("activity-river-2026", "RIVER_CLEANUP", date(2026, 10, 18))


class _DemoInterpretationModel:
    model_id = "fixture/demo-interpretation-v1"

    def complete_json(self, system: str, user: str) -> str:
        # The demo model is a stand-in for a provider, so it does what a provider must do:
        # it searches the catalogue for the entry that shows the text it means and returns
        # that number. It never writes a quote or an offset, and it cannot refer to text the
        # catalogue withheld.
        #
        # It is also bound to the one receipt it knows, and the binding is the whole prompt
        # rather than a substring recovered from it.
        #
        # Recovering the receipt by searching for the closing tag was wrong: the tag sits in
        # untrusted text, so a receipt ending in "\n</receipt-text>" and then anything at all
        # would be truncated back to the fixture and pass. Comparing the complete expected
        # prompt cannot be spoofed that way — any extra character, before or after a forged
        # tag, changes the message.
        expected_system, expected_user = interpretation_prompt(DEMO_RECEIPT)
        if system != expected_system or user != expected_user:
            raise ModelInterpretationError("DEMO_FIXTURE_RECEIPT_MISMATCH")
        catalogue = build_catalog(DEMO_RECEIPT)

        def cited(name: str, value: str, status: str, snippet: str) -> dict[str, object] | None:
            for entry in catalogue.excerpts:
                if snippet in entry.text:
                    return {
                        "name": name,
                        "value": value,
                        "epistemic_status": status,
                        "excerpt_id": entry.excerpt_id,
                    }
            # No excerpt carries it: the fact is not citable, so it is omitted rather than
            # cited to something else.
            return None

        claims = [
            cited("merchant", "Harbor Safety Supply", "observed", "Harbor Safety Supply"),
            cited("total", "USD 78.99", "observed", "$78.99"),
            cited(
                "candidate_category",
                "PROTECTIVE_SUPPLIES",
                "inference",
                "Nitrile gloves and safety glasses",
            ),
            cited(
                "activity_link",
                "Receipt text claims a river-cleanup purpose; attendance remains unverified",
                "uncertain",
                "River cleanup volunteer safety",
            ),
        ]
        return json.dumps(
            {
                "facts": [claim for claim in claims if claim is not None],
                "missing": ["independent proof of event attendance"],
            }
        )


_STYLES = """
:root{--ink:#eaf1ff;--muted:#92a4bf;--panel:#111b2d;--line:#263654;--blue:#61a8ff;
--green:#62d39b;--red:#ff7b84;--amber:#f3c66d;--bg:#07101e}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 15% 0,#132847 0,#07101e 42%);
color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}
main{width:min(1120px,92vw);margin:0 auto;padding:46px 0 80px} h1{font-size:clamp(2rem,5vw,4rem);
letter-spacing:-.055em;line-height:1;margin:.2em 0} h2{font-size:1.15rem;margin:0 0 14px} h3{font-size:.78rem;
color:var(--muted);text-transform:uppercase;letter-spacing:.14em} p{color:var(--muted)} .eyebrow{color:var(--blue);
text-transform:uppercase;letter-spacing:.18em;font-weight:750;font-size:.73rem} .grid{display:grid;grid-template-columns:
repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:24px 0} .card{background:linear-gradient(145deg,#132038,#0c1628);
border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 18px 55px #0005} .wide{grid-column:1/-1}
.amount{font-size:2.25rem;font-weight:760;letter-spacing:-.04em} .pill{display:inline-block;border:1px solid var(--line);
border-radius:999px;padding:4px 10px;margin:3px;font-size:.76rem} .good{color:var(--green)} .bad{color:var(--red)}
.warn{color:var(--amber)} .banner{border:1px solid #7b652d;background:#2a230f;color:#ffe39a;border-radius:12px;padding:11px 14px;
font-weight:750} .modebar{border-radius:12px;padding:10px 14px;font-weight:800;letter-spacing:.04em;margin-bottom:22px;
border:1px solid var(--line)} .modebar.mock{background:#101c2e;color:#9fb6d6}
.modebar.genuine{background:#2a230f;color:#ffe39a;border-color:#7b652d}
button,.button{border:0;border-radius:10px;background:var(--blue);color:#04101f;padding:11px 16px;
font-weight:800;cursor:pointer;text-decoration:none;display:inline-block} .danger{background:var(--red);color:#210a0c}
input,select{width:100%;padding:12px;border-radius:10px;
border:1px solid var(--line);background:#091426;color:var(--ink);margin:6px 0 14px} label{color:var(--muted)}
code{color:#bcd8ff;word-break:break-all} .timeline{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.timeline span{padding:7px 10px;border:1px solid var(--line);border-radius:9px;color:var(--muted)}
blockquote{margin:10px 0;border-left:3px solid var(--blue);padding:6px 12px;color:#cbd8eb;background:#081426}
a{color:#91c4ff} nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px}
table{width:100%;border-collapse:collapse;font-size:.86rem} th,td{text-align:left;padding:8px 6px;
border-bottom:1px solid var(--line);vertical-align:top} th{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto}
"""


def _page(title: str, body: str, *, config: RuntimeConfig) -> str:
    css_mode = "genuine" if config.is_genuine else "mock"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · MissionPay</title><style>{_STYLES}</style></head>
<body><main><nav><strong>MISSIONPAY</strong><span class="eyebrow">Proof → authority → reimbursement</span></nav>
<div class="modebar {css_mode}">MODE: {html.escape(config.mode_label)}</div>
{body}</main></body></html>"""


def _error_page(title: str, code: str, detail: str, *, config: RuntimeConfig, status: int) -> HTMLResponse:
    body = f"""<div class="eyebrow">Refused</div><h1>{html.escape(title)}</h1>
    <section class="card"><h2>{html.escape(code)}</h2><p>{html.escape(detail)}</p>
    <p><a href="/app">Return to workspace</a></p></section>"""
    return HTMLResponse(_page(title, body, config=config), status_code=status)


def _model_identity(config: RuntimeConfig) -> str:
    """The identity recorded for the model that ran, and shown to the reviewer.

    One function, because the page and the authority must never name different models:
    a divergence there is exactly how a reviewer approves one configuration while the
    dispatch binds another.
    """
    if config.ai_mode is AiMode.OLLAMA:
        return config.ollama_model or "unconfigured"
    if config.ai_mode is AiMode.HOSTED:
        # Provider-qualified: the same model name served by two providers is two
        # different products, and a stored id must not be ambiguous.
        return f"openai:{config.hosted_model or 'unconfigured'}"
    return _DemoInterpretationModel.model_id


def _interpret(
    config: RuntimeConfig, evidence: EvidenceIdentity, text: str
) -> tuple[
    ReceiptInterpretation | None, str | None, dict[str, int] | None, dict[str, object] | None
]:
    """Interpret evidence, returning ``(result, error, usage)``.

    ``usage`` is the provider's reported token counters when one was returned. It is
    carried out of here rather than dropped because an interpretation whose cost cannot
    be recorded is not evidence a reviewer can audit.
    """
    if config.ai_mode is AiMode.OLLAMA and config.ollama_model:
        model = OllamaJsonModel(
            base_url=config.ollama_base_url, model_id=config.ollama_model
        )
    elif config.ai_mode is AiMode.HOSTED:
        try:
            model = HostedJsonModel(
                base_url=config.hosted_base_url,
                model_id=config.hosted_model or "",
                api_key=config.hosted_api_key or "",
                max_output_tokens=config.hosted_max_output_tokens or 0,
                max_total_tokens=config.hosted_max_total_tokens or 0,
                timeout_seconds=config.hosted_timeout_seconds,
                late_response_seconds=config.hosted_late_response_seconds,
            )
        except HostedModelError as exc:
            # Configuration that passed validation but cannot construct an adapter is a
            # refusal, not a crash: the claim stays uninterpreted and says why.
            return None, exc.code, None, None
    else:
        model = _DemoInterpretationModel()
    try:
        result = interpret_receipt_text(evidence=evidence, text=text, model=model)
    except (ModelInterpretationError, ModelInvocationError, HostedModelError) as exc:
        usage = getattr(model, "last_usage", None)
        # A citation refusal carries where it failed, never what it contained. Sanitised
        # here, once, so the audit event and any rendering share one cleaned object rather
        # than each deciding for itself what may be shown.
        diagnostic = clean_diagnostic(getattr(exc, "diagnostic", None))
        return (
            None,
            str(exc),
            usage if isinstance(usage, dict) else None,
            diagnostic if isinstance(diagnostic, dict) else None,
        )
    usage = getattr(model, "last_usage", None)
    return result, None, usage if isinstance(usage, dict) else None, None




def _generation_identity(config: RuntimeConfig) -> dict[str, object]:
    """Everything besides the prompt that can change the model's answer.

    Kept separate from :func:`_model_identity` because a stored model name is a label,
    while the digest is a claim that two runs are interchangeable. A provider change,
    or a larger output ceiling, makes them not interchangeable even when the label is
    identical — and the reuse lookup must see that before it serves a cached fact set.
    """
    if config.ai_mode is AiMode.HOSTED:
        return {
            "provider": "openai",
            "base_url": config.hosted_base_url.rstrip("/"),
            "max_output_tokens": config.hosted_max_output_tokens,
        }
    if config.ai_mode is AiMode.OLLAMA:
        return {
            "provider": "ollama-loopback",
            "base_url": config.ollama_base_url.rstrip("/"),
        }
    return {"provider": "fixture"}


def _interpretation_config_digest(config: RuntimeConfig, claim: Claim, model_id: str) -> str:
    """Identity of the configuration that produced an interpretation.

    Mode, model, provider and prompt version are what a reviewer is implicitly trusting
    when they accept a set of facts, so they are recorded and digested rather than
    assumed. Switching provider or model therefore misses the reuse lookup and runs
    inference instead of serving the previous answer.
    """
    return sha256(
        json.dumps(
            {
                "ai_mode": config.ai_mode.value,
                "model_id": model_id,
                "prompt_version": PROMPT_VERSION,
                "evidence_sha256": claim.evidence.sha256,
                "generation": _generation_identity(config),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def create_app(
    store: MissionPayStore,
    *,
    config: RuntimeConfig | None = None,
    service: PaymentService | None = None,
    secure_cookies: bool | None = None,
    evidence_store: EvidenceStore | None = None,
) -> FastAPI:
    if config is None:
        config = RuntimeConfig(
            database_path=str(store.path),
            secure_cookies=True if secure_cookies is None else secure_cookies,
        )
    elif secure_cookies is not None:
        config = replace(config, secure_cookies=secure_cookies)

    if service is None:
        service = PaymentService(store, config)
    if evidence_store is None:
        evidence_store = EvidenceStore(default_evidence_root(store.path))

    app = FastAPI(title="MissionPay", docs_url=None, redoc_url=None)
    # Enforced in front of routing, so the body is bounded before multipart parsing
    # can spool it to disk.
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

    def cookie_options() -> dict[str, object]:
        return {"secure": config.secure_cookies, "samesite": "strict", "path": "/"}

    def session_from_request(request: Request):
        token = request.cookies.get("mp_session", "")
        if not token:
            raise AuthenticationFailed("SESSION_REQUIRED")
        return store.authenticated_session(token)

    def require_csrf(request: Request, submitted: str) -> str:
        token = request.cookies.get("mp_session", "")
        cookie = request.cookies.get("mp_csrf", "")
        if not token or not cookie or not secrets.compare_digest(cookie, submitted):
            raise AuthorizationDenied("CSRF_INVALID")
        store.verify_csrf(token, submitted)
        return token

    @app.exception_handler(AuthenticationFailed)
    async def authentication_failed(_request: Request, _exc: AuthenticationFailed):
        return RedirectResponse("/login", status_code=303)

    @app.exception_handler(AuthorizationDenied)
    async def authorization_denied(_request: Request, exc: AuthorizationDenied):
        return _error_page(
            "Forbidden", str(exc), "This action is not permitted for this identity.",
            config=config, status=403,
        )

    # -- authentication --------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return RedirectResponse("/app", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        csrf = secrets.token_urlsafe(32)
        body = f"""<section style="max-width:520px;margin:9vh auto" class="card">
        <div class="eyebrow">Local reimbursement control plane</div><h1>MissionPay</h1>
        <p>Evidence can inform a claim. Only policy and an independent human can authorize it.</p>
        <form method="post" action="/login">
        <input type="hidden" name="csrf_token" value="{csrf}">
        <label>Email <input name="email" type="email" required></label>
        <label>Password <input name="password" type="password" required></label>
        <button type="submit">Enter MissionPay</button></form></section>"""
        response = HTMLResponse(_page("Sign in", body, config=config))
        response.set_cookie("mp_preauth", csrf, httponly=True, **cookie_options())
        return response

    @app.post("/login")
    async def login(
        request: Request,
        email: str = Form(),
        password: str = Form(),
        csrf_token: str = Form(),
        mp_preauth: str | None = Cookie(default=None),
    ):
        if not mp_preauth or not secrets.compare_digest(mp_preauth, csrf_token):
            raise AuthorizationDenied("LOGIN_CSRF_INVALID")
        session = store.login(email, password)
        response = RedirectResponse("/app", status_code=303)
        response.delete_cookie("mp_preauth", path="/")
        response.set_cookie(
            "mp_session", session.token, httponly=True, max_age=8 * 60 * 60, **cookie_options()
        )
        response.set_cookie(
            "mp_csrf", session.csrf_token, httponly=False, max_age=8 * 60 * 60, **cookie_options()
        )
        return response

    @app.post("/logout")
    async def logout(request: Request, csrf_token: str = Form()):
        token = require_csrf(request, csrf_token)
        store.revoke_session(token)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie("mp_session", path="/")
        response.delete_cookie("mp_csrf", path="/")
        return response

    # -- workspace -------------------------------------------------------

    @app.get("/app", response_class=HTMLResponse)
    async def dashboard(request: Request):
        session = session_from_request(request)
        csrf = request.cookies.get("mp_csrf", "")
        heading = "Volunteer workspace" if session.role is Role.VOLUNTEER else "Reviewer workspace"
        action = (
            f"""<form method="post" action="/claims/demo">
            <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <button type="submit">Load the bundled synthetic expense</button></form>"""
            if session.role is Role.VOLUNTEER
            else "<a class='button' href='/review'>Open review queue</a>"
        )
        upload = ""
        if session.role is Role.VOLUNTEER:
            upload = f"""<section class="card wide"><h2>Submit your own evidence</h2>
            <p>Text, PNG or JPEG up to 256 KiB. The file is stored under a
            content-derived name; the filename you choose never becomes a path.</p>
            <form method="post" action="/claims/upload" enctype="multipart/form-data">
            <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <label>Receipt file <input type="file" name="evidence" required></label>
            <label>Claimed amount (USD cents) <input name="amount_cents" type="number" min="1" value="7899" required></label>
            <label>Activity <select name="activity_code">
            <option value="RIVER_CLEANUP">River cleanup</option>
            <option value="COMMUNITY_GARDEN">Community garden</option></select></label>
            <label>Expense category <input name="expense_category" value="PROTECTIVE_SUPPLIES"></label>
            <button type="submit">Submit for policy review</button></form></section>"""
        return HTMLResponse(_page("Workspace", f"""<div class="eyebrow">Authenticated · {session.role.value}</div>
            <h1>{heading}</h1><p>One controlled scenario: protective supplies for a river cleanup.</p>
            <div class="grid"><section class="card"><h2>Bundled expense</h2><div class="amount">$78.99</div>
            <p>Nitrile gloves and safety glasses · synthetic receipt fixture.</p>{action}</section>
            <section class="card"><h2>Authority separation</h2><p>AI extracts candidate facts. Restricted-fund
            policy decides eligibility. A separate reviewer freezes the exact payment.</p></section>{upload}</div>
            <form method="post" action="/logout"><input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <button type="submit">Sign out</button></form>""", config=config))

    def _recipient_for(session) -> str:
        """Genuine mode pays a nominated sandbox account, never a real address."""
        if config.is_genuine and config.sandbox_recipient:
            return config.sandbox_recipient
        return store.user_email(session.user_id)

    @app.post("/claims/demo")
    async def submit_demo(request: Request, csrf_token: str = Form()):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.VOLUNTEER:
            raise AuthorizationDenied("VOLUNTEER_ROLE_REQUIRED")

        stored = evidence_store.save(
            data=DEMO_RECEIPT.encode("utf-8"),
            media_type="text/plain",
            filename="harbor-safety-receipt.txt",
        )
        claim = Claim(
            claim_id="river-safety-001",
            volunteer_id=session.user_id,
            activity=DEMO_ACTIVITY,
            evidence=EvidenceIdentity(stored.sha256, "text/plain", stored.original_name),
            requested=Money("USD", 7899),
            recipient=_recipient_for(session),
            expense_category="PROTECTIVE_SUPPLIES",
        )
        try:
            store.submit_claim(token, claim, revision=1)
            for fund_id in ("GARDEN_FUND", "RIVER_CLEANUP_FUND"):
                policy = store.load_fund_policy(fund_id, "2026.1")
                store.record_policy_decision(
                    evaluate_claim(claim, policy, prior_evidence_digests=frozenset()),
                    revision=1,
                )
        except PersistenceConflict:
            return _error_page(
                "Duplicate blocked", "DUPLICATE_REIMBURSEMENT_BLOCKED",
                "This exact evidence was already reimbursed or is already in progress.",
                config=config, status=409,
            )
        return RedirectResponse("/claims/river-safety-001", status_code=303)

    @app.post("/claims/upload")
    async def submit_upload(
        request: Request,
        csrf_token: str = Form(),
        evidence: UploadFile = File(...),
        amount_cents: int = Form(),
        activity_code: str = Form(),
        expense_category: str = Form(),
    ):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.VOLUNTEER:
            raise AuthorizationDenied("VOLUNTEER_ROLE_REQUIRED")

        try:
            raw = await read_upload_bounded(evidence)
            stored = evidence_store.save(
                data=raw,
                media_type=evidence.content_type or "",
                filename=evidence.filename or "",
            )
        except EvidenceRejected as exc:
            return _error_page(
                "Evidence refused", str(exc),
                "The upload did not meet the intake rules, so nothing was stored.",
                config=config, status=400,
            )

        if amount_cents <= 0:
            return _error_page(
                "Amount refused", "AMOUNT_NOT_POSITIVE",
                "A reimbursement amount must be a positive number of cents.",
                config=config, status=400,
            )
        try:
            claim = Claim(
                claim_id=f"claim-{stored.sha256[:16]}",
                volunteer_id=session.user_id,
                activity=Activity(f"activity-{activity_code.lower()}", activity_code, DEMO_ACTIVITY.occurred_on),
                evidence=EvidenceIdentity(stored.sha256, stored.media_type, stored.original_name),
                requested=Money("USD", amount_cents),
                recipient=_recipient_for(session),
                expense_category=expense_category.strip().upper() or None,
            )
        except ValueError as exc:
            return _error_page(
                "Claim refused", "CLAIM_FIELDS_INVALID", str(exc), config=config, status=400
            )

        try:
            store.submit_claim(token, claim, revision=1)
        except PersistenceConflict:
            return _error_page(
                "Duplicate blocked", "DUPLICATE_REIMBURSEMENT_BLOCKED",
                "This exact evidence was already submitted.",
                config=config, status=409,
            )
        for fund_id in ("GARDEN_FUND", "RIVER_CLEANUP_FUND"):
            try:
                policy = store.load_fund_policy(fund_id, "2026.1")
            except PersistenceConflict:
                continue
            store.record_policy_decision(
                evaluate_claim(claim, policy, prior_evidence_digests=frozenset()), revision=1
            )
        return RedirectResponse(f"/claims/{claim.claim_id}", status_code=303)

    # -- review ----------------------------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    async def review_queue(request: Request):
        token = request.cookies.get("mp_session", "")
        rows = store.review_queue(token)
        cards = "".join(
            f"""<section class="card"><div class="eyebrow">Awaiting independent review</div>
            <h2>{html.escape(str(row['claim_id']))}</h2><div class="amount">${row['amount_cents'] / 100:.2f}</div>
            <p>{html.escape(str(row['recipient']))}</p>
            <a class="button" href="/claims/{row['claim_id']}">Inspect frozen claim</a></section>"""
            for row in rows
        ) or "<section class='card'><p>No eligible claims await review.</p></section>"
        return HTMLResponse(_page("Review queue", f"<div class='eyebrow'>Reviewer queue</div>"
            f"<h1>Independent authorization</h1><div class='grid'>{cards}</div>", config=config))

    def _evidence_text(claim: Claim) -> str | None:
        return evidence_store.read_text(claim.evidence.sha256)

    def _interpretation_panel(claim, record, *, can_run: bool, csrf: str) -> str:
        """Render the *persisted* interpretation. Never runs inference."""
        run_form = ""
        if can_run:
            run_form = f"""<form method="post" action="/claims/{html.escape(claim.claim_id)}/interpret">
            <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <button type="submit">Run AI interpretation</button></form>"""

        if record is None:
            return f"""<section class="card"><h3>AI interpretation</h3>
            <div class="banner">NOT RUN — THE AI STAGE IS INCOMPLETE</div>
            <p>No interpretation has been recorded for this evidence. Approval is blocked
            until one is accepted, because a reviewer cannot be said to have reviewed an
            interpretation that does not exist.</p>{run_form}</section>"""

        source_label = (
            f"LIVE LOCAL MODEL — {html.escape(str(record['model_id']))}"
            if str(record["ai_mode"]) == AiMode.OLLAMA.value
            else (
                f"LIVE HOSTED MODEL — {html.escape(str(record['model_id']))}"
                if str(record["ai_mode"]) == AiMode.HOSTED.value
                else "SIMULATED INTERPRETATION FIXTURE — NO LIVE MODEL INVOKED"
            )
        )
        banner = f'<div class="banner">{source_label}</div>'
        identity = (
            f"<p><code>interpretation {html.escape(str(record['interpretation_id']))}</code> · "
            f"model {html.escape(str(record['model_id']))} · "
            f"config {html.escape(str(record['config_sha256'])[:16])}… · "
            f"output {html.escape(str(record['output_sha256'])[:16])}…</p>"
        )

        if str(record["status"]) == "REFUSED":
            return f"""<section class="card"><h3>AI interpretation</h3>{banner}
            <p class="bad">Interpretation refused: {html.escape(str(record.get('error_code') or 'UNKNOWN'))}</p>
            <p>A refusal is recorded as evidence and cannot satisfy the AI stage. No facts
            were accepted and no authority changed.</p>{identity}{run_form}</section>"""

        facts_data = json.loads(record["facts_json"] or "[]")
        missing_data = json.loads(record["missing_json"] or "[]")
        labels = {"observed": "Observed fact", "inference": "Inference", "uncertain": "Uncertainty"}
        def _span_note(fact: dict) -> str:
            """Say where the offsets came from, because it is not always the model.

            A reviewer approving this span is entitled to know whether the model got the
            arithmetic right or whether the span was located from the quote. Records
            written before this field existed have no value and are shown as unrecorded
            rather than guessed at.
            """
            number = fact.get("excerpt_id")
            if isinstance(number, int) and not isinstance(number, bool):
                return (
                    f"excerpt {number} chosen by the model from a catalogue of exact "
                    "receipt text; the span was resolved by MissionPay"
                )
            derived = fact.get("offsets_derived")
            if derived is True:
                return "span located from the quote, not supplied by the model (earlier contract)"
            if derived is False:
                return "span as supplied by the model (earlier contract)"
            return "span provenance not recorded in this record"

        facts = "".join(
            f"""<div><span class="pill">{labels.get(str(fact.get('epistemic_status')), 'Unknown')}</span>
            <strong>{html.escape(str(fact.get('name', '')).replace('_', ' ').title())}</strong>:
            {html.escape(str(fact.get('value', '')))}
            <blockquote>Citation: “{html.escape(str(fact.get('quote', '')))}” ·
            chars {fact.get('start')}–{fact.get('end')}<br>
            <small>{html.escape(_span_note(fact))}</small></blockquote></div>"""
            for fact in facts_data
        )
        stored = str(record.get("contract_version") or "").strip()

        def _contract_note() -> str:
            """Say what rule this record was actually judged under — not what the rule is today.

            An interpretation accepted before the uniqueness contract was introduced was
            checked only for an exact span. Printing today's sentence beneath it would claim
            a check that never ran on this evidence, so the older case gets its own wording.
            """
            if stored == PROMPT_VERSION:
                return (
                    "A citation shows that the quoted text appears exactly once in the "
                    "receipt. It does not show that the quote supports the fact, or that "
                    "the fact means what the model says it means — that is your judgement, "
                    "not the machine's."
                )
            # Say what is known and no more. A NULL version does not prove the old contract
            # lacked the uniqueness rule — it proves we did not record which contract applied.
            # Asserting what the earlier rule required would be inventing history.
            if stored:
                where = (
                    f"recorded under contract {html.escape(stored)}, which is not the contract "
                    "in force today"
                )
            else:
                where = (
                    "recorded before contract versions were stored, so the contract it was "
                    "judged under is unrecorded"
                )
            return (
                f"This interpretation was {where}. Its citations were checked for an exact "
                "match against the receipt at the time; whether each quote would also satisfy "
                "the current uniqueness rule is unverified. Treat these citations as historical "
                "evidence, not as a qualification under the current contract."
            )

        return f"""<section class="card"><h3>AI interpretation</h3>{banner}{facts}
        <p><strong>Uncertainty / missing:</strong>
        {html.escape(', '.join(str(m) for m in missing_data))}</p>
        <p>{_contract_note()}</p>
        <p>Interpretation proposes facts. It does not decide eligibility, change a fund,
        alter the payee or amount, or authorize a payment.</p>{identity}{run_form}</section>"""

    def _observation_panel(operation_key: str | None) -> str:
        if operation_key is None:
            return ""
        history = store.reconciliation_history(operation_key)
        if not history:
            return """<p>No independent provider observation has been retrieved yet.
            An accepted submission is not a completed payment.</p>"""
        rows = "".join(
            f"""<tr><td>{html.escape(str(entry['observed_at']))}</td>
            <td>{html.escape(str(entry['provider_status']))}</td>
            <td>{html.escape(str(entry['outcome']))}</td>
            <td>{html.escape(', '.join(entry['reason_codes']))}</td>
            <td><code>{html.escape(str(entry['payout_item_id']))}</code></td>
            <td><code>{html.escape(str(entry['raw_sha256'])[:16])}…</code></td></tr>"""
            for entry in history
        )
        return f"""<div class="scroll"><table><thead><tr><th>Observed</th><th>Provider status</th>
        <th>Reconciliation</th><th>Reasons</th><th>Item</th><th>Evidence digest</th></tr></thead>
        <tbody>{rows}</tbody></table></div>"""

    def _audit_panel(operation_key: str | None) -> str:
        if operation_key is None:
            return ""
        trail = store.audit_trail(operation_key)
        if not trail:
            return ""
        rows = "".join(
            f"""<tr><td>{html.escape(str(event['occurred_at']))}</td>
            <td>{html.escape(str(event['event_type']))}</td>
            <td><code>{html.escape(json.dumps(event['payload'], sort_keys=True))}</code></td></tr>"""
            for event in trail
        )
        return f"""<div class="scroll"><table><thead><tr><th>When</th><th>Event</th>
        <th>Payload</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    @app.get("/claims/{claim_id}", response_class=HTMLResponse)
    async def claim_detail(request: Request, claim_id: str):
        token = request.cookies.get("mp_session", "")
        snapshot = store.claim_snapshot(token, claim_id)
        claim = snapshot["claim"]
        session = session_from_request(request)
        csrf = request.cookies.get("mp_csrf", "")
        operation_key = snapshot["operation_key"]
        state = snapshot["operation_state"]

        decisions = "".join(
            f"""<section class="card"><h2>{html.escape(str(item['fund_id']).replace('_', ' ').title())}</h2>
            <div class="{'good' if item['outcome'] == 'eligible' else 'bad'}">
            <strong>{'Eligible' if item['outcome'] == 'eligible' else 'Refused'}</strong></div>
            <p>{html.escape(', '.join(item['reason_codes']))}</p>
            <p class="eyebrow">policy {html.escape(str(item['policy_version']))}</p></section>"""
            for item in snapshot["decisions"]
        )

        approval = ""
        accepted = store.accepted_interpretation(claim_id, snapshot["revision"])
        if session.role is Role.REVIEWER and snapshot["authorization_id"] is None and accepted:
            approval = f"""<section class="card wide"><h2>Human authorization</h2>
            <p>Freeze recipient <code>{html.escape(claim.recipient)}</code>,
            the eligible fund at its recorded policy version, and exactly
            <strong>${claim.requested.cents / 100:.2f}</strong>, against interpretation
            <code>{html.escape(str(accepted['interpretation_id']))}</code>
            (output <code>{html.escape(str(accepted['output_sha256'])[:16])}…</code>).</p>
            <form method="post" action="/claims/{html.escape(claim_id)}/approve">
            <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <input type="hidden" name="revision" value="{snapshot['revision']}">
            <input type="hidden" name="amount_cents" value="{claim.requested.cents}">
            <input type="hidden" name="interpretation_id"
                   value="{html.escape(str(accepted['interpretation_id']))}">
            <input type="hidden" name="interpretation_output_sha256"
                   value="{html.escape(str(accepted['output_sha256']))}">
            <button type="submit">Authorize exact reimbursement</button></form></section>"""
        elif session.role is Role.REVIEWER and snapshot["authorization_id"] is None:
            approval = """<section class="card wide"><h2>Human authorization</h2>
            <p class="warn">Blocked: no accepted interpretation exists for this evidence.
            Run the AI stage before reviewing, because an approval must name the output
            the reviewer actually saw.</p></section>"""

        authority = ""
        if snapshot["authorization_id"]:
            authority = f"""<section class="card wide"><h2>Human authorization</h2>
            <div class="good">Exact authority frozen</div>
            <p>Reviewer {html.escape(str(snapshot['reviewer_id']))} ·
            <code>{html.escape(str(snapshot['authorization_id']))}</code></p></section>"""

        payment = _payment_panel(
            config=config,
            service=service,
            session=session,
            csrf=csrf,
            claim_id=claim_id,
            operation_key=operation_key,
            state=state,
        )

        return HTMLResponse(_page("Claim", f"""<div class="eyebrow">Claim {html.escape(claim_id)} ·
        revision {snapshot['revision']}</div>
        <h1>{html.escape(claim.evidence.original_name)}</h1><div class="grid">
        <section class="card"><h3>Expense evidence</h3>
        <div class="amount">${claim.requested.cents / 100:.2f}</div>
        <p>{html.escape(claim.activity.activity_code.replace('_', ' ').title())} ·
        {html.escape(claim.expense_category or 'uncategorised')}</p>
        <blockquote>{html.escape((_evidence_text(claim) or '(binary evidence preserved unmodified)'))}</blockquote>
        <p><code>SHA-256 {html.escape(claim.evidence.sha256)}</code></p></section>
        {_interpretation_panel(claim, store.accepted_interpretation(claim_id, snapshot['revision']) or (store.interpretation_history(claim_id, snapshot['revision']) or [None])[-1], can_run=True, csrf=csrf)}
        {decisions}{approval}{authority}{payment}</div>""", config=config))

    @app.post("/claims/{claim_id}/interpret")
    async def run_interpretation(request: Request, claim_id: str, csrf_token: str = Form()):
        """Explicit, persisted AI stage. Never triggered by a GET or a refresh."""
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        snapshot = store.claim_snapshot(token, claim_id)
        claim = snapshot["claim"]
        revision = int(snapshot["revision"])

        if snapshot["authorization_id"] is not None:
            return _error_page(
                "Already authorized", "INTERPRETATION_AFTER_AUTHORIZATION",
                "This claim has been authorized. A new interpretation would invalidate that "
                "approval, so it cannot be attached to an approved claim revision.",
                config=config, status=409,
            )

        text = _evidence_text(claim)
        model_id = _model_identity(config)
        config_sha256 = _interpretation_config_digest(config, claim, model_id)

        # Reuse is decided on the configuration that is about to run. Asking only
        # "does some accepted interpretation exist" would serve a fixture result when
        # Ollama was requested, which is both wrong and invisible.
        existing = store.accepted_interpretation_for_config(
            claim_id,
            revision,
            claim.evidence.sha256,
            config.ai_mode.value,
            model_id,
            config_sha256,
        )
        if existing is not None:
            return RedirectResponse(f"/claims/{claim_id}", status_code=303)

        if text is None:
            store.record_interpretation(
                claim_id=claim_id,
                revision=revision,
                evidence_sha256=claim.evidence.sha256,
                ai_mode=config.ai_mode.value,
                model_id=model_id,
                config_sha256=config_sha256,
                status="REFUSED",
                output_sha256=sha256(b"EVIDENCE_TEXT_UNAVAILABLE").hexdigest(),
                error_code="EVIDENCE_TEXT_UNAVAILABLE",
            )
            return RedirectResponse(f"/claims/{claim_id}", status_code=303)

        result, error, usage, diagnostic = _interpret(config, claim.evidence, text)
        # Recorded before the branch: a refused answer that still cost tokens is a fact
        # about the spend, and the audit trail is where an operator looks for it.
        if usage:
            store.record_interpretation_usage(
                claim_id, revision, usage, model_id=model_id
            )
        if diagnostic:
            # Where the citation failed, so an operator can see it without the page
            # ever rendering the quote the model returned.
            store.record_citation_refusal(
                claim_id, revision, diagnostic, model_id=model_id
            )
        if result is None:
            store.record_interpretation(
                claim_id=claim_id,
                revision=revision,
                evidence_sha256=claim.evidence.sha256,
                ai_mode=config.ai_mode.value,
                model_id=model_id,
                config_sha256=config_sha256,
                status="REFUSED",
                output_sha256=sha256(f"REFUSED:{error}".encode()).hexdigest(),
                error_code=error or "INTERPRETATION_REFUSED",
            )
            return RedirectResponse(f"/claims/{claim_id}", status_code=303)

        facts = [
            {
                "name": fact.name,
                "value": fact.value,
                "epistemic_status": fact.epistemic_status.value,
                "start": fact.citation.start,
                "end": fact.citation.end,
                "quote": fact.citation.quote,
                # Recorded, not inferred later: a reviewer approving this span is entitled
                # to know it was located in the text rather than supplied by the model.
                "offsets_derived": fact.offsets_derived,
                "excerpt_id": fact.citation.excerpt_id,
            }
            for fact in result.facts
        ]
        output_sha256 = sha256(
            json.dumps(
                {"facts": facts, "missing": list(result.missing)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        store.record_interpretation(
            claim_id=claim_id,
            revision=revision,
            evidence_sha256=claim.evidence.sha256,
            ai_mode=config.ai_mode.value,
            model_id=model_id,
            config_sha256=config_sha256,
            status="ACCEPTED",
            output_sha256=output_sha256,
            facts=facts,
            missing=result.missing,
            # What this answer was actually judged under, so a later rule change cannot
            # silently re-label an older record as compliant with the new one.
            contract_version=PROMPT_VERSION,
        )
        return RedirectResponse(f"/claims/{claim_id}", status_code=303)

    @app.post("/claims/{claim_id}/approve")
    async def approve_claim(
        request: Request,
        claim_id: str,
        csrf_token: str = Form(),
        revision: int = Form(),
        amount_cents: int = Form(),
        interpretation_id: str = Form(default=""),
        interpretation_output_sha256: str = Form(default=""),
    ):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.REVIEWER:
            raise AuthorizationDenied("REVIEWER_ROLE_REQUIRED")
        snapshot = store.claim_snapshot(token, claim_id)
        claim = snapshot["claim"]
        if revision != snapshot["revision"]:
            return _error_page("Frozen revision mismatch", "FROZEN_REVISION_MISMATCH",
                "The claim changed after you opened it. Review it again.", config=config, status=409)
        if amount_cents != claim.requested.cents:
            return _error_page("Frozen amount mismatch", "FROZEN_AMOUNT_MISMATCH",
                "The amount you approved is not the amount the claim requests.",
                config=config, status=409)
        if store.existing_operation_for_claim(claim_id, revision):
            return _error_page("Duplicate blocked", "DUPLICATE_REIMBURSEMENT_BLOCKED",
                "A payment intent already exists for this claim revision.",
                config=config, status=409)
        authorization_id = store.existing_authorization_id(claim_id, revision)
        if authorization_id is None:
            try:
                authorization_id = store.authorize_claim(
                    token,
                    claim_id,
                    revision=revision,
                    interpretation_id=interpretation_id,
                    interpretation_output_sha256=interpretation_output_sha256,
                ).authorization_id
            except AuthorizationDenied as exc:
                return _error_page(
                    "Approval refused",
                    str(exc),
                    "The interpretation you reviewed is not the one currently accepted, "
                    "or is not bound to this evidence. Reload the claim and review it again.",
                    config=config,
                    status=409,
                )
        store.create_payment_intent(authorization_id)
        return RedirectResponse(f"/claims/{claim_id}", status_code=303)

    # -- genuine execution surface --------------------------------------

    @app.post("/claims/{claim_id}/pay")
    async def pay_claim(request: Request, claim_id: str, csrf_token: str = Form()):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.REVIEWER:
            raise AuthorizationDenied("REVIEWER_ROLE_REQUIRED")
        snapshot = store.claim_snapshot(token, claim_id)
        operation_key = snapshot["operation_key"]
        if operation_key is None:
            return _error_page("No authority", "PAYMENT_INTENT_REQUIRED",
                "Authorize the exact reimbursement before executing it.",
                config=config, status=409)
        if not service.can_execute():
            return _error_page("Mock mode", "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS",
                "This build is in mock mode. It will not contact PayPal. Set "
                "MISSIONPAY_PROVIDER_MODE=paypal_sandbox with sandbox credentials to execute.",
                config=config, status=409)
        try:
            outcome = service.execute(operation_key, owner_id=session.user_id)
        except DispatchRefused as exc:
            return _error_page("Dispatch refused", str(exc),
                "Nothing was sent. The operation remains in its prior state; no payout "
                "request was transmitted.", config=config, status=409)
        except PaymentServiceRefused as exc:
            return _error_page("Refused", exc.reason, "The configured mode refused this action.",
                config=config, status=409)
        return RedirectResponse(f"/claims/{claim_id}", status_code=303)

    @app.post("/claims/{claim_id}/reconcile")
    async def reconcile_claim(request: Request, claim_id: str, csrf_token: str = Form()):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.REVIEWER:
            raise AuthorizationDenied("REVIEWER_ROLE_REQUIRED")
        snapshot = store.claim_snapshot(token, claim_id)
        operation_key = snapshot["operation_key"]
        if operation_key is None:
            return _error_page("No authority", "PAYMENT_INTENT_REQUIRED",
                "There is no payment to reconcile.", config=config, status=409)
        if not service.can_execute():
            return _error_page("Mock mode", "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS",
                "Mock mode performs no provider readback.", config=config, status=409)
        try:
            service.observe(operation_key)
        except (ValueError, PaymentServiceRefused) as exc:
            return _error_page("Readback refused", type(exc).__name__,
                str(exc) or "The provider readback could not be performed.",
                config=config, status=409)
        return RedirectResponse(f"/claims/{claim_id}", status_code=303)

    @app.post("/claims/{claim_id}/attach-batch")
    async def attach_batch(
        request: Request, claim_id: str, csrf_token: str = Form(), payout_batch_id: str = Form()
    ):
        token = require_csrf(request, csrf_token)
        session = store.authenticated_session(token)
        if session.role is not Role.REVIEWER:
            raise AuthorizationDenied("REVIEWER_ROLE_REQUIRED")
        snapshot = store.claim_snapshot(token, claim_id)
        operation_key = snapshot["operation_key"]
        if operation_key is None:
            return _error_page("No authority", "PAYMENT_INTENT_REQUIRED",
                "There is no ambiguous operation to resolve.", config=config, status=409)
        candidate = payout_batch_id.strip()
        if not candidate:
            return _error_page("Batch id required", "PAYOUT_BATCH_ID_REQUIRED",
                "An authoritative PayPal batch id is required. MissionPay will not guess one.",
                config=config, status=400)
        if not service.can_execute():
            return _error_page("Mock mode", "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS",
                "Mock mode cannot reconcile against a provider.", config=config, status=409)
        try:
            service.attach_batch(operation_key, candidate)
        except (ValueError, PaymentServiceRefused) as exc:
            return _error_page("Binding refused", type(exc).__name__,
                str(exc) or "The batch id could not be bound.", config=config, status=409)
        return RedirectResponse(f"/claims/{claim_id}", status_code=303)

    return app


_STATE_LABELS = {
    "PREPARED": "Not attempted",
    "SENDING": "Sending",
    "UNCERTAIN": "Uncertain",
    "ACKNOWLEDGED": "Acknowledged",
    "VERIFIED": "Independently reconciled",
    "CONFLICT": "Conflict",
}


def _payment_panel(
    *,
    config: RuntimeConfig,
    service: PaymentService,
    session,
    csrf: str,
    claim_id: str,
    operation_key: str | None,
    state: str | None,
) -> str:
    """Render the payment stage strictly from persisted state."""
    genuine = config.is_genuine
    banner = (
        '<div class="banner">GENUINE PAYPAL SANDBOX — SANDBOX FUNDS, NOT REAL MONEY MOVED</div>'
        if genuine
        else '<div class="banner">SIMULATED — NO PAYPAL PAYMENT EXECUTED</div>'
    )

    if operation_key is None:
        return f"""<section class="card wide"><h2>PayPal execution &amp; independent reconciliation</h2>
        {banner}<p>Authoritative state: <strong>No payment intent</strong></p>
        <p>No authorized payment exists yet, so nothing can be executed or reconciled.</p></section>"""

    financial = store_outcome(service.store, operation_key)
    outcome_text = str(financial.get("outcome") or "not yet observed")
    provider_status = financial.get("provider_status")

    controls = ""
    if session.role is not Role.REVIEWER:
        controls = "<p class='eyebrow'>Reviewer authority required for payment actions.</p>"
    elif state == "PREPARED":
        if genuine:
            controls = f"""<form method="post" action="/claims/{html.escape(claim_id)}/pay">
            <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
            <button type="submit" class="danger">Execute one sandbox payout</button></form>
            <p>This sends exactly one payout request. A second click is refused by the
            database, not by the button.</p>"""
        else:
            controls = """<p class="warn">Mock mode will not contact PayPal. The intent is
            authorized and held; execution requires sandbox mode.</p>"""
    elif state == "UNCERTAIN":
        controls = f"""<p class="bad">Outcome unresolved. A payout request may or may not have
        reached PayPal. MissionPay will not send another one.</p>
        <p>Bind an authoritative PayPal batch id to reconcile this operation:</p>
        <form method="post" action="/claims/{html.escape(claim_id)}/attach-batch">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
        <label>Payout batch id <input name="payout_batch_id" required></label>
        <button type="submit">Bind and reconcile</button></form>"""
    elif state in {"ACKNOWLEDGED", "VERIFIED", "CONFLICT"}:
        controls = f"""<form method="post" action="/claims/{html.escape(claim_id)}/reconcile">
        <input type="hidden" name="csrf_token" value="{html.escape(csrf)}">
        <button type="submit">Retrieve provider item and reconcile</button></form>
        <p>Submission acceptance is not payment completion. The verdict below comes from
        an independently retrieved provider item.</p>"""

    history = ""
    try:
        entries = service.store.reconciliation_history(operation_key)
    except Exception:  # noqa: BLE001 - history is optional presentation
        entries = []
    if entries:
        rows = "".join(
            f"""<tr><td>{html.escape(str(e['observed_at']))}</td>
            <td>{html.escape(str(e['provider_status']))}</td>
            <td>{html.escape(str(e['outcome']))}</td>
            <td>{html.escape(', '.join(e['reason_codes']))}</td></tr>"""
            for e in entries
        )
        history = f"""<div class="scroll"><table><thead><tr><th>Observed</th>
        <th>Provider status</th><th>Reconciliation</th><th>Reasons</th></tr></thead>
        <tbody>{rows}</tbody></table></div>"""

    trail_rows = "".join(
        f"""<tr><td>{html.escape(str(ev['occurred_at']))}</td>
        <td>{html.escape(str(ev['event_type']))}</td>
        <td><code>{html.escape(json.dumps(ev['payload'], sort_keys=True))}</code></td></tr>"""
        for ev in service.store.audit_trail(operation_key)
    )
    trail = f"""<h3 style="margin-top:18px">Audit trail</h3><div class="scroll"><table>
    <thead><tr><th>When</th><th>Event</th><th>Payload</th></tr></thead>
    <tbody>{trail_rows}</tbody></table></div>""" if trail_rows else ""

    return f"""<section class="card wide"><h2>PayPal execution &amp; independent reconciliation</h2>
    {banner}<p>Authoritative state: <strong>{html.escape(str(_STATE_LABELS.get(str(state), state)))}</strong> ·
    financial outcome: <strong>{html.escape(outcome_text)}</strong>
    {f'· provider status: <strong>{html.escape(str(provider_status))}</strong>' if provider_status else ''}</p>
    <p>Operation key (stable identity): <code>{html.escape(operation_key)}</code></p>
    <div class="timeline"><span>Prepared</span><span>Sending</span><span>Uncertain</span>
    <span>Acknowledged</span><span>Pending</span><span>Independently reconciled</span></div>
    {controls}{history}{trail}</section>"""


def store_outcome(store: MissionPayStore, operation_key: str) -> dict[str, object]:
    try:
        return store.latest_financial_outcome(operation_key)
    except PersistenceConflict:
        return {"outcome": None}
