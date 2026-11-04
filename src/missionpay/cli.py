"""Operator command line: serve, inspect configuration, back up, export evidence.

The backup and export commands exist because a reimbursement system that cannot
hand over its own records is not usable by a nonprofit. Both are read-only against
the live database: the backup uses SQLite's online backup API and the export only
reads rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from .config import ConfigurationError, from_environment
from .evidence import EvidenceStore, default_evidence_root


def _config():
    try:
        return from_environment()
    except ConfigurationError as exc:
        print(f"CONFIGURATION_REFUSED: {exc}", file=sys.stderr)
        print(
            "MissionPay will not start with this configuration. "
            "See INSTALL.md for the required variables.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _serve(args) -> int:
    import uvicorn

    config = _config()
    print(f"provider_mode={config.provider_mode.value}")
    print(f"mode_label={config.mode_label}")
    print(f"database={config.database_path}")
    uvicorn.run(
        "missionpay.entrypoint:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


def _check(_args) -> int:
    config = _config()
    report = config.describe()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not config.is_genuine:
        print("GENUINE_EXECUTION: unavailable (mock mode)")
    else:
        print("GENUINE_EXECUTION: configured (credentials present, sandbox origin pinned)")
    print("NO PROVIDER CALL WAS MADE BY THIS COMMAND")
    return 0


def _backup(args) -> int:
    config = _config()
    source = Path(config.database_path)
    if not source.exists():
        print(f"DATABASE_NOT_FOUND: {source}", file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(args.destination)
    if destination.is_dir() or args.destination.endswith(("/", "\\")):
        destination = destination / f"missionpay-backup-{stamp}.sqlite3"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Online backup API: consistent even with a live writer, unlike a file copy of a WAL db.
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)

    evidence_source = default_evidence_root(source)
    if evidence_source.is_dir():
        evidence_target = destination.parent / f"{destination.stem}-evidence"
        if evidence_target.exists():
            shutil.rmtree(evidence_target)
        shutil.copytree(evidence_source, evidence_target)
        print(f"evidence_copied_to={evidence_target}")

    print(f"database_backed_up_to={destination}")
    print("BACKUP_COMPLETE: stop the server before moving the original database")
    return 0


def _export(args) -> int:
    config = _config()
    source = Path(config.database_path)
    if not source.exists():
        print(f"DATABASE_NOT_FOUND: {source}", file=sys.stderr)
        return 1
    with sqlite3.connect(source) as conn:
        conn.row_factory = sqlite3.Row

        def rows(sql: str, params: tuple = ()) -> list[dict]:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "provider_mode": config.provider_mode.value,
            "claims": rows("SELECT * FROM claims ORDER BY claim_id, revision"),
            "interpretations": rows(
                "SELECT * FROM interpretations ORDER BY claim_id, claim_revision, attempt"
            ),
            "authorizations": rows("SELECT * FROM authorizations ORDER BY authorization_id"),
            "payment_intents": rows("SELECT * FROM payment_intents ORDER BY operation_key"),
            "payment_operations": rows("SELECT * FROM payment_operations ORDER BY operation_key"),
            "provider_observations": rows(
                "SELECT * FROM provider_observations ORDER BY observed_at"
            ),
            "reconciliations": rows("SELECT * FROM reconciliations ORDER BY reconciled_at"),
            "audit_events": rows("SELECT * FROM audit_events ORDER BY sequence"),
        }
    if args.operation_key:
        key = args.operation_key
        payload = {
            name: [item for item in value if key in json.dumps(item)]
            if isinstance(value, list)
            else value
            for name, value in payload.items()
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"export_written_to={args.output}")
    else:
        print(rendered)
    return 0


def _evidence_paths(_args) -> int:
    config = _config()
    store = EvidenceStore(default_evidence_root(config.database_path))
    if not store.root.is_dir():
        print("NO_EVIDENCE_DIRECTORY")
        return 0
    for item in sorted(store.root.iterdir()):
        print(f"{item.stat().st_size}\t{item.name}")
    return 0


def _credentials_init(args) -> int:
    from .credentials import CredentialError, write_template

    try:
        path = write_template(Path(args.path) if args.path else None)
    except CredentialError as exc:
        print(f"CREDENTIALS_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"template_written_to={path}")
    print("Fill in PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET in that file.")
    print("The file is outside the repository and is never read by mock mode.")
    return 0


def _credentials_status(args) -> int:
    from .credentials import status

    report = status(Path(args.path) if args.path else None).describe()
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PRESENCE_ONLY: no credential value was read into this output")
    return 0


def _preflight(args) -> int:
    """Acquire one OAuth token and report granted scope. Never creates a payout."""
    from .credentials import default_credentials_path
    from .oauth import PayPalClientCredentials, PayPalOAuthError

    config = _config()
    if not config.is_genuine:
        print("PREFLIGHT_REFUSED: provider mode is mock", file=sys.stderr)
        print("Set MISSIONPAY_PROVIDER_MODE=paypal_sandbox with credentials to preflight.")
        return 2
    if not args.confirm:
        print("PREFLIGHT_REFUSED: this contacts PayPal", file=sys.stderr)
        print("Re-run with --confirm to perform exactly one OAuth token request.")
        print("No payout is created by this command. No money moves.")
        return 2

    credentials = PayPalClientCredentials(
        client_id=config.paypal_client_id or "",
        client_secret=config.paypal_client_secret or "",
        base_url=config.paypal_base_url,
    )
    print(f"base_url={config.paypal_base_url}")
    print(f"credentials_file={default_credentials_path()}")
    try:
        token = credentials.access_token()
    except PayPalOAuthError as exc:
        print(f"OAUTH_FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"oauth_token_acquired=yes length_class={'set' if token else 'empty'}")
    print("PAYOUT_CREATED: no")
    print("This proves token acquisition only. It does not prove a payout works.")
    return 0


def _hosted_qualify(args) -> int:
    """One authorised hosted request, with the attempt recorded before it is sent.

    The expectation is loaded and validated first, for a dry run as well as a real send.
    A missing or malformed file must cost nothing, so it is refused before the plan is
    built and long before any socket exists.
    """
    from .hosted_qualify import (
        QualificationRefused,
        build_plan,
        load_expectation,
        qualify,
    )
    from .store import MissionPayStore

    config = _config()
    store = MissionPayStore(config.database_path)
    # Idempotent: creates the attempt journal if this database has never been served.
    store.initialize()

    try:
        expectation = load_expectation(args.expect)
    except QualificationRefused as exc:
        print(f"REFUSED: {exc.code}", file=sys.stderr)
        print(
            "The expectation is loaded and validated before anything is sent.\n"
            "Nothing was recorded and no request was sent.",
            file=sys.stderr,
        )
        return 2

    try:
        plan, _text, _evidence = build_plan(
            store,
            config,
            authorization_id=args.authorization,
            claim_id=args.claim,
            revision=args.revision,
            expectation=expectation,
        )
    except QualificationRefused as exc:
        print(f"REFUSED: {exc.code}", file=sys.stderr)
        print("Nothing was recorded and no request was sent.", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        print("DRY RUN: nothing recorded, nothing sent, no provider was contacted")
        return 0

    if not args.confirm:
        print("REFUSED: HOSTED_QUALIFY_NOT_CONFIRMED", file=sys.stderr)
        print(
            "This command can spend money and send a donor's receipt to a provider.\n"
            "Re-run with --confirm only under a written authorisation reference.",
            file=sys.stderr,
        )
        return 2

    try:
        outcome = qualify(
            store,
            config,
            authorization_id=args.authorization,
            claim_id=args.claim,
            revision=args.revision,
            expectation=expectation,
        )
    except QualificationRefused as exc:
        print(f"REFUSED: {exc.code}", file=sys.stderr)
        print("Nothing was recorded and no request was sent.", file=sys.stderr)
        return 2

    status = outcome["status"]
    print(f"PROVIDER_STATUS {status}")
    print(f"provider_completed {outcome.get('provider_completed', False)}")
    print(f"recorded_before_send {'yes' if outcome.get('sent') else 'no request left this process'}")
    if outcome.get("usage") is not None:
        print(f"usage {json.dumps(outcome['usage'], sort_keys=True)}")
        print("usage_source provider_reported (not an estimate; retained even on refusal)")
    estimate = plan.estimate or {}
    if estimate:
        print(f"token_bound {json.dumps(estimate, sort_keys=True)}")
        print(f"bound_basis {estimate.get('bound_basis')}")
        print(f"estimate_basis {estimate.get('estimate_basis')}")
    print(f"timeout_seconds {plan.timeout_seconds}")
    print(f"late_response_seconds {plan.late_response_seconds}")
    if outcome.get("elapsed_seconds") is not None:
        print(f"elapsed_seconds {outcome['elapsed_seconds']:.3f}")
    if outcome.get("code"):
        print(f"code {outcome['code']}")
    if outcome.get("diagnostic"):
        # Names, categories and offsets only; the store would refuse anything else.
        print(f"diagnostic {json.dumps(outcome['diagnostic'], sort_keys=True)}")

    if status == "ALREADY_ATTEMPTED":
        print(outcome["detail"], file=sys.stderr)
        return 4

    verdict = outcome.get("verdict")
    if verdict is not None:
        print(json.dumps(verdict, indent=2, sort_keys=True))
    if outcome.get("interpretation") is not None:
        print(json.dumps(outcome["interpretation"], indent=2, sort_keys=True))

    if args.json:
        Path(args.json).write_text(
            json.dumps(outcome, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Qualification is the verdict, not the completion. A provider that answered and a
    # claim that qualified are two different facts and two different exit codes.
    if status == "QUALIFIED":
        print("QUALIFICATION PASS — the answer matched the frozen expectation")
        return 0
    if status == "VERDICT_FAILED":
        print(
            "QUALIFICATION FAIL — the provider completed and the answer did not match "
            "the frozen expectation; the expectation was not adjusted",
            file=sys.stderr,
        )
        return 3
    print(f"NO VERDICT — the run ended as {status}", file=sys.stderr)
    return 1


def _evidence_text_for(store, plan) -> str:
    from .evidence import EvidenceStore, default_evidence_root

    text = EvidenceStore(default_evidence_root(store.path)).read_text(
        plan["evidence_sha256"]
    )
    return text or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionpay", description="MissionPay restricted-fund reimbursement"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9876)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    check = sub.add_parser("check", help="print the resolved configuration (no network)")
    check.set_defaults(func=_check)

    backup = sub.add_parser("backup", help="online-backup the database and evidence")
    backup.add_argument("destination")
    backup.set_defaults(func=_backup)

    export = sub.add_parser("export", help="export payments and audit trail as JSON")
    export.add_argument("--operation-key", default=None)
    export.add_argument("--output", default=None)
    export.set_defaults(func=_export)

    evidence = sub.add_parser("evidence", help="list stored evidence blobs")
    evidence.set_defaults(func=_evidence_paths)

    credentials = sub.add_parser("credentials", help="prepare or inspect the private credential file")
    credentials_sub = credentials.add_subparsers(dest="credentials_command", required=True)
    c_init = credentials_sub.add_parser("init", help="write an empty credential template")
    c_init.add_argument("--path", default=None)
    c_init.set_defaults(func=_credentials_init)
    c_status = credentials_sub.add_parser("status", help="report presence only, never values")
    c_status.add_argument("--path", default=None)
    c_status.set_defaults(func=_credentials_status)

    preflight = sub.add_parser(
        "preflight",
        help="acquire one sandbox OAuth token (no payout, no money moves)",
    )
    preflight.add_argument("--confirm", action="store_true")
    preflight.set_defaults(func=_preflight)

    hosted = sub.add_parser(
        "hosted-qualify",
        help="one authorised hosted request, recorded before it is sent",
    )
    hosted.add_argument("--authorization", required=True)
    hosted.add_argument("--claim", required=True)
    hosted.add_argument("--revision", type=int, default=None)
    hosted.add_argument(
        "--expect",
        required=True,
        help="directory (or file) holding the frozen expected.json; validated before any send",
    )
    hosted.add_argument("--json", default=None)
    hosted.add_argument("--dry-run", action="store_true")
    hosted.add_argument("--confirm", action="store_true")
    hosted.set_defaults(func=_hosted_qualify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
