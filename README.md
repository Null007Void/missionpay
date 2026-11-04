# MissionPay

Restricted-fund reimbursement: evidence intake, structured interpretation, deterministic policy checks, authenticated human review, exact PayPal sandbox intent, and item-level reconciliation.

Original work by **Null007Void | Null Foundry**. [MIT license](LICENSE). Project: https://github.com/Null007Void/missionpay

Start with [INSTALL.md](INSTALL.md), which provides separate Bash, PowerShell and CMD instructions. The no-key walkthrough uses fixture AI and mock payments. Mock payments refuse provider execution; they do not simulate a successful genuine payout. The separate judge demo is not included in this real product.

[DEMO.md](DEMO.md) describes the local walkthrough. [ARCHITECTURE.md](ARCHITECTURE.md) explains trust boundaries. Read [LIMITATIONS.md](LIMITATIONS.md) before configuring external services and [THIRD_PARTY.md](THIRD_PARTY.md) for dependency attribution.

AI output is a proposal, not an authorization or proof of truth. Receipt-bound excerpt IDs resolve to exact unique text; this establishes provenance, not semantic correctness. Human review and payment controls remain necessary.

The default database is `missionpay-demo.sqlite3`. Set `MISSIONPAY_DB` to a path in a new empty dedicated data directory for each independent walkthrough. Preserve the previous database, WAL/SHM sidecars and evidence directory. There is no automatic reset or wildcard deletion step. Reusing an already reimbursed claim correctly produces `DUPLICATE_REIMBURSEMENT_BLOCKED`.

This history-free distribution contains source, synthetic fixtures and tests, not private recordings, live journals, credentials or historical release evidence. Offline checks cannot establish that a future model response or provider request will succeed.
