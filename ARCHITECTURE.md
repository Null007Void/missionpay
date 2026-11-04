# Architecture

MissionPay is a Python application with a FastAPI web interface and SQLite persistence. `src/missionpay` contains the shipped runtime. `tests` contains synthetic regression cases; `scripts/verify` contains the installed-package HTTP journey; `scripts/release` builds and verifies public source and wheel artifacts.

## Evidence and interpretation

Uploaded evidence is stored separately from database records. Fixture, local-model and hosted modes are explicit configuration choices. The current hosted contract asks for canonical facts referring to receipt-bound excerpt IDs. The application resolves those IDs to exact spans and checks uniqueness, including overlaps. Catalogue identity and receipt identity are bound to the prompt. Historical interpretations retain their contract labels rather than silently gaining current guarantees.

Provenance shows which text a model selected. It does not prove that text supports the proposed fact. Authenticated human review remains necessary. Deterministic policy evaluates eligibility independently of AI assertions, while approval binds the accepted interpretation identity and payment intent.

## Payments

The mock adapter refuses provider operations. The genuine adapter is restricted to the PayPal sandbox origin and configured recipient and maximum payout. Approval, intent hashing, idempotency and independent item-level reconciliation govern the payment lifecycle. A submitted request alone is not proof of success. Production PayPal is refused.

## State and operation

SQLite state, its sidecars, stored evidence and authorization journals are private operational data. Backups and exports can contain sensitive records and belong outside a public repository. No destructive demo reset is shipped. Start independent evaluations in new empty dedicated directories. Do not relabel old simulated records as genuine.

## Packaging

Public exports omit Git history, credentials, runtime stores, recordings and private replay tooling. LF manifests verify exact payload inventory and bytes. The wheel builder validates packaged source against Git or the extracted manifest and uses a pinned backend and timestamp. Reproducibility is verified on the tested platform; cross-platform container identity is not promised.
