# Limitations and operating boundaries

This release retains the real application and removes the withdrawn integrated judge-demo paths. The separate simulated judge application is a different project. Do not reuse a database from the withdrawn integrated demo as genuine payment data. Preserve it separately and start with a fresh database or independently verified genuine data.

Fixture AI and mock payments need no provider credentials or calls. Mock payment execution refuses; it is not a successful sandbox transaction. Hosted AI and PayPal sandbox are explicit operator choices. Production PayPal endpoints are refused.

The v7 AI contract selects receipt-bound catalogue excerpt IDs. Resolution enforces exact unique occurrences, including overlapping matches. A model can still choose a unique but irrelevant or contradictory excerpt. Syntax, provenance, benchmark agreement and human judgment are different checks; none proves general semantic reliability.

A hosted request transmits receipt text, catalogue, instructions and schema. Do not submit sensitive material without authority. Provider pricing and tokenization are external. Payload bytes and character estimates are not a proven input-token bound or a hard dollar cap. Output-token limits constrain requested output; reported usage is checked after a response and cannot undo charges. Socket timeouts and rejection of late responses are not a guarantee of a total wall-clock deadline or cancellation at the provider. Qualification journals prevent reusing an authorization within their persisted state; they are not a provider-side spending account limit.

Payment controls bind reviewer approval, exact amount, recipient and intent; reconciliation must independently retrieve provider item status. Sandbox success does not establish production readiness. AI does not authorize payments.

The documented bootstrap accounts and insecure localhost cookies are for isolated loopback evaluation only. Do not expose this setup publicly. Public hosting requires a separate deployment review covering TLS, account provisioning, secrets, backups, monitoring, process topology and resource limits. Multi-instance deployment is not qualified here.

Private historical recordings and live evidence are deliberately absent. This distribution makes no claim that every prior model attempt succeeded, that no historical provider calls occurred, or that offline tests prove live reliability. The synthetic tests and installed-package verification script are included so operators can repeat offline checks; private validation reports are not part of this distribution.
