# Local no-key walkthrough

Follow the fixture/mock instructions in INSTALL.md and open http://127.0.0.1:9876/login.

1. Sign in as `volunteer@missionpay.local` with `volunteer-demo-passphrase`.
2. Use the bundled synthetic evidence and submit a reimbursement for review.
3. Sign out; sign in as `reviewer@missionpay.local` with `reviewer-demo-passphrase`.
4. Inspect the evidence, interpretation, policy outcome and exact payment intent before authorizing it.
5. Attempt payment. Mock mode must refuse provider execution; it must not claim a genuine payout.

These accounts are public evaluation fixtures, not production credentials. This walkthrough is not a recording of a live provider run. To repeat independently, use a new empty dedicated data directory and preserve the previous database, sidecars and evidence. The installed-package verifier exercises the HTTP workflow with synthetic inputs; it does not call a payment provider.
