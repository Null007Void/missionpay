# Install and operate MissionPay

Python 3.11 is the independently tested runtime. Start in the extracted source directory. Dependency installation may access package indexes; running fixture/mock does not need an external service. Use a fresh virtual environment. Do not expose the bootstrap server beyond loopback.

## Bash (Linux/macOS)
```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
mkdir missionpay-local-run
export MISSIONPAY_PROVIDER_MODE=mock MISSIONPAY_AI_MODE=fixture
export MISSIONPAY_DEMO_BOOTSTRAP=1 MISSIONPAY_ALLOW_INSECURE_LOCALHOST=1
export MISSIONPAY_DB="$PWD/missionpay-local-run/missionpay-demo.sqlite3"
export MISSIONPAY_CREDENTIALS_FILE="$PWD/missionpay-local-run/absent.env"
.venv/bin/python -m missionpay.cli check
.venv/bin/python -m missionpay.cli serve --port 9876
```

## Windows interpreter discovery

The Windows examples use the Python launcher. Run `py -0p` to list installed interpreter paths and `py -3.11 --version` to confirm Python 3.11 is available. If `py` is absent or cannot find 3.11, locate your installed Python 3.11 executable and replace only the environment-creation line with its full path:

- PowerShell: `& 'C:\path\to\Python311\python.exe' -m venv .venv`
- CMD: `"C:\path\to\Python311\python.exe" -m venv .venv`

Replace the example path with the real installation path. Stop if environment creation fails. Every subsequent command uses the new environment's interpreter explicitly; activation is unnecessary. On Linux/macOS, confirm `python3 --version` before creating the environment.

## PowerShell (Windows)
```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install .
New-Item -ItemType Directory -Path missionpay-local-run -ErrorAction Stop
$env:MISSIONPAY_PROVIDER_MODE='mock'
$env:MISSIONPAY_AI_MODE='fixture'
$env:MISSIONPAY_DEMO_BOOTSTRAP='1'
$env:MISSIONPAY_ALLOW_INSECURE_LOCALHOST='1'
$env:MISSIONPAY_DB="$PWD/missionpay-local-run/missionpay-demo.sqlite3"
$env:MISSIONPAY_CREDENTIALS_FILE="$PWD/missionpay-local-run/absent.env"
& .\.venv\Scripts\python.exe -m missionpay.cli check
& .\.venv\Scripts\python.exe -m missionpay.cli serve --port 9876
```

## CMD (Windows Command Prompt)
```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install .
mkdir missionpay-local-run
set "MISSIONPAY_PROVIDER_MODE=mock"
set "MISSIONPAY_AI_MODE=fixture"
set "MISSIONPAY_DEMO_BOOTSTRAP=1"
set "MISSIONPAY_ALLOW_INSECURE_LOCALHOST=1"
set "MISSIONPAY_DB=%CD%\missionpay-local-run\missionpay-demo.sqlite3"
set "MISSIONPAY_CREDENTIALS_FILE=%CD%\missionpay-local-run\absent.env"
.venv\Scripts\python.exe -m missionpay.cli check
.venv\Scripts\python.exe -m missionpay.cli serve --port 9876
```

Stop if the data directory already exists; choose a new name and update both paths. Preserve prior database, WAL/SHM sidecars and evidence. Open http://127.0.0.1:9876/login and follow DEMO.md. Never put credentials in the source directory or use these public bootstrap accounts on a public server. Explicit fixture/mock modes ignore provider secrets; a conflicting inherited production PayPal setting may still cause safe startup refusal. Use a clean shell if necessary.

## Genuine PayPal sandbox

This is optional and makes real network requests to sandbox services. Do not run it as part of the no-key walkthrough. Use a separate new private data directory and sandbox-only credentials. Never reuse the withdrawn integrated judge-demo database. Replace the recipient placeholder with your authorized sandbox recipient. The example cap is 10000 cents ($100), allowing the bundled $78.99 claim; choose a lower cap if appropriate. A cap is not payment authorization: an authenticated reviewer must approve the exact claim and intent.

First set MISSIONPAY_CREDENTIALS_FILE to a new private path outside the source directory (replace the example paths below). Then create a private credential template with the **Create credential template** command in the shell-specific command table below, then fill it locally as instructed by that command. Do not commit or paste its contents. Set MISSIONPAY_CREDENTIALS_FILE to that private file. The following configuration examples do not send a request by themselves:

```bash
export MISSIONPAY_PROVIDER_MODE=paypal_sandbox MISSIONPAY_AI_MODE=fixture
export PAYPAL_ENVIRONMENT=sandbox PAYPAL_MODE=sandbox
export MISSIONPAY_SANDBOX_RECIPIENT='YOUR_AUTHORIZED_SANDBOX_EMAIL'
export MISSIONPAY_MAX_PAYOUT_CENTS=10000
export MISSIONPAY_DB='/absolute/private/new-run/missionpay.sqlite3'
export MISSIONPAY_CREDENTIALS_FILE='/absolute/private/credentials.env'
```
```powershell
$env:MISSIONPAY_PROVIDER_MODE='paypal_sandbox'; $env:MISSIONPAY_AI_MODE='fixture'
$env:PAYPAL_ENVIRONMENT='sandbox'; $env:PAYPAL_MODE='sandbox'
$env:MISSIONPAY_SANDBOX_RECIPIENT='YOUR_AUTHORIZED_SANDBOX_EMAIL'
$env:MISSIONPAY_MAX_PAYOUT_CENTS='10000'
$env:MISSIONPAY_DB='C:/private/new-run/missionpay.sqlite3'
$env:MISSIONPAY_CREDENTIALS_FILE='C:/private/credentials.env'
```
```bat
set "MISSIONPAY_PROVIDER_MODE=paypal_sandbox"
set "MISSIONPAY_AI_MODE=fixture"
set "PAYPAL_ENVIRONMENT=sandbox"
set "PAYPAL_MODE=sandbox"
set "MISSIONPAY_SANDBOX_RECIPIENT=YOUR_AUTHORIZED_SANDBOX_EMAIL"
set "MISSIONPAY_MAX_PAYOUT_CENTS=10000"
set "MISSIONPAY_DB=C:/private/new-run/missionpay.sqlite3"
set "MISSIONPAY_CREDENTIALS_FILE=C:/private/credentials.env"
```

Use **Check configuration** below to validate configuration without network. Use **Inspect preflight** to read the explicitly confirmed OAuth preflight options before choosing to run it. **Start server** starts the app. Local evaluation may retain the bootstrap and localhost-cookie flags above; those are not public deployment settings. A sandbox payout requires the complete review workflow and explicit operator action. Fixture AI can remain enabled while payments use sandbox; hosted AI is not required.

## Hosted AI and cost

Hosted AI is a separate opt-in configuration. Use **Inspect hosted qualification** below and the documented environment names in `src/missionpay/config.py` before creating a new authorization. Model, request identity, expected facts and persisted authorization matter. Do not reuse consumed authorizations. Receipt text, catalogue, instructions and schema leave the machine. A byte count is not a proven input-token bound. Dollar figures are estimates, not an enforced provider billing cap. Output limits and post-response usage checks do not guarantee a total spend cap or undo charges. See LIMITATIONS.md.

## Verify and build

Release assets have a separate SHA256SUMS.txt alongside them; do not assume an unpublished hash. Compare with `Get-FileHash -Algorithm SHA256` on PowerShell or `shasum -a 256` on macOS/Linux. To install a downloaded wheel, retain the full virtual-environment interpreter and `-m pip install` from your shell example, replacing only the final `.` argument with the quoted wheel path.

Source archives include MANIFEST.sha256. **Verify reproducible build** below verifies canonical source and repeated wheel builds using the pinned backend; it may need access to that backend unless cached. For development tests, use **Install test dependencies**, then **Run tests** below. **Verify HTTP journey** runs against an already running isolated fixture/mock server. Keep operational data outside the source tree.

## Commands using the installed environment

Run these from the source directory after creating `.venv`. These are separate operations, not a script to run wholesale. Credential initialization writes a private template; do not run it for the no-key walkthrough. Help commands do not contact providers.

| Operation | Bash | PowerShell | CMD |
| --- | --- | --- | --- |
| Create credential template | `.venv/bin/python -m missionpay.cli credentials init` | `& .\.venv\Scripts\python.exe -m missionpay.cli credentials init` | `.venv\Scripts\python.exe -m missionpay.cli credentials init` |
| Check configuration | `.venv/bin/python -m missionpay.cli check` | `& .\.venv\Scripts\python.exe -m missionpay.cli check` | `.venv\Scripts\python.exe -m missionpay.cli check` |
| Inspect preflight | `.venv/bin/python -m missionpay.cli preflight --help` | `& .\.venv\Scripts\python.exe -m missionpay.cli preflight --help` | `.venv\Scripts\python.exe -m missionpay.cli preflight --help` |
| Start server | `.venv/bin/python -m missionpay.cli serve --port 9876` | `& .\.venv\Scripts\python.exe -m missionpay.cli serve --port 9876` | `.venv\Scripts\python.exe -m missionpay.cli serve --port 9876` |
| Inspect hosted qualification | `.venv/bin/python -m missionpay.cli hosted-qualify --help` | `& .\.venv\Scripts\python.exe -m missionpay.cli hosted-qualify --help` | `.venv\Scripts\python.exe -m missionpay.cli hosted-qualify --help` |
| Verify reproducible build | `.venv/bin/python scripts/release/build_release.py --verify-only` | `& .\.venv\Scripts\python.exe scripts/release/build_release.py --verify-only` | `.venv\Scripts\python.exe scripts/release/build_release.py --verify-only` |
| Install test dependencies | `.venv/bin/python -m pip install -r requirements-dev.txt` | `& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` | `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` |
| Run tests | `.venv/bin/python -m pytest tests` | `& .\.venv\Scripts\python.exe -m pytest tests` | `.venv\Scripts\python.exe -m pytest tests` |
| Verify HTTP journey | `.venv/bin/python scripts/verify/clean_install_journey.py --base-url http://127.0.0.1:9876 --expect mock` | `& .\.venv\Scripts\python.exe scripts/verify/clean_install_journey.py --base-url http://127.0.0.1:9876 --expect mock` | `.venv\Scripts\python.exe scripts/verify/clean_install_journey.py --base-url http://127.0.0.1:9876 --expect mock` |
