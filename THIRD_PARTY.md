# Third-party components and integration references

MissionPay's original code is MIT-licensed under `LICENSE` (Null007Void | Null Foundry).
The components below remain under their own licences. Their terms are recorded here so
they are never absorbed into, or relabelled by, the MIT licence above.

## Runtime dependencies

Declared in `pyproject.toml`. MissionPay requires four directly; the rest are
transitive and resolved by pip. The table records the reviewed environment, not a complete lockfile; transitive versions may differ on a fresh install. Preserve dependency license files when redistributing them.

| Package | Version | Licence | Role |
| --- | --- | --- | --- |
| fastapi | 0.133.1 | MIT | HTTP routing and request parsing |
| starlette | 1.3.1 | BSD-3-Clause | ASGI toolkit beneath FastAPI |
| uvicorn | 0.41.0 | BSD-3-Clause | ASGI server |
| python-multipart | 0.0.32 | Apache-2.0 | Multipart form and file-upload parsing |
| argon2-cffi | 25.1.0 | MIT | Argon2 password hashing |
| argon2-cffi-bindings | 26.1.0 | MIT | Native Argon2 bindings (transitive) |
| cffi | 2.1.1 | MIT-0 | C bindings support (transitive) |
| pycparser | 3.12 | BSD-3-Clause | C parser used by cffi (transitive) |
| pydantic | 2.14.1 | MIT | Request/response models via FastAPI (transitive) |
| pydantic_core | 2.50.1 | MIT | Pydantic runtime (transitive) |
| annotated-types | 0.8.0 | MIT | Type constraints for pydantic (transitive) |
| annotated-doc | 0.0.5 | MIT | Documentation metadata helper (transitive) |
| typing_extensions | 4.16.0 | PSF-2.0 | Backported typing features (transitive) |
| typing-inspection | 0.4.4 | MIT | Typing inspection helper (transitive) |
| anyio | 4.15.1 | MIT | Async primitives used by Starlette (transitive) |
| idna | 3.20 | BSD-3-Clause | Internationalised domain names (transitive) |
| click | 8.5.0 | BSD-3-Clause | CLI parsing used by uvicorn (transitive) |
| h11 | 0.16.0 | MIT | HTTP/1.1 protocol (transitive) |
| colorama | 0.4.6 | BSD-3-Clause | Terminal colour on Windows (transitive) |

## Development and test tooling

| Package | Version | Licence | Role |
| --- | --- | --- | --- |
| pytest | 9.1.1 | MIT | Test runner (`dev` extra) |
| httpx | 0.28.1 | BSD-3-Clause | HTTP client used by Starlette's `TestClient` (`dev` extra) |
| httpcore | 1.0.9 | BSD-3-Clause | httpx transport (transitive, dev) |
| certifi | 2026.10.19 | MPL-2.0 | CA bundle for httpx (transitive, dev) |
| Jinja2 | 3.1.6 | BSD-3-Clause | Templating present in the dev environment |
| MarkupSafe | 3.0.4 | BSD-3-Clause | Jinja2 dependency |
| iniconfig / pluggy / Pygments | environment-resolved | MIT / MIT / BSD-2-Clause | pytest support |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | pytest support |
| setuptools | 84.2.1 | MIT | Build backend |

Python itself is distributed under the PSF Licence.

## No copied source

No third-party source code has been copied into this repository. PayPal is reached
through a narrow REST client written for this project using the Python standard
library's `urllib`. The official PayPal Python SDK is not vendored, imported, or
required.

## Official integration references

- PayPal Payouts overview: https://developer.paypal.com/payouts/use-payouts/overview
- Payout batch retrieval: https://developer.paypal.com/api/payments.payouts-batch/v1/payouts-get/
- Payout item retrieval: https://developer.paypal.com/api/payments.payouts-batch/v1/payouts-item-get/
- OAuth 2.0 client credentials: https://developer.paypal.com/api/rest/authentication/
- Sandbox accounts: https://developer.paypal.com/tools/sandbox/accounts/
- Payouts access help: https://www.paypal.com/us/cshelp/article/what-is-the-payouts-api-formerly-known-as-mass-pay-and-how-to-apply-for-it-help250
- Hackathon rules: https://paypalaihackathon.devpost.com/rules

## Trademarks

"PayPal" is a trademark of PayPal, Inc. MissionPay is an independent project, not
affiliated with, endorsed by, or sponsored by PayPal. References name the
integration target only.
