# Security boundaries

Market Gate Lab is a local educational app. It reads public market data and generates
synthetic quotes; it has no exchange-order endpoint.

Run the backend on `127.0.0.1`. Runtime configuration, local training, data capture and
optional Modal jobs mutate local state. The API checks browser origins for mutations,
but does not authenticate local clients. Do not expose the API through a public tunnel
or deploy it as a shared service without authentication and separate job/credential controls.

Modal credentials belong in the ignored root `.env`, or the local Training Lab credential
form. Saved keys use private file permissions and are not returned by the API. Avoid
including `.env` files, credential values or private dataset contents in issues and PRs.

Before sharing a fork, scan both tracked files and Git history. Removing a secret from
the latest revision does not remove older copies; revoke any exposed credential first.
