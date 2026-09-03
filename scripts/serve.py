"""Start the replay-first read-only API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("market_gate.api:app", host="127.0.0.1", port=8000, reload=False)
