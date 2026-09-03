.PHONY: setup test backend frontend train

setup:
	uv sync --extra dev
	cd frontend && pnpm install --frozen-lockfile

test:
	uv run ruff format --check .
	uv run ruff check .
	uv run pytest
	cd frontend && pnpm test && pnpm lint && pnpm build

backend:
	uv run uvicorn market_gate.api:app --app-dir src --reload

frontend:
	cd frontend && pnpm dev

train:
	uv sync --extra training
	uv run python scripts/train_gate.py
