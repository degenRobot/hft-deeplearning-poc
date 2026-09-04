.PHONY: setup test backend frontend record train modal-plan

setup:
	uv sync --extra dev
	cd frontend && pnpm install --frozen-lockfile

test:
	uv run ruff format --check .
	uv run ruff check .
	uv run pytest
	cd frontend && pnpm format:check && pnpm test && pnpm lint && pnpm build

backend:
	uv run uvicorn market_gate.api:app --app-dir src --reload

frontend:
	cd frontend && pnpm dev

record:
	uv run --extra training python scripts/record_binance.py

train:
	uv run --extra training python scripts/train_gate.py

modal-plan:
	uv run --extra modal python scripts/train_on_modal.py
