.PHONY: setup test backend frontend record train modal-plan experiment-check

RUN_DIR ?= artifacts/local-example
INPUT ?= fixtures/parity-replay-v2.jsonl

setup:
	uv sync --extra dev --extra training
	cd frontend && pnpm install --frozen-lockfile

test:
	uv run --extra dev --extra training ruff format --check .
	uv run --extra dev --extra training ruff check .
	uv run --extra dev --extra training pytest
	cd frontend && pnpm format:check && pnpm test && pnpm lint && pnpm build

backend:
	uv run uvicorn market_gate.api:app --app-dir src --reload

frontend:
	cd frontend && pnpm dev

record:
	uv run python scripts/record_binance.py --output $(RUN_DIR)/recording.jsonl

train:
	uv run --extra training python scripts/train_gate.py --input $(INPUT) --output $(RUN_DIR)/gate.npz --receipt $(RUN_DIR)/training.json

modal-plan:
	uv run --with modal==1.5.2 python scripts/train_on_modal.py

experiment-check:
	uv run python scripts/verify_phase2.py
