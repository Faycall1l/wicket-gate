.PHONY: help install test lint bench demo dataset eval

help:
	@echo "install  uv sync (project + dev)"
	@echo "test     offline unit tests (no cluster, no LLM key)"
	@echo "lint     ruff"
	@echo "bench    score both checks on the cluster-labeled benchmark"
	@echo "demo     scripted walkthrough"
	@echo "dataset  regenerate the seeded decision dataset"
	@echo "eval     offline cascade eval (teacher=oracle)"

install:
	uv sync

test:
	uv run python -m pytest tests/ -q

lint:
	uvx ruff check src tests benchmark evals

bench:
	uv run python benchmark/evaluate.py

demo:
	uv run python docs/assets/demo.py

dataset:
	uv run python evals/gen_decision_dataset.py

eval:
	uv run python evals/cascade_eval.py --teacher oracle --alpha 0.10
