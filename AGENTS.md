# Repository Guidelines

## Project Structure & Module Organization

This repository is a minimal, local-first RAG toolkit optimized for Chinese-language retrieval. Core code lives in `tinyrag/` (ingest, chunking, recall, fusion, reranking, and LangChain tool wrappers), and the CLI entrypoint is `script/rag_cli.py`. Data inputs and built indexes live under `data/` (`data/raw_data/` and `data/db/<db_name>/`), tests are in `test/`, evaluation helpers are in `eval/`, and local model files are expected in `models/`.

## Build, Test, and Development Commands

Use a Python environment where UTF-8 I/O works reliably (Windows users should prefer a modern terminal and UTF-8 code page). Typical commands are:

1. Install dependencies.
```bash
pip install -r requirements.txt
```

2. Build an index (example: the `law` dataset).
```bash
python -m script.rag_cli build --db-name law --path data\raw_data\law
```

3. Run a query against an existing index.
```bash
python -m script.rag_cli search --db-name law --query "什么是合同法中的不可抗力？"
```

4. Run unit tests (install `pytest` if needed).
```bash
python -m pytest -q
```

5. Optional interactive multi-turn demo.
```bash
python -m tinyrag.langchain_tools
```

## Coding Style & Naming Conventions

The codebase is Python-first and generally follows PEP 8 conventions: 4-space indentation, `snake_case` for modules and functions, and `CapWords` for classes. Keep changes small and consistent with surrounding code, and preserve UTF-8 text so Chinese strings remain readable in diffs and console output.

## Testing Guidelines

Tests live in `test/` and follow the `test_*.py` naming pattern with pytest-style `test_*` functions. Prefer tests that do not require network access and do not trigger large model downloads. When a test needs local models or indexes, document the prerequisite paths under `models/` or `data/db/`.

## Pull Request Guidelines

A good PR includes a clear description, reproduction steps, and any new data or model requirements. Do not commit secrets; keep `.env` local and document configuration via environment variable names such as `TINYRAG_DEVICE`, `TINYRAG_BM25_BACKEND`, and `TINYRAG_EMB_BATCH_SIZE`. When changing retrieval behavior, include a small example query and expected output characteristics.

## Agent-Specific Instructions

Automated agents and scripts should prioritize UTF-8 correctness and avoid producing mojibake or `??` in Chinese text. Avoid long-running operations by default; model downloads and full index builds should be explicit, opt-in actions.
