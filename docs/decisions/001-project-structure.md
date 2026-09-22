# ADR 001 — Project structure and initial tooling

## Context

We need a Python project structure that supports:
- a growing application with multiple layers
- reliable dependency management
- automated linting and formatting
- a test suite that can grow to cover RAG evaluation
- reproducibility across environments

## Problem

Which project layout, package manager, and toolset should we use?

## Alternatives

| Option        | Pros                                        | Cons                                   |
|---------------|---------------------------------------------|----------------------------------------|
| `src/` layout | prevents accidental imports of source tree  | slightly more configuration needed     |
| flat layout   | simpler                                     | import resolution is unpredictable     |
| Poetry        | well-known, lockfile support                | slower than uv, no PEP 621 by default  |
| pip + venv    | zero extra tooling                          | no lockfile, no fast resolution        |
| **uv**        | fastest resolver, PEP 621, lockfile, modern | newer, smaller community than Poetry   |
| Ruff          | linter + formatter in one tool, very fast   | does not replace mypy for types        |
| black + flake8| widely known                                | two tools, slower, less integrated     |

## Decision

- **`src/` layout** — isolates the installable package; prevents test code from accidentally importing the wrong version.
- **uv** — modern, fast, PEP 621-compliant. Replaces pip, virtualenv, and Poetry.
- **`pyproject.toml`** — single configuration file for project metadata, dependencies, Ruff, pytest, and mypy.
- **Ruff** — single tool for linting and formatting. Configured to enforce type annotations (`ANN`), import sorting (`I`), and common correctness rules.
- **pydantic-settings** — typed configuration from environment variables. No scattered `os.getenv()` calls.

## Consequences

- All configuration lives in `pyproject.toml`.
- Adding a new dependency: `uv add <package>`.
- Running tests: `uv run pytest`.
- Running the formatter: `uv run ruff format .`.
- The `src/` layout means the package must be installed (`uv sync`) before tests can import it.
