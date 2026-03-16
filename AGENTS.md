# Repository Guidelines

## Project Structure & Module Organization
This repo is anchored by `pyproject.toml`; add runtime code under `src/qutip_sampler/` so module resolution stays consistent with editable installs, and keep domain-specific data or fixtures in `tests/data/`. A typical layout is:
```
src/qutip_sampler/samplers.py
src/qutip_sampler/__init__.py
tests/test_samplers.py
assets/notebooks/
```
Treat `assets/` as read-only inputs; generated artifacts belong under `outputs/` (git-ignored). Use package-level `__all__` exports to control the public API.

## Build, Test, and Development Commands
Create a virtualenv (`python -m venv .venv && source .venv/bin/activate`) and install the project in editable mode: `python -m pip install -e .[dev]`. Run `python -m pytest` for the full test suite; `python -m pytest tests/test_samplers.py::test_draw_complex_state` runs a focused test when debugging. Use `python -m build` to produce source and wheel artifacts before publishing, and `python -m pip install -e . --config-settings editable_mode=strict` to verify packaging metadata.

## Coding Style & Naming Conventions
Target Python 3.13 and follow PEP 8 with 4-space indentation. Public modules and functions use `snake_case`, classes use `PascalCase`, and test helpers may use leading underscores when private. Prefer dataclasses for structured sampler configuration and type-annotate all public callables. Keep functions <50 lines by extracting helper utilities into `src/qutip_sampler/utils/`. When formatting JSON-serializable payloads, ensure deterministic key ordering to simplify test snapshots.

## Testing Guidelines
Pytest is the canonical framework; place files beside the code they exercise under `tests/` and name them `test_<module>.py`. Each new sampler must include at least one statistical sanity test and one deterministic edge-case test. Use parametrize blocks for multi-backend checks and `pytest.mark.slow` for stochastic trials exceeding ~1s. Before opening a PR, run `python -m pytest --maxfail=1 --ff` to surface flakes quickly and ensure any fixtures seeded via `numpy.random.default_rng` are local to the test to avoid global state bleed.

## Commit & Pull Request Guidelines
Even without history, follow Conventional Commit-style summaries (`feat: add lindblad sampler`) limited to 72 characters and include context in the body when API behavior changes. Reference issues with `Fixes #12` or `Refs #12`. Pull requests must describe the motivation, summarize functional changes, list test evidence (commands + results), and attach plots or tables when sampler accuracy is affected. Request review once CI is green and label work-in-progress branches clearly to avoid premature merges.

## Security & Configuration Tips
Never commit tokens, notebooks with embedded credentials, or experiment data sampled from restricted datasets; store secrets in environment variables loaded via `dotenv`. Check `git status --ignored` before pushing to confirm temporary calibration files remain untracked. When adding optional dependencies (e.g., GPU solvers), guard imports with runtime checks and document required environment variables inside `README.md`.
