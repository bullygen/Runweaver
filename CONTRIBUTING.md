# Contributing

Use Python 3.11–3.13 and an isolated environment:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,docs]'
```

Before a change:

1. identify the domain contract and backend boundary;
2. add a focused unit/contract test;
3. add failure injection for recovery changes;
4. update an ADR when dependency direction or source-of-truth ownership changes;
5. keep optional integrations out of root imports.

Run Ruff, strict mypy, pytest, strict documentation build, tutorials and package
build. Public objects require type annotations and docstrings. New serializers,
stores and planners should reuse the contract-test suites.

Releases update the changelog, schema migration notes and compatibility matrix.
