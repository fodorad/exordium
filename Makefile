.PHONY: install dev install-docs fix lint type-check test docs docs-serve docs-deploy check clean help

help:
	@echo "Dev (modify files):  fix"
	@echo "Checks (read-only):  lint | type-check | test | docs | check"
	@echo "Setup:               install | dev | install-docs"
	@echo "Docs:                docs-serve | docs-deploy"
	@echo "Cleanup:             clean"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv pip install "exordium[all]"

dev:
	uv pip install -e ".[all,dev]"

install-docs:
	uv pip install -e ".[docs]"

# ── Dev helpers (modify files) ─────────────────────────────────────────────────

fix:
	ruff format .
	ruff check --fix .

# ── Checks (read-only — mirrors GitHub CI) ─────────────────────────────────────

lint:
	ruff check .
	ruff format --check .

type-check:
	ty check exordium

test:
	coverage run -m unittest discover -s tests
	coverage report
	coverage html
	coverage xml -o coverage.xml

docs:
	sphinx-build -b html docs/ site/

check: lint type-check test docs

# ── Docs ───────────────────────────────────────────────────────────────────────

docs-serve:
	sphinx-autobuild docs/ site/

docs-deploy:
	@echo "Docs are deployed automatically via GitHub Actions on push to main."

# ── Misc ───────────────────────────────────────────────────────────────────────

clean:
	rm -rf .venv coverage_html dist/ .pytest_cache/ site/ tmp/
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
