# Contributing to Exordium

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Quick start

```bash
git clone https://github.com/fodorad/exordium
cd exordium
pip install -e ".[all,dev,docs]"
pre-commit install   # optional: runs ruff automatically before every commit
```

---

## Development workflow

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — keep them focused and minimal.
3. **Write or update tests** in `tests/` to cover your changes.
4. **Run checks locally** before pushing:

   ```bash
   make fix    # auto-format and fix lint issues
   make check  # lint + tests + docs build (mirrors CI)
   ```

5. **Open a Pull Request** against `main` and fill in the template.

---

## Commit message convention

Exordium follows **Conventional Commits** to make the version history readable
and to signal the correct version bump when a release is made.

| Prefix | Meaning | Version bump |
|--------|---------|--------------|
| `fix:` | Bug fix, regression, hotfix | **Patch** (1.x.y) |
| `feat:` | New feature, new extractor | **Minor** (1.x.0) |
| `feat!:` or `BREAKING CHANGE:` | API change that breaks existing usage | **Major** (x.0.0) |
| `docs:` | Documentation only | No bump |
| `test:` | Tests only | No bump |
| `refactor:` | Code refactor with no behaviour change | No bump |
| `chore:` | Build, CI, dependency updates | No bump |

### Examples

```
fix: handle empty frame directory in detect_frame_dir
feat: add CLAP audio feature extractor
feat!: rename WavlmWrapper.audio_to_feature → predict_pipeline
docs: add quick-start example for gaze estimation
chore: bump torch to 2.10
```

---

## Release process

Releases are **tag-driven**. The version is derived from the git tag at build
time (`hatch-vcs`) — there is no version number stored in `pyproject.toml`.

When you are ready to release:

```bash
# 1. Update CHANGELOG.md — rename [Unreleased] → [1.5.0] and add the date
# 2. Commit the changelog
git add CHANGELOG.md
git commit -m "chore: release v1.5.0"

# 3. Tag and push
git tag v1.5.0
git push origin main --tags
```

GitHub Actions then builds the wheel, publishes to PyPI, deploys docs, and
creates a GitHub Release automatically.

**Version bump decision** — look at the commits since the last tag:

- Any `feat!` or `BREAKING CHANGE` → **major**
- Any `feat:` → **minor**
- Only `fix:`, `docs:`, `chore:` → **patch**

---

## Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) — run `make fix` to auto-apply.
- **Line length**: 100 characters.
- **Python version**: 3.12+.
- **Type hints**: encouraged but not required for internal helpers.

---

## Tests

```bash
make test              # run all tests with coverage
coverage html          # open coverage_html/index.html to browse
```

Test files live in `tests/` and mirror the `exordium/` package structure.
Model-dependent tests use `unittest.SkipTest` when weights are unavailable.

---

## Reporting bugs and requesting features

Please use the GitHub issue templates:

- **Bug report**: include a minimal reproducible example, Python/PyTorch version, and OS.
- **Feature request**: describe the problem you are trying to solve, not just the solution.

---

## License

By contributing you agree that your work will be released under the [MIT License](LICENSE).
