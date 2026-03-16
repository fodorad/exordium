# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- GitHub Actions CI/CD workflows (CI, CD, Docs)
- GitHub issue templates (bug report, feature request)
- GitHub PR template and Dependabot configuration
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`
- `codecov.yml` for Codecov integration
- `Makefile` with `fix`, `lint`, `type-check`, `test`, `docs`, `check` targets
- Sphinx autodoc documentation deployed to GitHub Pages
- `py.typed` marker for PEP 561 type-checking support

### Changed
- License changed from BSD-3-Clause to MIT
- `exordium/utils/concurent.py` renamed to `exordium/utils/concurrent.py` (typo fix)
- Visualization utilities moved from `exordium/visualization/` to `exordium/utils/`

### Removed
- `exordium/video/rvm.py` and `exordium/video/emonet.py` (removed external dependencies)
- `exordium/video/bodypose.py`, `exordium/video/facedetector.py`, `exordium/video/openface.py`
- Legacy IO backends: `io_av.py`, `io_decord.py`, `io_moviepy.py`, `io_torch.py`
- `exordium/audio/wav2vec.py` (replaced by `wav2vec2.py`)

---

## [1.4.0] — 2024-01-01

### Added
- WavLM audio feature extractor (`exordium/audio/wavlm.py`)

---

## [1.3.0] — 2023-01-01

### Added
- Initial public release with audio, video, and text feature extractors

---

[Unreleased]: https://github.com/fodorad/exordium/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/fodorad/exordium/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/fodorad/exordium/releases/tag/v1.3.0
