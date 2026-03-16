---
name: Bug report
about: Report something that is broken or not working as expected
labels: bug
assignees: fodorad
---

## Description

A clear and concise description of the bug.

## Steps to reproduce

```python
# Minimal reproducible example
from exordium.video.face import RetinaFaceDetector

detector = RetinaFaceDetector()
# ...
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback if applicable.

```
Traceback (most recent call last):
  ...
```

## Environment

- Exordium version: <!-- e.g. 1.5.0 — run `pip show exordium` -->
- PyTorch version: <!-- e.g. 2.10.0 — run `python -c "import torch; print(torch.__version__)"` -->
- Python version: <!-- e.g. 3.12.3 -->
- OS: <!-- e.g. Ubuntu 22.04 / macOS 15 / Windows 11 -->

## Additional context

Any other information that might be helpful (model name, input shape, etc.).
