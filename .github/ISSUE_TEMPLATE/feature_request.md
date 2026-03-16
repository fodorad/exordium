---
name: Feature request
about: Suggest a new feature or improvement
labels: enhancement
assignees: fodorad
---

## Problem statement

A clear description of the problem or limitation you are facing.
Example: "I need to extract features from a custom audio format, but currently..."

## Proposed solution

Describe the feature or change you would like, including any API changes.

```python
# Example of what the new API might look like
from exordium.audio.wavlm import WavlmWrapper

model = WavlmWrapper(device_id=-1, new_option=True)  # <-- proposed argument
```

## Alternatives considered

Any alternative approaches or workarounds you have already tried.

## Additional context

Links to papers, related issues, or other projects that do something similar.
