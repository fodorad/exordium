"""Every weight the library downloads must be reachable — checked without downloading.

The test suite builds all model wrappers with random weights (``pretrained=False``), so
no checkpoint is ever fetched. That keeps CI fast, but it means nothing else would notice
if a weight file vanished from the Hub — the failure would only surface for a user at
runtime.

These tests close that gap the cheap way: a HEAD request per file. They assert the
mirror (``fodorad/exordium-weights``) still serves every checkpoint the wrappers ask for,
and that the upstream fallbacks are still there too.
"""

import unittest

from exordium.utils.ckpt import _HF_REPO_ID
from tests.fixtures import hf_file_exists, hf_repo_exists

MIRRORED_WEIGHTS = [
    "fabnet_weights.pth",
    "iris_weights.pth",
    "l2csnet_weights.pkl",
    "opengraphau-swint-1s_weights.pth",
    "opengraphau-swint-2s_weights.pth",
    "swin_tiny_patch4_window7_224.pth",
    "sixdrepnet_weights.pth",
    "marlin_vit_small_ytf.safetensors",
    "marlin_vit_base_ytf.safetensors",
    "marlin_vit_large_ytf.safetensors",
    "adaface_ir18.safetensors",
    "adaface_ir50.safetensors",
    "adaface_ir101.safetensors",
    "emotion2vec_plus_seed.pt",
]
"""Every file the wrappers fetch from the exordium mirror."""

UPSTREAM_FALLBACKS = [
    "osanseviero/6DRepNet_300W_LP_AFLW2000",
    "ControlNet/marlin_vit_small_ytf",
    "ControlNet/marlin_vit_base_ytf",
    "ControlNet/marlin_vit_large_ytf",
    "minchul/cvlface_adaface_ir18_vgg2",
    "minchul/cvlface_adaface_ir50_ms1mv2",
    "minchul/cvlface_adaface_ir101_webface4m",
    "emotion2vec/emotion2vec_plus_seed",
]
"""Original sources, used as fallbacks when the mirror cannot be reached."""


class TestMirrorWeightsAvailable(unittest.TestCase):
    """The mirror must serve every checkpoint the wrappers ask it for."""

    def test_mirror_repo_reachable(self):
        self.assertTrue(hf_repo_exists(_HF_REPO_ID), f"Mirror unreachable: {_HF_REPO_ID}")

    def test_every_mirrored_weight_is_present(self):
        missing = [f for f in MIRRORED_WEIGHTS if not hf_file_exists(_HF_REPO_ID, f)]
        self.assertEqual(missing, [], f"Missing from {_HF_REPO_ID}: {missing}")


class TestUpstreamFallbacksAvailable(unittest.TestCase):
    """The fallbacks must still exist, or the mirror is a single point of failure."""

    def test_upstream_repos_reachable(self):
        unreachable = [r for r in UPSTREAM_FALLBACKS if not hf_repo_exists(r)]
        self.assertEqual(unreachable, [], f"Upstream fallbacks unreachable: {unreachable}")


if __name__ == "__main__":
    unittest.main()
