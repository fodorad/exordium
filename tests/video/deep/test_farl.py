"""Tests for FarlWrapper visual feature extractor."""

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from exordium.utils.ckpt import _HF_REPO_ID
from exordium.video.core.detection import DetectionFromTorchTensor, Track
from exordium.video.deep.farl import (
    _DEFAULT_MODEL,
    _FEATURE_DIM,
    _HIDDEN_SIZE,
    _MODELS,
    _NUM_LAYERS,
    FarlWrapper,
    _convert_openai_to_hf,
    _vision_config,
)
from tests.fixtures import (
    IMAGE_EMMA,
    IMAGE_FACE,
    PRETRAINED,
    VIDEO_MULTISPEAKER_SHORT,
    ModelTestCase,
    hf_file_exists,
)


def _fake_farl_state_dict() -> dict[str, torch.Tensor]:
    """Build an OpenAI-CLIP-style FaRL state_dict of the right shapes.

    Values are random; only the *key layout* and *shapes* mirror the real
    checkpoint, which is what the converter is responsible for handling. The MIM
    pre-training keys are included so the test proves they are dropped rather
    than leaking into the encoder.
    """
    d = _HIDDEN_SIZE
    sd: dict[str, torch.Tensor] = {
        "visual.class_embedding": torch.randn(d),
        "visual.positional_embedding": torch.randn(197, d),
        "visual.proj": torch.randn(d, _FEATURE_DIM),
        "visual.conv1.weight": torch.randn(d, 3, 16, 16),
        "visual.ln_pre.weight": torch.randn(d),
        "visual.ln_pre.bias": torch.randn(d),
        "visual.ln_post.weight": torch.randn(d),
        "visual.ln_post.bias": torch.randn(d),
        # Pre-training-only heads: must not survive conversion.
        "visual.mask_token": torch.randn(d),
        "visual.ln_lm.weight": torch.randn(d),
        "visual.lm_head.weight": torch.randn(8192, d),
        # Text tower: also irrelevant to the vision encoder.
        "text_projection": torch.randn(d, _FEATURE_DIM),
        "logit_scale": torch.tensor(4.6),
    }
    for i in range(_NUM_LAYERS):
        p = f"visual.transformer.resblocks.{i}."
        sd |= {
            f"{p}attn.in_proj_weight": torch.randn(3 * d, d),
            f"{p}attn.in_proj_bias": torch.randn(3 * d),
            f"{p}attn.out_proj.weight": torch.randn(d, d),
            f"{p}attn.out_proj.bias": torch.randn(d),
            f"{p}ln_1.weight": torch.randn(d),
            f"{p}ln_1.bias": torch.randn(d),
            f"{p}ln_2.weight": torch.randn(d),
            f"{p}ln_2.bias": torch.randn(d),
            f"{p}mlp.c_fc.weight": torch.randn(4 * d, d),
            f"{p}mlp.c_fc.bias": torch.randn(4 * d),
            f"{p}mlp.c_proj.weight": torch.randn(d, 4 * d),
            f"{p}mlp.c_proj.bias": torch.randn(d),
        }
    return sd


def _track(num_frames: int = 6) -> Track:
    """A short synthetic face track with per-frame boxes of drifting size."""
    frame = torch.randint(0, 255, (3, 400, 400), dtype=torch.uint8)
    landmarks = torch.zeros(5, 2, dtype=torch.long)
    track = Track(track_id=0)
    for fid in range(num_frames):
        side = 100 + (fid % 5) * 10
        x = y = 200 - side // 2
        track.add(
            DetectionFromTorchTensor(
                frame_id=fid,
                source=frame,
                score=0.99,
                bb_xywh=torch.tensor([x, y, side, side], dtype=torch.long),
                landmarks=landmarks,
            )
        )
    return track


class TestFarlConversion(unittest.TestCase):
    """The OpenAI -> HuggingFace state_dict conversion is the load-bearing logic."""

    def test_converted_dict_loads_with_no_missing_or_unexpected_keys(self):
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection(_vision_config())
        missing, unexpected = model.load_state_dict(
            _convert_openai_to_hf(_fake_farl_state_dict()), strict=False
        )
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])

    def test_pretraining_heads_are_dropped(self):
        converted = _convert_openai_to_hf(_fake_farl_state_dict())
        for key in converted:
            self.assertNotIn("lm_head", key)
            self.assertNotIn("mask_token", key)
            self.assertNotIn("text_projection", key)

    def test_fused_qkv_is_split_in_q_k_v_order(self):
        sd = _fake_farl_state_dict()
        converted = _convert_openai_to_hf(sd)
        fused = sd["visual.transformer.resblocks.0.attn.in_proj_weight"]
        d = _HIDDEN_SIZE
        torch.testing.assert_close(
            converted["vision_model.encoder.layers.0.self_attn.q_proj.weight"], fused[:d]
        )
        torch.testing.assert_close(
            converted["vision_model.encoder.layers.0.self_attn.k_proj.weight"], fused[d : 2 * d]
        )
        torch.testing.assert_close(
            converted["vision_model.encoder.layers.0.self_attn.v_proj.weight"], fused[2 * d :]
        )

    def test_visual_projection_is_transposed(self):
        sd = _fake_farl_state_dict()
        converted = _convert_openai_to_hf(sd)
        # OpenAI stores (D, P) applied as x @ proj; nn.Linear needs (P, D).
        self.assertEqual(
            tuple(converted["visual_projection.weight"].shape), (_FEATURE_DIM, _HIDDEN_SIZE)
        )
        torch.testing.assert_close(
            converted["visual_projection.weight"], sd["visual.proj"].t().contiguous()
        )

    def test_all_layers_are_converted(self):
        converted = _convert_openai_to_hf(_fake_farl_state_dict())
        for i in range(_NUM_LAYERS):
            self.assertIn(f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight", converted)
        self.assertNotIn(
            f"vision_model.encoder.layers.{_NUM_LAYERS}.self_attn.q_proj.weight", converted
        )


class TestFarlWrapperInit(unittest.TestCase):
    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            FarlWrapper(model_name="nonexistent_model")

    def test_module_constants_consistent(self):
        self.assertIn(_DEFAULT_MODEL, _MODELS)
        for cfg in _MODELS.values():
            self.assertIn("filename", cfg)
            self.assertIn("epochs", cfg)


class TestFarlWrapper(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FarlWrapper(model_name=_DEFAULT_MODEL, device_id=None, pretrained=PRETRAINED)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, _FEATURE_DIM)

    def test_from_image_path(self):
        self.assertEqual(self.model(IMAGE_FACE).shape, (1, _FEATURE_DIM))

    def test_from_numpy_hwc(self):
        out = self.model(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        self.assertEqual(out.shape, (1, _FEATURE_DIM))

    def test_from_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, _FEATURE_DIM))

    def test_preprocess_output_shape(self):
        out = self.model.preprocess(torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8))
        self.assertEqual(out.shape, (3, 3, 224, 224))

    def test_output_is_l2_normalised(self):
        norms = self.model(torch.randint(0, 255, (3, 3, 224, 224), dtype=torch.uint8)).norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_feature_dim_consistent(self):
        self.assertEqual(self.model(IMAGE_FACE).shape[1], self.model(IMAGE_EMMA).shape[1])

    def test_track_to_feature(self):
        track = _track(num_frames=6)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.track_to_feature(
                track, batch_size=4, output_path=pathlib.Path(tmp_dir) / "farl.st"
            )
        self.assertEqual(result["features"].shape, (6, _FEATURE_DIM))
        self.assertEqual(result["frame_ids"].tolist(), list(range(6)))

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=pathlib.Path(tmp_dir) / "farl.st",
            )
        self.assertEqual(result["features"].shape[1], _FEATURE_DIM)


class TestFarlTrackDensification(ModelTestCase):
    """densify=True must put a real wrapper's sparse output on a gap-free grid."""

    @classmethod
    def setUpClass(cls):
        cls.model = FarlWrapper(model_name=_DEFAULT_MODEL, device_id=None, pretrained=PRETRAINED)

    @staticmethod
    def _gapped_track(frame_ids: list[int]) -> Track:
        """A track detected only on `frame_ids` — i.e. with holes, like a real one."""
        frame = torch.randint(0, 255, (3, 400, 400), dtype=torch.uint8)
        track = Track(track_id=0)
        for fid in frame_ids:
            track.add(
                DetectionFromTorchTensor(
                    frame_id=fid,
                    source=frame,
                    score=0.99,
                    bb_xywh=torch.tensor([150, 150, 100, 100], dtype=torch.long),
                    landmarks=torch.zeros(5, 2, dtype=torch.long),
                )
            )
        return track

    def test_sparse_is_still_the_default(self):
        out = self.model.track_to_feature(self._gapped_track([0, 2, 7]))
        self.assertEqual(out["features"].shape, (3, _FEATURE_DIM))
        self.assertNotIn("mask", out)

    def test_densify_fills_gaps_and_flags_them(self):
        out = self.model.track_to_feature(
            self._gapped_track([0, 2, 7]), densify=True, end_frame_id=8
        )
        self.assertEqual(out["features"].shape, (8, _FEATURE_DIM))
        self.assertEqual(
            out["mask"].tolist(), [True, False, True, False, False, False, False, True]
        )
        # Undetected frames hold the fill value, not a stale or garbage embedding.
        self.assertTrue(torch.equal(out["features"][1], torch.zeros(_FEATURE_DIM)))

    def test_densified_rows_match_the_sparse_features(self):
        track = self._gapped_track([0, 2])
        sparse = self.model.track_to_feature(track)
        dense = self.model.track_to_feature(track, densify=True, end_frame_id=4)
        torch.testing.assert_close(dense["features"][0], sparse["features"][0])
        torch.testing.assert_close(dense["features"][2], sparse["features"][1])

    def test_window_densification_does_not_materialise_the_whole_video(self):
        out = self.model.track_to_feature(
            self._gapped_track([0, 120, 200]),
            densify=True,
            start_frame_id=100,
            end_frame_id=150,
        )
        self.assertEqual(out["features"].shape, (50, _FEATURE_DIM))
        self.assertEqual(out["frame_ids"][0].item(), 100)
        self.assertEqual(int(out["mask"].sum()), 1)  # only frame 120 falls in the window


class TestFarlWeightAvailability(unittest.TestCase):
    def test_checkpoints_mirrored(self):
        # Checked against the mirror, not the FaRL GitHub release: GitHub rate-limits
        # unauthenticated requests from shared CI address ranges, so a healthy weight can
        # read as missing. The release URL remains the runtime fallback.
        missing = [
            cfg["filename"]
            for cfg in _MODELS.values()
            if not hf_file_exists(_HF_REPO_ID, f"{cfg['filename']}.pth")
        ]
        self.assertEqual(missing, [], f"Missing from {_HF_REPO_ID}: {missing}")


if __name__ == "__main__":
    unittest.main()
