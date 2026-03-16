import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from exordium.video.core.io import image_to_np
from exordium.video.face import MediaPipeFaceDetector
from exordium.video.face.gaze.l2csnet import L2csNetWrapper
from exordium.video.face.headpose import SixDRepNetWrapper
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE


class L2csNetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        cls.model = L2csNetWrapper()
        cls.face_detector = MediaPipeFaceDetector()
        cls.headpose = SixDRepNetWrapper(device_id=None)

        # Detect and crop faces from fixtures
        cls.face_image_rgb = image_to_np(IMAGE_FACE, "RGB")
        cls.emma_rgb = image_to_np(IMAGE_EMMA, "RGB")

        # IMAGE_FACE - single face looking at camera
        cls.face_dets = cls.face_detector.detect_image(cls.face_image_rgb)
        cls.face_crop = cls.face_dets[0].bb_crop_wide()

        # IMAGE_EMMA - 3 faces (not looking at camera)
        cls.emma_dets = cls.face_detector.detect_image(cls.emma_rgb)
        cls.emma_crops = [det.bb_crop_wide() for det in cls.emma_dets]

        # Use first Emma face as "not looking at camera" example
        cls.nocam_crop = cls.emma_crops[0]

        # Get roll angles from 3DDFA-V2
        cls.face_headpose = cls.headpose.predict_single(cls.face_crop)["headpose"]
        cls.face_roll = cls.face_headpose[2]

        cls.emma_rolls = []
        for crop in cls.emma_crops:
            headpose = cls.headpose.predict_single(crop)["headpose"]
            cls.emma_rolls.append(headpose[2])

        cls.nocam_roll = cls.emma_rolls[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TMP_DIR)

    # --- Initialization ---

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.device)

    # --- __call__ (preprocessed tensors) ---

    def test_call_single_face(self):
        """Test __call__ with a single preprocessed face tensor."""
        # Manually prepare tensor as predict_pipeline does internally
        import torch.nn.functional as F

        from exordium.video.core.io import images_to_np
        from exordium.video.core.transform import rotate_face

        faces_rgb = images_to_np([self.face_crop], "RGB", resize=None)
        faces_rgb = [rotate_face(face, self.face_roll)[0] for face in faces_rgb]
        samples = torch.stack(
            [torch.from_numpy(face).permute(2, 0, 1).float() / 255.0 for face in faces_rgb]
        ).to(self.model.device)
        samples = F.interpolate(samples, size=(448, 448), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.model.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.model.device).view(1, 3, 1, 1)
        samples = (samples - mean) / std

        yaw, pitch = self.model(samples)
        self.assertIsInstance(yaw, torch.Tensor)
        self.assertIsInstance(pitch, torch.Tensor)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_call_batch(self):
        """Test __call__ with a batch of preprocessed face tensors."""
        import torch.nn.functional as F

        from exordium.video.core.io import images_to_np
        from exordium.video.core.transform import rotate_face

        faces_rgb = images_to_np([self.face_crop, self.nocam_crop], "RGB", resize=None)
        roll_angles = [self.face_roll, self.nocam_roll]
        faces_rgb = [rotate_face(face, roll)[0] for face, roll in zip(faces_rgb, roll_angles)]
        samples = torch.stack(
            [torch.from_numpy(face).permute(2, 0, 1).float() / 255.0 for face in faces_rgb]
        ).to(self.model.device)
        samples = F.interpolate(samples, size=(448, 448), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.model.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.model.device).view(1, 3, 1, 1)
        samples = (samples - mean) / std

        yaw, pitch = self.model(samples)
        self.assertEqual(yaw.shape, (2,))
        self.assertEqual(pitch.shape, (2,))

    def test_call_returns_radians(self):
        """Test that __call__ returns values in radians."""
        import torch.nn.functional as F

        from exordium.video.core.io import images_to_np
        from exordium.video.core.transform import rotate_face

        faces_rgb = images_to_np([self.face_crop], "RGB", resize=None)
        faces_rgb = [rotate_face(face, self.face_roll)[0] for face in faces_rgb]
        samples = torch.stack(
            [torch.from_numpy(face).permute(2, 0, 1).float() / 255.0 for face in faces_rgb]
        ).to(self.model.device)
        samples = F.interpolate(samples, size=(448, 448), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.model.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.model.device).view(1, 3, 1, 1)
        samples = (samples - mean) / std

        yaw, pitch = self.model(samples)
        self.assertTrue(torch.all(torch.abs(yaw) < np.pi))
        self.assertTrue(torch.all(torch.abs(pitch) < np.pi))

    # --- predict_pipeline ---

    def test_predict_pipeline_single(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        self.assertIsInstance(yaw, np.ndarray)
        self.assertIsInstance(pitch, np.ndarray)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_predict_pipeline_batch(self):
        yaw, pitch = self.model.predict_pipeline(self.emma_crops, self.emma_rolls)
        self.assertEqual(yaw.shape, (len(self.emma_crops),))
        self.assertEqual(pitch.shape, (len(self.emma_crops),))

    def test_predict_pipeline_values_in_range(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        self.assertTrue(np.all(np.abs(yaw) < np.pi))
        self.assertTrue(np.all(np.abs(pitch) < np.pi))

    # --- looking_at_camera ---

    def test_looking_at_camera_face(self):
        """IMAGE_FACE should be looking at the camera."""
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        result = L2csNetWrapper.looking_at_camera(yaw, pitch)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, bool)
        self.assertTrue(result[0], "Face fixture should be looking at camera")

    def test_looking_at_camera_emma(self):
        """All emma faces should NOT be looking at the camera."""
        yaw, pitch = self.model.predict_pipeline(self.emma_crops, self.emma_rolls)
        result = L2csNetWrapper.looking_at_camera(yaw, pitch)
        self.assertEqual(result.shape, (len(self.emma_crops),))
        self.assertTrue(
            all(~result),
            "All emma faces should not be looking at camera",
        )

    def test_looking_at_camera_batch_shape(self):
        yaw = np.array([0.1, 0.5, -0.8])
        pitch = np.array([0.05, 0.3, -0.1])
        result = L2csNetWrapper.looking_at_camera(yaw, pitch)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, bool)

    # --- visualize ---

    def test_visualize_single(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        images = L2csNetWrapper.visualize([self.face_crop], yaw, pitch, [self.face_roll])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, self.face_crop.shape)

    def test_visualize_emma(self):
        yaw, pitch = self.model.predict_pipeline(self.emma_crops, self.emma_rolls)
        images = L2csNetWrapper.visualize(self.emma_crops, yaw, pitch, self.emma_rolls)
        self.assertEqual(len(images), len(self.emma_crops))
        for img, crop in zip(images, self.emma_crops):
            self.assertEqual(img.shape, crop.shape)

    def test_visualize_saves_images(self):
        """Save visualizations to TMP_DIR for manual inspection."""
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)

        # Face looking at camera
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        images = L2csNetWrapper.visualize([self.face_crop], yaw, pitch, [self.face_roll])
        out_path = self.TMP_DIR / "gaze_face_cam.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out_path.exists())

        # Face NOT looking at camera
        yaw, pitch = self.model.predict_pipeline([self.nocam_crop], [self.nocam_roll])
        images = L2csNetWrapper.visualize([self.nocam_crop], yaw, pitch, [self.nocam_roll])
        out_path = self.TMP_DIR / "gaze_face_nocam.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out_path.exists())

        # Emma - multiple faces
        yaw, pitch = self.model.predict_pipeline(self.emma_crops, self.emma_rolls)
        images = L2csNetWrapper.visualize(self.emma_crops, yaw, pitch, self.emma_rolls)
        for i, img in enumerate(images):
            out_path = self.TMP_DIR / f"gaze_emma_{i}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()


class TestL2CSBuilder(unittest.TestCase):
    """Tests for L2CS_Builder factory function."""

    def test_resnet18(self):
        from exordium.video.face.gaze.l2csnet import L2CS_Builder

        model = L2CS_Builder(arch="ResNet18", bins=90)
        self.assertIsNotNone(model)

    def test_resnet34(self):
        from exordium.video.face.gaze.l2csnet import L2CS_Builder

        model = L2CS_Builder(arch="ResNet34", bins=90)
        self.assertIsNotNone(model)

    def test_resnet101(self):
        from exordium.video.face.gaze.l2csnet import L2CS_Builder

        model = L2CS_Builder(arch="ResNet101", bins=90)
        self.assertIsNotNone(model)

    def test_resnet152(self):
        from exordium.video.face.gaze.l2csnet import L2CS_Builder

        model = L2CS_Builder(arch="ResNet152", bins=90)
        self.assertIsNotNone(model)

    def test_invalid_arch_raises(self):
        from exordium.video.face.gaze.l2csnet import L2CS_Builder

        with self.assertRaises(ValueError):
            L2CS_Builder(arch="InvalidArch", bins=90)
