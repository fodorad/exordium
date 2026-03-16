import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from exordium.video.core.io import image_to_np
from exordium.video.face import MediaPipeFaceDetector
from exordium.video.face.gaze.unigaze import UnigazeWrapper
from exordium.video.face.headpose import SixDRepNetWrapper
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE


class UnigazeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        cls.TMP_DIR.mkdir(parents=True, exist_ok=True)

        cls.model = UnigazeWrapper()
        cls.face_detector = MediaPipeFaceDetector()
        cls.headpose = SixDRepNetWrapper(device_id=None)

        # Detect and crop faces from fixtures
        cls.face_image_rgb = image_to_np(IMAGE_FACE, "RGB")
        cls.emma_rgb = image_to_np(IMAGE_EMMA, "RGB")

        # IMAGE_FACE - single face looking at camera
        cls.face_dets = cls.face_detector.detect_image(cls.face_image_rgb)
        cls.face_crop = cls.face_dets[0].bb_crop()

        # IMAGE_EMMA - 3 faces (not looking at camera)
        cls.emma_dets = cls.face_detector.detect_image(cls.emma_rgb)
        cls.emma_crops = [det.bb_crop() for det in cls.emma_dets]

        # Use first Emma face as "not looking at camera" example
        cls.nocam_crop = cls.emma_crops[0]

        # Get roll angles from 3DDFA-V2
        cls.face_roll = cls.headpose.predict_single(cls.face_crop)["headpose"][2]
        cls.emma_rolls = [cls.headpose.predict_single(c)["headpose"][2] for c in cls.emma_crops]
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
        face_pil = Image.fromarray(np.uint8(self.face_crop))
        sample = self.model.transform(face_pil).unsqueeze(0).to(self.model.device)
        yaw, pitch = self.model(sample)
        self.assertIsInstance(yaw, torch.Tensor)
        self.assertIsInstance(pitch, torch.Tensor)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_call_batch(self):
        samples = []
        for crop in [self.face_crop, self.nocam_crop]:
            face_pil = Image.fromarray(np.uint8(crop))
            samples.append(self.model.transform(face_pil))
        batch = torch.stack(samples).to(self.model.device)
        yaw, pitch = self.model(batch)
        self.assertEqual(yaw.shape, (2,))
        self.assertEqual(pitch.shape, (2,))

    # --- predict_pipeline ---

    def test_predict_pipeline_no_rotation(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop])
        self.assertIsInstance(yaw, np.ndarray)
        self.assertIsInstance(pitch, np.ndarray)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_predict_pipeline_with_rotation(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_predict_pipeline_batch(self):
        yaw, pitch = self.model.predict_pipeline(self.emma_crops)
        self.assertEqual(yaw.shape, (len(self.emma_crops),))
        self.assertEqual(pitch.shape, (len(self.emma_crops),))

    # --- looking_at_camera ---

    def test_looking_at_camera_face_no_rotation(self):
        """IMAGE_FACE should be looking at the camera (no rotation)."""
        yaw, pitch = self.model.predict_pipeline([self.face_crop])
        result = UnigazeWrapper.looking_at_camera(yaw, pitch)
        self.assertTrue(result[0], "Face fixture should be looking at camera")

    def test_looking_at_camera_nocam_no_rotation(self):
        """IMAGE_FACE_NOCAM should NOT be looking at the camera (no rotation)."""
        yaw, pitch = self.model.predict_pipeline([self.nocam_crop])
        result = UnigazeWrapper.looking_at_camera(yaw, pitch)
        self.assertFalse(result[0], "Nocam fixture should not be looking at camera")

    def test_looking_at_camera_emma_no_rotation(self):
        """At least one emma face should NOT be looking at camera (no rotation)."""
        yaw, pitch = self.model.predict_pipeline(self.emma_crops)
        result = UnigazeWrapper.looking_at_camera(yaw, pitch)
        self.assertTrue(
            any(~result),
            "At least one emma face should not be looking at camera",
        )

    def test_looking_at_camera_batch_shape(self):
        yaw = np.array([0.1, 0.5, -0.8])
        pitch = np.array([0.05, 0.3, -0.1])
        result = UnigazeWrapper.looking_at_camera(yaw, pitch)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, bool)

    # --- visualize ---

    def test_visualize_single(self):
        yaw, pitch = self.model.predict_pipeline([self.face_crop])
        images = UnigazeWrapper.visualize([self.face_crop], yaw, pitch)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, self.face_crop.shape)

    def test_visualize_emma(self):
        yaw, pitch = self.model.predict_pipeline(self.emma_crops)
        images = UnigazeWrapper.visualize(self.emma_crops, yaw, pitch)
        self.assertEqual(len(images), len(self.emma_crops))
        for img, crop in zip(images, self.emma_crops):
            self.assertEqual(img.shape, crop.shape)

    # --- visualize saves: no rotation vs with rotation ---

    def test_visualize_saves_no_rotation(self):
        """Save visualizations WITHOUT rotation correction to tmp2."""

        # Face looking at camera
        yaw, pitch = self.model.predict_pipeline([self.face_crop])
        images = UnigazeWrapper.visualize([self.face_crop], yaw, pitch)
        out = self.TMP_DIR / "norot_face_cam.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out.exists())

        # Face NOT looking at camera
        yaw, pitch = self.model.predict_pipeline([self.nocam_crop])
        images = UnigazeWrapper.visualize([self.nocam_crop], yaw, pitch)
        out = self.TMP_DIR / "norot_face_nocam.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out.exists())

        # Emma - multiple faces
        yaw, pitch = self.model.predict_pipeline(self.emma_crops)
        images = UnigazeWrapper.visualize(self.emma_crops, yaw, pitch)
        for i, img in enumerate(images):
            out = self.TMP_DIR / f"norot_emma_{i}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.assertTrue(out.exists())

    def test_visualize_saves_with_rotation(self):
        """Save visualizations WITH rotation correction to tmp2."""

        # Face looking at camera
        yaw, pitch = self.model.predict_pipeline([self.face_crop], [self.face_roll])
        images = UnigazeWrapper.visualize([self.face_crop], yaw, pitch, [self.face_roll])
        out = self.TMP_DIR / "rot_face_cam.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out.exists())

        # Face NOT looking at camera
        yaw, pitch = self.model.predict_pipeline([self.nocam_crop], [self.nocam_roll])
        images = UnigazeWrapper.visualize([self.nocam_crop], yaw, pitch, [self.nocam_roll])
        out = self.TMP_DIR / "rot_face_nocam.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        self.assertTrue(out.exists())

        # Emma - multiple faces
        yaw, pitch = self.model.predict_pipeline(self.emma_crops, self.emma_rolls)
        images = UnigazeWrapper.visualize(self.emma_crops, yaw, pitch, self.emma_rolls)
        for i, img in enumerate(images):
            out = self.TMP_DIR / f"rot_emma_{i}.jpg"
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
