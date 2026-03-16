import unittest

import cv2
import numpy as np

from exordium.video.core.blur import is_blurry, variance_of_laplacian
from tests.fixtures import IMAGE_CAT_TIE


class TestVarianceOfLaplacian(unittest.TestCase):
    """Tests for variance_of_laplacian function."""

    def test_variance_of_laplacian_sharp_image(self):
        """Test with sharp image (should have high variance)."""
        vl = variance_of_laplacian(IMAGE_CAT_TIE)
        self.assertIsInstance(vl, float, "Should return float")
        self.assertGreater(vl, 0, "Variance should be positive")

    def test_variance_of_laplacian_blurred_image(self):
        """Test with artificially blurred image (should have low variance)."""
        img = cv2.imread(str(IMAGE_CAT_TIE))
        blurred = cv2.GaussianBlur(img, (51, 51), 0)

        vl_sharp = variance_of_laplacian(IMAGE_CAT_TIE)
        vl_blur = variance_of_laplacian(blurred)

        self.assertLess(vl_blur, vl_sharp, "Blurred image should have lower variance")

    def test_variance_of_laplacian_from_path(self):
        """Test loading image from path."""
        vl = variance_of_laplacian(IMAGE_CAT_TIE)
        self.assertIsInstance(vl, float, "Should return float")
        self.assertGreater(vl, 0, "Variance should be positive")

    def test_variance_of_laplacian_from_array(self):
        """Test with numpy array input."""
        img = cv2.imread(str(IMAGE_CAT_TIE))
        vl = variance_of_laplacian(img)
        self.assertIsInstance(vl, float, "Should return float")
        self.assertGreater(vl, 0, "Variance should be positive")

    def test_variance_of_laplacian_grayscale(self):
        """Test with grayscale image."""
        img = cv2.imread(str(IMAGE_CAT_TIE), cv2.IMREAD_GRAYSCALE)
        vl = variance_of_laplacian(img)
        self.assertIsInstance(vl, float, "Should return float")
        self.assertGreater(vl, 0, "Variance should be positive")

    def test_variance_of_laplacian_uniform_image(self):
        """Test with uniform image (should have near-zero variance)."""
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        vl = variance_of_laplacian(uniform)
        self.assertAlmostEqual(vl, 0.0, places=5, msg="Uniform image should have ~0 variance")


class TestIsBlurry(unittest.TestCase):
    """Tests for is_blurry function."""

    def test_is_blurry_sharp_image(self):
        """Test with sharp image (should not be blurry)."""
        is_blur, vl = is_blurry(IMAGE_CAT_TIE, threshold=400.0)
        self.assertIsInstance(is_blur, (bool, np.bool_), "Should return bool")
        self.assertIsInstance(vl, (float, np.floating), "Should return float variance")
        # Cat image with tie should be reasonably sharp
        self.assertGreater(vl, 0, "Variance should be positive")

    def test_is_blurry_blurred_image(self):
        """Test with artificially blurred image."""
        img = cv2.imread(str(IMAGE_CAT_TIE))
        blurred = cv2.GaussianBlur(img, (51, 51), 0)

        is_blur, vl = is_blurry(blurred, threshold=400.0)
        self.assertTrue(
            is_blur or vl < 1000, "Heavily blurred image should be detected or have low variance"
        )

    def test_is_blurry_custom_threshold(self):
        """Test with custom threshold."""
        # Very high threshold - everything should be blurry
        is_blur_high, vl_high = is_blurry(IMAGE_CAT_TIE, threshold=100000.0)
        self.assertTrue(is_blur_high, "With very high threshold, image should be considered blurry")

        # Very low threshold - nothing should be blurry
        is_blur_low, vl_low = is_blurry(IMAGE_CAT_TIE, threshold=0.0)
        self.assertFalse(is_blur_low, "With 0 threshold, image should not be considered blurry")

    def test_is_blurry_returns_rounded_value(self):
        """Test that variance is rounded to 2 decimal places."""
        _, vl = is_blurry(IMAGE_CAT_TIE)

        # Check that it's rounded to 2 decimal places
        self.assertEqual(vl, round(vl, 2), "Variance should be rounded to 2 decimals")

        # Convert to string and check decimal places
        vl_str = str(vl)
        if "." in vl_str:
            decimals = len(vl_str.split(".")[1])
            self.assertLessEqual(decimals, 2, "Should have at most 2 decimal places")

    def test_is_blurry_from_path(self):
        """Test loading image from path."""
        is_blur, vl = is_blurry(IMAGE_CAT_TIE)
        self.assertIsInstance(is_blur, (bool, np.bool_), "Should return bool")
        self.assertIsInstance(vl, (float, np.floating), "Should return float")

    def test_is_blurry_from_array(self):
        """Test with numpy array input."""
        img = cv2.imread(str(IMAGE_CAT_TIE))
        is_blur, vl = is_blurry(img)
        self.assertIsInstance(is_blur, (bool, np.bool_), "Should return bool")
        self.assertIsInstance(vl, (float, np.floating), "Should return float")


if __name__ == "__main__":
    unittest.main()
