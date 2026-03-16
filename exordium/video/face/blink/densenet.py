"""Blink detection wrapper using DenseNet121 from blinklinmult."""

import cv2
import numpy as np
import torch
from blinklinmult.models import DenseNet121

from exordium.utils.device import get_torch_device


class BlinkDenseNet121Wrapper:
    """Blink detection wrapper using BlinkDenseNet121.

    Predicts eye state (open/closed) from eye patch crops using a DenseNet121
    model trained for blink detection. Handles self-occlusion based on head pose.

    Args:
        model_name: Model variant from blinklinmult. Default: ``'densenet121-union'``.
        device_id: Device index. ``None`` or negative for CPU.
        yaw_threshold: Head yaw angle (degrees) beyond which an eye
            is considered occluded. Default: 40.0.

    """

    def __init__(
        self,
        device_id: int | None = None,
        yaw_threshold: float = 40.0,
    ) -> None:
        self.device = get_torch_device(device_id)
        self.yaw_threshold = yaw_threshold

        self.model = DenseNet121(weights="densenet121-union")
        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """Predict eye state from preprocessed eye patch tensors.

        Args:
            samples: Preprocessed eye patch tensor of shape ``(B, 3, 64, 64)``.

        Returns:
            Blink probabilities of shape ``(B,)``.
            Values in [0, 1] where >0.5 indicates closed/blinking.

        """
        logits = self.model(samples)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B,)
        return probs

    @torch.inference_mode()
    def predict_pipeline(
        self,
        frames: np.ndarray,
        landmarks: np.ndarray,
        headpose: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict eye state from frames with landmark-based cropping (batched).

        Extracts eye patches from frames using MediaPipe FaceDetector landmarks,
        runs blink detection, and applies occlusion masking based on head pose.

        Args:
            frames: Batch of input frames in RGB format, shape ``(B, H, W, 3)``.
            landmarks: MediaPipe FaceDetector keypoints, shape ``(B, 6, 2)``.
                Keypoint order: [right_eye, left_eye, nose, mouth, right_ear, left_ear].
            headpose: Head pose angles, shape ``(B, 3)`` with ``[yaw, pitch, roll]``
                in degrees. If provided, masks predictions for occluded eyes based
                on yaw threshold. ``None`` means no occlusion filtering.

        Returns:
            Tuple of 4 arrays, each of shape ``(B,)``:
                - left_eye_state: Left eye blink probabilities (0-1)
                - right_eye_state: Right eye blink probabilities (0-1)
                - left_eye_valid: Boolean mask indicating valid left eye predictions
                - right_eye_valid: Boolean mask indicating valid right eye predictions

        """
        B = frames.shape[0]

        # Determine visibility masks based on head pose
        left_eye_valid = np.ones(B, dtype=bool)
        right_eye_valid = np.ones(B, dtype=bool)

        if headpose is not None:
            yaw = headpose[:, 0]  # (B,) - positive = head turned right
            # If head turned right, left eye occluded by nose
            left_eye_valid = yaw <= self.yaw_threshold
            # If head turned left, right eye occluded
            right_eye_valid = yaw >= -self.yaw_threshold

        # Extract eye coordinates from landmarks
        # MediaPipe FaceDetector: 0=left_eye, 1=right_eye
        left_eye_coords = landmarks[:, 0]  # (B, 2)
        right_eye_coords = landmarks[:, 1]  # (B, 2)

        # Extract all eye patches
        left_eye_patches = []
        right_eye_patches = []

        for i in range(B):
            frame = frames[i]
            h, w = frame.shape[:2]

            left_x, left_y = int(left_eye_coords[i, 0]), int(left_eye_coords[i, 1])
            right_x, right_y = int(right_eye_coords[i, 0]), int(right_eye_coords[i, 1])

            # Calculate eye distance for adaptive crop size
            eye_distance = abs(right_x - left_x)
            if eye_distance < 10:
                # Invalid detection - mark as invalid
                left_eye_valid[i] = False
                right_eye_valid[i] = False
                left_eye_patches.append(np.zeros((64, 64, 3), dtype=np.uint8))
                right_eye_patches.append(np.zeros((64, 64, 3), dtype=np.uint8))
                continue

            crop_size = max(int(eye_distance * 1.5), 100)

            # Extract left eye patch
            x1 = max(0, left_x - crop_size // 2)
            x2 = min(w, left_x + crop_size // 2)
            y1 = max(0, left_y - crop_size // 2)
            y2 = min(h, left_y + crop_size // 2)
            left_crop = frame[y1:y2, x1:x2]

            if left_crop.size == 0:
                left_eye_valid[i] = False
                left_crop = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                left_crop = cv2.resize(left_crop, (64, 64), interpolation=cv2.INTER_CUBIC)

            left_eye_patches.append(left_crop)

            # Extract right eye patch
            x1 = max(0, right_x - crop_size // 2)
            x2 = min(w, right_x + crop_size // 2)
            y1 = max(0, right_y - crop_size // 2)
            y2 = min(h, right_y + crop_size // 2)
            right_crop = frame[y1:y2, x1:x2]

            if right_crop.size == 0:
                right_eye_valid[i] = False
                right_crop = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                right_crop = cv2.resize(right_crop, (64, 64), interpolation=cv2.INTER_CUBIC)

            right_eye_patches.append(right_crop)

        # Stack all eye patches: (B, H, W, C) -> (B, C, H, W)
        left_eye_patches = np.stack(left_eye_patches)  # (B, 64, 64, 3)
        right_eye_patches = np.stack(right_eye_patches)  # (B, 64, 64, 3)

        # Convert to tensors and normalize
        left_tensor = (
            torch.from_numpy(left_eye_patches).permute(0, 3, 1, 2).float() / 255.0
        )  # (B, 3, 64, 64)
        right_tensor = (
            torch.from_numpy(right_eye_patches).permute(0, 3, 1, 2).float() / 255.0
        )  # (B, 3, 64, 64)

        # Normalize with ImageNet stats
        left_tensor = (left_tensor - self.mean) / self.std
        right_tensor = (right_tensor - self.mean) / self.std

        # Move to device
        left_tensor = left_tensor.to(self.device)
        right_tensor = right_tensor.to(self.device)

        # Run inference
        left_eye_state = self(left_tensor).cpu().numpy()  # (B,)
        right_eye_state = self(right_tensor).cpu().numpy()  # (B,)

        return left_eye_state, right_eye_state, left_eye_valid, right_eye_valid

    def predict_frame(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        headpose: np.ndarray | None = None,
        return_patches: bool = False,
    ) -> tuple[float, float, bool, bool] | tuple[float, float, bool, bool, np.ndarray, np.ndarray]:
        """Predict eye state from a single frame (convenience method).

        Args:
            frame: Single input frame in RGB format, shape ``(H, W, 3)``.
            landmarks: MediaPipe FaceDetector keypoints, shape ``(6, 2)``.
                Keypoint order: [right_eye, left_eye, nose, mouth, right_ear, left_ear].
            headpose: Head pose angles ``[yaw, pitch, roll]`` in degrees.
                If provided, masks predictions for occluded eyes based on yaw.
                ``None`` means no occlusion filtering.
            return_patches: If ``True``, also return the extracted 64x64 eye patches.

        Returns:
            If ``return_patches=False``:
                Tuple of 4 values:
                    - left_eye_state: Left eye blink probability (0-1)
                    - right_eye_state: Right eye blink probability (0-1)
                    - left_eye_valid: Boolean indicating valid left eye prediction
                    - right_eye_valid: Boolean indicating valid right eye prediction
            If ``return_patches=True``:
                Tuple of 6 values (4 above plus):
                    - left_patch: Left eye patch, shape ``(64, 64, 3)`` in RGB
                    - right_patch: Right eye patch, shape ``(64, 64, 3)`` in RGB

        """
        h, w = frame.shape[:2]

        # Extract eye coordinates
        left_eye_x, left_eye_y = int(landmarks[0, 0]), int(landmarks[0, 1])
        right_eye_x, right_eye_y = int(landmarks[1, 0]), int(landmarks[1, 1])

        # Calculate crop size
        eye_distance = abs(right_eye_x - left_eye_x)
        crop_size = max(int(eye_distance * 1.5), 100)

        # Extract eye patches for visualization
        if return_patches:
            # Left eye patch
            x1 = max(0, left_eye_x - crop_size // 2)
            x2 = min(w, left_eye_x + crop_size // 2)
            y1 = max(0, left_eye_y - crop_size // 2)
            y2 = min(h, left_eye_y + crop_size // 2)
            left_patch = frame[y1:y2, x1:x2]
            if left_patch.size > 0:
                left_patch = cv2.resize(left_patch, (64, 64), interpolation=cv2.INTER_CUBIC)
            else:
                left_patch = np.zeros((64, 64, 3), dtype=np.uint8)

            # Right eye patch
            x1 = max(0, right_eye_x - crop_size // 2)
            x2 = min(w, right_eye_x + crop_size // 2)
            y1 = max(0, right_eye_y - crop_size // 2)
            y2 = min(h, right_eye_y + crop_size // 2)
            right_patch = frame[y1:y2, x1:x2]
            if right_patch.size > 0:
                right_patch = cv2.resize(right_patch, (64, 64), interpolation=cv2.INTER_CUBIC)
            else:
                right_patch = np.zeros((64, 64, 3), dtype=np.uint8)

        # Add batch dimension
        frames = frame[np.newaxis, ...]  # (1, H, W, 3)
        landmarks_batch = landmarks[np.newaxis, ...]  # (1, 6, 2)
        headpose_batch = headpose[np.newaxis, ...] if headpose is not None else None  # (1, 3)

        # Run batched prediction
        left_state, right_state, left_valid, right_valid = self.predict_pipeline(
            frames, landmarks_batch, headpose_batch
        )

        # Extract single values
        result = (
            float(left_state[0]),
            float(right_state[0]),
            bool(left_valid[0]),
            bool(right_valid[0]),
        )

        if return_patches:
            return result + (left_patch, right_patch)
        return result

    @staticmethod
    def eyes_open(
        left_eye_state: np.ndarray,
        right_eye_state: np.ndarray,
        left_eye_valid: np.ndarray,
        right_eye_valid: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Determine if eyes are open based on blink predictions.

        Args:
            left_eye_state: Left eye blink probabilities, shape ``(B,)``.
            right_eye_state: Right eye blink probabilities, shape ``(B,)``.
            left_eye_valid: Boolean mask for valid left eye predictions, shape ``(B,)``.
            right_eye_valid: Boolean mask for valid right eye predictions, shape ``(B,)``.
            threshold: Threshold for closed eye. Default: 0.5.

        Returns:
            Boolean array of shape ``(B,)`` indicating if eyes are open.
            Eyes are considered open if BOTH valid eyes have prob < threshold.

        """
        # Start with all True
        eyes_open = np.ones(len(left_eye_state), dtype=bool)

        # If left eye is valid and closed, mark as not open
        eyes_open &= ~(left_eye_valid & (left_eye_state >= threshold))

        # If right eye is valid and closed, mark as not open
        eyes_open &= ~(right_eye_valid & (right_eye_state >= threshold))

        # If neither eye is valid, mark as not open
        eyes_open &= left_eye_valid | right_eye_valid

        return eyes_open

    @staticmethod
    def visualize(
        frames: np.ndarray,
        left_eye_state: np.ndarray,
        right_eye_state: np.ndarray,
        left_eye_valid: np.ndarray,
        right_eye_valid: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize blink predictions on frames.

        Args:
            frames: Input frames in RGB format, shape ``(B, H, W, 3)``.
            left_eye_state: Left eye blink probabilities, shape ``(B,)``.
            right_eye_state: Right eye blink probabilities, shape ``(B,)``.
            left_eye_valid: Boolean mask for valid left eye predictions, shape ``(B,)``.
            right_eye_valid: Boolean mask for valid right eye predictions, shape ``(B,)``.
            landmarks: Optional MediaPipe landmarks for drawing eye locations, shape ``(B, 6, 2)``.

        Returns:
            Frames with visualization overlays, shape ``(B, H, W, 3)``.

        """
        vis_frames = frames.copy()
        B = frames.shape[0]

        for i in range(B):
            frame = vis_frames[i]
            y_offset = 30

            # Draw left eye state
            if left_eye_valid[i]:
                left_prob = left_eye_state[i]
                text = f"L: {left_prob:.2f}"
                color = (255, 0, 0) if left_prob > 0.5 else (0, 255, 0)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 35

            # Draw right eye state
            if right_eye_valid[i]:
                right_prob = right_eye_state[i]
                text = f"R: {right_prob:.2f}"
                color = (255, 0, 0) if right_prob > 0.5 else (0, 255, 0)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw eye locations if landmarks provided
            if landmarks is not None:
                # MediaPipe FaceDetector: 0=left_eye, 1=right_eye
                left_x, left_y = int(landmarks[i, 0, 0]), int(landmarks[i, 0, 1])
                right_x, right_y = int(landmarks[i, 1, 0]), int(landmarks[i, 1, 1])

                if left_eye_valid[i]:
                    left_prob = left_eye_state[i]
                    color = (255, 0, 0) if left_prob > 0.5 else (0, 255, 0)
                    cv2.circle(frame, (left_x, left_y), 5, color, -1)

                if right_eye_valid[i]:
                    right_prob = right_eye_state[i]
                    color = (255, 0, 0) if right_prob > 0.5 else (0, 255, 0)
                    cv2.circle(frame, (right_x, right_y), 5, color, -1)

        return vis_frames
