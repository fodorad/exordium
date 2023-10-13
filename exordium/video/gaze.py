import numpy as np
import math
import torch


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def gazeto3d(gaze: np.ndarray) -> np.ndarray:
    """Convert the gaze pitch and yaw angles into 3D vector"""

    if not gaze.shape == (2,):
        raise ValueError(f"Invalid gaze vector. The values should be pitch and yaw angles. Expected shape is (2,) got instead {gaze.shape}")

    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def pitchyaw_to_pixel(pitch: float, yaw: float, length: float = 1.0) -> np.ndarray:
    """Convert the gaze pitch and yaw angles to XY coords.

    Args:
        pitch (float): pitch angle (looking vertically) in degree.
        yaw (float): yaw angle (looking horizontally) in degree.
        length (float, optional): length of the vector. 1.0 means unit length. Defaults to 1.0.

    Returns:
        np.ndarray: XY coords
    """
    dx = -length * np.sin(pitch) * np.cos(yaw)
    dy = -length * np.sin(yaw)
    return np.array([dx, dy])


def spherical2cartesial(x):
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


def compute_angular_error(input, target):
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)
    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180 * torch.mean(output_dot) / math.pi
    return output_dot


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def looking_at_camera_yaw_pitch(yaw, pitch, thr: float = 0.5) -> bool:
    xy = pitchyaw_to_pixel(pitch, yaw, length=1)
    return looking_at_camera_xy(xy, thr)

def looking_at_camera_xy(xy: np.ndarray, thr: float = 0.5) -> bool:
    return bool(np.linalg.norm(xy) < thr)