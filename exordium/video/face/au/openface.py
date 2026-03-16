from pathlib import Path

import numpy as np
import pandas as pd


def read_openface_au(
    csv_path: str | Path, confidence_thr: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    """Reads an OpenFace output CSV file and returns facial action unit data.

    Filters rows by confidence threshold and, when multiple faces are present,
    selects the track corresponding to the largest bounding box.

    Args:
        csv_path (str | Path): Path to the OpenFace output CSV file.
        confidence_thr (float, optional): Minimum confidence score for a row
            to be included. Defaults to 0.85.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of
            (frame_ids, au_values, au_names) where frame_ids has shape (T,),
            au_values has shape (T, 35), and au_names has shape (35,).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If no rows pass the confidence filter.
        ValueError: If the resulting AU array does not have shape (T, 35).

    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path, delimiter=",")
    header = df.columns.tolist()
    au_names = np.array(header[header.index("AU01_r") :])

    au_values = np.array(df.iloc[:, header.index("AU01_r") :].values.tolist())
    frame_ids = np.array(df.iloc[:, header.index("frame")].values.tolist())
    face_ids = np.array(df.iloc[:, header.index("face_id")].values.tolist())

    confidence = np.array(df.iloc[:, header.index("confidence")].astype(float).tolist())
    confidence_filter = confidence > confidence_thr

    frame_ids = frame_ids[confidence_filter]
    face_ids = face_ids[confidence_filter]
    au_values = au_values[confidence_filter, :]

    if sum(confidence_filter) == 0:
        raise ValueError("No valid data... skip")

    biggest_face_id = 0
    if len(set(list(face_ids))) > 1:
        print("More than one face detected. Select the biggest bb...")
        biggest_face_area = 0

        for face_id in set(list(face_ids)):
            face_df = df[df["face_id"] == face_id]
            x_coords = [face_df[f"x_{i}"].values[0] for i in range(68)]
            y_coords = [face_df[f"y_{i}"].values[0] for i in range(68)]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            bounding_box_area = width * height
            if bounding_box_area > biggest_face_area:
                biggest_face_id = face_id
                biggest_face_area = bounding_box_area

    face_filter = face_ids == biggest_face_id
    frame_ids = frame_ids[face_filter]
    au_values = au_values[face_filter, :]

    if au_values.ndim != 2 or au_values.shape[-1] != 35:
        raise ValueError(f"Expected shape is (T, 35) got instead {au_values.shape}")

    return frame_ids, au_values, au_names
