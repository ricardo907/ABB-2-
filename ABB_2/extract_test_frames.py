"""Extract Exp002 frames with provisional first-frame/log-zero alignment.

The offset is unverified; see research/README.md before interpreting MAE.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from time_alignment import nearest_sample_indices


DATA_DIR = Path("data2/test/Exp002")
VIDEO_PATH = DATA_DIR / "Test_Angle2_real.mp4"
EXCEL_PATH = DATA_DIR / "Test_Angle2_real.xlsx"
OUTPUT_DIR = DATA_DIR / "frames"
LABEL_OUTPUT = DATA_DIR / "test_labels.csv"
TIME_COLUMN = "Time(s)"
ANGLE_COLUMNS = [f"position(rad)_axis_{i}" for i in range(1, 7)]


def extract():
    samples = pd.read_excel(EXCEL_PATH)
    required = [TIME_COLUMN, *ANGLE_COLUMNS]
    missing = set(required) - set(samples.columns)
    if missing:
        raise ValueError(f"Missing Excel columns: {sorted(missing)}")
    numeric = samples[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Excel timestamps or angle values contain NaN/infinity")
    sample_times = numeric[TIME_COLUMN].to_numpy(dtype=np.float64)
    nearest_sample_indices(np.array([], dtype=np.float64), sample_times)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_names = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image_name = f"frame_{len(image_names):04d}.jpg"
            image_path = OUTPUT_DIR / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not write frame: {image_path}")
            image_names.append(image_name)
    finally:
        capture.release()
    if not image_names:
        raise ValueError(f"No frames decoded from {VIDEO_PATH}")

    frame_times = np.arange(len(image_names), dtype=np.float64) / fps
    sample_duration = sample_times[-1] - sample_times[0]
    sample_period = np.median(np.diff(sample_times)) if len(sample_times) > 1 else 0.0
    tolerance = max(1.0 / fps, sample_period)
    if frame_times[-1] > sample_duration + tolerance:
        raise ValueError(
            "Video extends beyond Excel timestamps; check recording synchronization "
            "instead of reusing the last angle for extra frames"
        )
    indices = nearest_sample_indices(frame_times, sample_times)
    angles = numeric[ANGLE_COLUMNS].to_numpy()[indices]
    result = pd.DataFrame(angles, columns=[f"axis{i}" for i in range(1, 7)])
    result.insert(0, "image", image_names)
    result.to_csv(LABEL_OUTPUT, index=False)
    print(f"Extracted {len(image_names)} provisionally labelled frames: {LABEL_OUTPUT}")
    print("WARNING: camera/log start offset is unverified; do not report these as ground-truth accuracy.")


if __name__ == "__main__":
    extract()
