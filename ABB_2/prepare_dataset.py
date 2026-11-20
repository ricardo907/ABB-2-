"""Create provisional Exp001 labels assuming video frame zero equals log time zero.

The camera/log offset has not been independently calibrated. See research/README.md.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from time_alignment import nearest_sample_indices


DATA_DIR = Path("data/train/Exp001")
VIDEO_PATH = DATA_DIR / "Train_Angle1_real.mp4"
EXCEL_PATH = DATA_DIR / "Train_Angle1_real.xlsx"
OUTPUT_CSV = DATA_DIR / "synchronized.csv"
TIME_COLUMN = "Time(s)"
ANGLE_COLUMNS = [f"position(rad)_axis_{i}" for i in range(1, 7)]


def prepare():
    samples = pd.read_excel(EXCEL_PATH)
    required = [TIME_COLUMN, *ANGLE_COLUMNS]
    missing = set(required) - set(samples.columns)
    if missing:
        raise ValueError(f"Missing Excel columns: {sorted(missing)}")
    numeric = samples[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Excel timestamps or angle values contain NaN/infinity")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: frames={frame_count}, fps={fps}")

    frame_times = np.arange(frame_count, dtype=np.float64) / fps
    sample_indices = nearest_sample_indices(frame_times, numeric[TIME_COLUMN].to_numpy())
    angles = numeric[ANGLE_COLUMNS].to_numpy()[sample_indices]
    output = pd.DataFrame(angles, columns=[f"axis{i}" for i in range(1, 7)])
    output.insert(0, "time", frame_times)
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {frame_count} provisional frame labels from {len(samples)} angle samples: {OUTPUT_CSV}")
    print("WARNING: camera/log start offset is unverified; do not report these as ground-truth accuracy.")


if __name__ == "__main__":
    prepare()
