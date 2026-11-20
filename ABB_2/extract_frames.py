"""Extract Exp001 frames and pair with provisional zero-offset angle rows."""

from pathlib import Path

import cv2
import pandas as pd


DATA_DIR = Path("data/train/Exp001")
VIDEO_PATH = DATA_DIR / "Train_Angle1_real.mp4"
LABEL_PATH = DATA_DIR / "synchronized.csv"
OUTPUT_DIR = DATA_DIR / "frames"
LABEL_OUTPUT = DATA_DIR / "train_labels.csv"
AXIS_COLUMNS = [f"axis{i}" for i in range(1, 7)]


def extract():
    labels = pd.read_csv(LABEL_PATH)
    missing = set(AXIS_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"Missing synchronized angle columns: {sorted(missing)}")
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id >= len(labels):
                raise ValueError(
                    f"Video has more frames than synchronized labels ({len(labels)}); "
                    "rerun prepare_dataset.py and check video metadata"
                )
            image_name = f"frame_{frame_id:04d}.jpg"
            image_path = OUTPUT_DIR / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise OSError(f"Could not write frame: {image_path}")
            records.append([image_name, *labels.iloc[frame_id][AXIS_COLUMNS].tolist()])
            frame_id += 1
    finally:
        capture.release()

    if frame_id != len(labels):
        raise ValueError(
            f"Decoded {frame_id} frames but found {len(labels)} synchronized labels; "
            "check video metadata and synchronization"
        )
    pd.DataFrame(records, columns=["image", *AXIS_COLUMNS]).to_csv(
        LABEL_OUTPUT, index=False
    )
    print(f"Extracted {frame_id} labelled frames: {LABEL_OUTPUT}")


if __name__ == "__main__":
    extract()
