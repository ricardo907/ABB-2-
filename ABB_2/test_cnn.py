"""Compare RobotCNN with provisional Exp002 labels.

The video/log offset is unverified, so this MAE is not a validated accuracy.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_cnn import AXIS_COLUMNS, IMAGE_SIZE, RobotCNN


IMAGE_DIR = Path("data2/test/Exp002/frames")
LABEL_FILE = Path("data2/test/Exp002/test_labels.csv")
CHECKPOINT = Path("robot_angle_cnn.pth")
OUTPUT_CSV = Path("Exp002_prediction_result.csv")


class TestDataset(Dataset):
    def __init__(self, dataframe, image_dir=IMAGE_DIR):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self.image_dir / str(row["image"])) as source:
            image = source.convert("RGB")
        angles = torch.from_numpy(row[AXIS_COLUMNS].to_numpy(dtype=np.float32))
        return self.transform(image), angles


def evaluate():
    df = pd.read_csv(LABEL_FILE)
    missing = {"image", *AXIS_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No labelled test frames in {LABEL_FILE}")
    df[AXIS_COLUMNS] = df[AXIS_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[AXIS_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Test labels contain NaN or infinite values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotCNN().to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    true_values = []
    loader = DataLoader(TestDataset(df), batch_size=16, pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for images, angles in loader:
            predictions.append(model(images.to(device, non_blocking=True)).cpu().numpy())
            true_values.append(angles.numpy())
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)
    absolute_error = np.abs(predictions - true_values)
    print("WARNING: Exp002 video/log alignment is unverified; MAE is a provisional diagnostic.")
    print(f"Test frames: {len(df)}")
    print(f"Overall MAE: {absolute_error.mean():.6f} rad")
    for axis, value in enumerate(absolute_error.mean(axis=0), start=1):
        print(f"Axis {axis} MAE: {value:.6f} rad")

    result = pd.DataFrame({"image": df["image"]})
    for axis, name in enumerate(AXIS_COLUMNS):
        result[f"pred_{name}"] = predictions[:, axis]
        result[f"true_{name}"] = true_values[:, axis]
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate()
