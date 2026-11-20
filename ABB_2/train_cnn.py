"""Train the six-axis image regressor from Exp001 frames.

Run this module as a script. Importing RobotCNN for inference has no side effects.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATA_DIR = Path("data/train/Exp001")
IMAGE_DIR = DATA_DIR / "frames"
LABEL_FILE = DATA_DIR / "train_labels.csv"
CHECKPOINT = Path("robot_angle_cnn.pth")
AXIS_COLUMNS = [f"axis{i}" for i in range(1, 7)]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
PATIENCE = 10


class RobotDataset(Dataset):
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
        image_path = self.image_dir / str(row["image"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        angle = torch.from_numpy(row[AXIS_COLUMNS].to_numpy(dtype=np.float32))
        return image, angle


class RobotCNN(nn.Module):
    """Keep layer names and shapes compatible with existing .pth checkpoints."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def split_chronologically(dataframe, validation_fraction=0.2, gap_fraction=0.05):
    """Hold out the last recording segment, leaving a gap between sets.

    This prevents adjacent and near-duplicate frames from being randomly mixed.
    Validation on another recording is still preferable when one is available.
    """
    frame_number = dataframe["image"].astype(str).str.extract(r"frame_(\d+)\.[^.]+$", expand=False)
    if frame_number.notna().all():
        dataframe = dataframe.assign(_frame=frame_number.astype(int)).sort_values("_frame")
    dataframe = dataframe.reset_index(drop=True).drop(columns="_frame", errors="ignore")
    count = len(dataframe)
    if count < 5:
        raise ValueError("At least 5 labelled frames are needed for a train/validation split")
    validation_count = max(1, math.ceil(count * validation_fraction))
    gap_count = max(1, math.ceil(count * gap_fraction))
    train_end = count - validation_count - gap_count
    if train_end < 1:
        raise ValueError("Not enough frames after reserving validation and gap segments")
    return dataframe.iloc[:train_end], dataframe.iloc[-validation_count:], gap_count


def read_labels(label_file=LABEL_FILE):
    df = pd.read_csv(label_file)
    missing = {"image", *AXIS_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    df[AXIS_COLUMNS] = df[AXIS_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[AXIS_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Angle labels contain NaN or infinite values")
    return df


def train_model():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_labels()
    train_df, val_df, gap_count = split_chronologically(df)
    print(
        f"Device={device}; train={len(train_df)}, gap={gap_count}, "
        f"validation={len(val_df)} (held-out recording tail)"
    )

    train_loader = DataLoader(
        RobotDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        RobotDataset(val_df), batch_size=BATCH_SIZE,
        pin_memory=device.type == "cuda",
    )
    model = RobotCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_mse = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_squared_error = 0.0
        train_values = 0
        for images, angles in train_loader:
            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, angles)
            loss.backward()
            optimizer.step()
            train_squared_error += loss.item() * angles.numel()
            train_values += angles.numel()

        model.eval()
        val_squared_error = 0.0
        val_absolute_error = 0.0
        val_values = 0
        with torch.inference_mode():
            for images, angles in val_loader:
                images = images.to(device, non_blocking=True)
                angles = angles.to(device, non_blocking=True)
                error = model(images) - angles
                val_squared_error += error.square().sum().item()
                val_absolute_error += error.abs().sum().item()
                val_values += angles.numel()

        train_mse = train_squared_error / train_values
        val_mse = val_squared_error / val_values
        val_mae = val_absolute_error / val_values
        print(
            f"Epoch {epoch}/{EPOCHS}: train MSE={train_mse:.6f}, "
            f"val MSE={val_mse:.6f}, val MAE={val_mae:.6f} rad"
        )
        if not math.isfinite(val_mse):
            raise RuntimeError("Validation loss is not finite; checkpoint was not replaced")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            stale_epochs = 0
            temporary = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".tmp")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, CHECKPOINT)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without improvement")
                break

    print(f"Best validation MSE={best_val_mse:.6f}; checkpoint={CHECKPOINT}")


if __name__ == "__main__":
    train_model()
