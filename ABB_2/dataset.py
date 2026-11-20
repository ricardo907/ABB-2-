"""Small image/angle dataset used by test_dataset.py."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


AXIS_COLUMNS = [f"axis{i}" for i in range(1, 7)]


class AngleDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        missing = {"image", *AXIS_COLUMNS} - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing label columns: {sorted(missing)}")
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.image_dir / str(row["image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row[AXIS_COLUMNS].to_numpy(dtype=np.float32)
        return image, label
