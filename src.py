# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.stats import skew, kurtosis
from scipy.optimize import lsq_linear

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# 永远以 src.py 所在目录为准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HAS_CATBOOST = True
HAS_XGBOOST = True
HAS_TORCH = True

try:
    from catboost import CatBoostRegressor
except Exception:
    HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    HAS_TORCH = False


@dataclass
class CFG:
    train_meta_path: Optional[str] = None
    test_meta_path: Optional[str] = None

    train_signal_dir: str = os.path.join(BASE_DIR, "train")
    test_signal_dir: str = os.path.join(BASE_DIR, "test")
    output_dir: str = os.path.join(BASE_DIR, "outputs")

    random_seed: int = 42
    n_splits: int = 5
    use_stratified_cv: bool = True
    n_target_bins: int = 10

    signal_length: int = 8192
    n_harmonics: int = 5
    harmonic_band_hz: float = 2.0

    target_mode: str = "absolute"   # absolute / residual / relative

    use_random_forest: bool = True
    use_mlp: bool = True
    use_catboost: bool = True
    use_xgboost: bool = True
    use_cnn: bool = True

    # 默认只用树模型做 stacking，先不要把 cnn / mlp 融进去
    stack_candidate_models: Tuple[str, ...] = ("catboost", "xgboost", "rf")
    stack_alpha: float = 1.0

    rf_estimators: int = 400
    rf_max_depth: Optional[int] = None

    mlp_hidden: Tuple[int, ...] = (256, 128)
    mlp_max_iter: int = 400

    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_lr: float = 0.05

    xgb_estimators: int = 500
    xgb_depth: int = 6
    xgb_lr: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample: float = 0.9

    cnn_epochs: int = 16
    cnn_batch_size: int = 32
    cnn_lr: float = 1e-3
    cnn_weight_decay: float = 1e-4
    cnn_patience: int = 4
    cnn_num_workers: int = 0
    cnn_device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    save_oof_csv: bool = True
    save_test_csv: bool = True
    save_metrics_json: bool = True

    # 论文资产导出
    save_paper_assets: bool = True
    paper_dir: str = os.path.join(BASE_DIR, "outputs", "paper_assets")
    save_pdf_figures: bool = True
    save_png_figures: bool = True
    export_latex_tables: bool = True


CFG_OBJ = CFG()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_name(name: str) -> str:
    name = os.path.splitext(str(name))[0].lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def auto_detect_metadata(preferred_name: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, f"{preferred_name}.csv"),
        os.path.join(BASE_DIR, f"{preferred_name}.xlsx"),
        os.path.join(BASE_DIR, f"{preferred_name}.xls"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Cannot find metadata file for {preferred_name}. Tried: {candidates}")


def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported metadata file format: {path}")


def robust_signal_path(file_name: str, signal_dir: str) -> str:
    print(f"[DEBUG] Looking for signal_dir: {signal_dir}")

    if not os.path.isdir(signal_dir):
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")

    direct = os.path.join(signal_dir, f"{file_name}.csv")
    if os.path.exists(direct):
        return direct

    target = normalize_name(file_name)
    candidates = []

    for fn in os.listdir(signal_dir):
        if not fn.lower().endswith(".csv"):
            continue
        if normalize_name(fn) == target:
            candidates.append(os.path.join(signal_dir, fn))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(f"Ambiguous exact-normalized match for {file_name}: {candidates}")

    candidates = []
    for fn in os.listdir(signal_dir):
        if not fn.lower().endswith(".csv"):
            continue
        norm = normalize_name(fn)
        if target in norm or norm in target:
            candidates.append(os.path.join(signal_dir, fn))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(f"Ambiguous substring match for {file_name}: {candidates}")

    raise FileNotFoundError(f"No signal file found for file_name={file_name} in {signal_dir}")


def crop_or_pad_signal(arr: np.ndarray, target_len: int) -> np.ndarray:
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n > target_len:
        start = (n - target_len) // 2
        return arr[start:start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(arr, ((left, right), (0, 0)), mode="constant")


def zscore_per_axis(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    return (arr - mean) / (std + eps)


def transform_target(y: np.ndarray, nominal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return y.copy()
    if mode == "residual":
        return y - nominal
    if mode == "relative":
        return (y - nominal) / (nominal + 1e-8)
    raise ValueError(f"Unknown target mode: {mode}")


def invert_target(y_pred: np.ndarray, nominal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return y_pred.copy()
    if mode == "residual":
        return y_pred + nominal
    if mode == "relative":
        return y_pred * (nominal + 1e-8) + nominal
    raise ValueError(f"Unknown target mode: {mode}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def bin_targets_for_stratification(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        return np.asarray(bins)
    except Exception:
        ranks = pd.Series(y).rank(method="average").values
        bins = np.floor((ranks - 1) / max(1, len(y) / n_bins)).astype(int)
        return bins


def summarize_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    resid = y_pred - y_true
    return {
        "mean_residual": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "median_absolute_error": float(np.median(abs_err)),
        "p90_absolute_error": float(np.quantile(abs_err, 0.90)),
        "p95_absolute_error": float(np.quantile(abs_err, 0.95)),
        "max_absolute_error": float(np.max(abs_err)),
    }


def binned_mae_by_nominal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nominal: np.ndarray,
    n_bins: int = 4
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "nominal": nominal})
    df["bin"] = pd.qcut(df["nominal"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=False).apply(
        lambda x: mean_absolute_error(x["y_true"], x["y_pred"])
    ).reset_index()
    out.columns = ["nominal_bin", "mae"]
    return out


def safe_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p / (np.sum(p) + eps)
    return float(-np.sum(p * np.log(p + eps)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c) or np.isinf(c):
        return 0.0
    return float(c)


def hjorth_params(x: np.ndarray) -> Tuple[float, float, float]:
    dx = np.diff(x, prepend=x[0])
    ddx = np.diff(dx, prepend=dx[0])
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def teager_kaiser_energy(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return y


def fft_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x))
    power = spec ** 2 + 1e-12

    centroid = np.sum(freqs * power) / np.sum(power)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power))
    cumulative = np.cumsum(power) / np.sum(power)
    rolloff_idx = np.searchsorted(cumulative, 0.85)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    peak_idx = np.argmax(spec[1:]) + 1 if len(spec) > 1 else 0
    peak_freq = freqs[peak_idx]
    peak_amp = spec[peak_idx]

    return {
        "spec_centroid": float(centroid),
        "spec_bandwidth": float(bandwidth),
        "spec_rolloff": float(rolloff),
        "spec_entropy": float(safe_entropy(power)),
        "peak_freq": float(peak_freq),
        "peak_amp": float(peak_amp),
    }


def harmonic_band_energy(
    x: np.ndarray,
    fs: float,
    nominal_speed: float,
    n_harmonics: int,
    band_hz: float
) -> Dict[str, float]:
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f0 = nominal_speed / 60.0
    feats = {}
    for k in range(1, n_harmonics + 1):
        center = k * f0
        lo = center - band_hz
        hi = center + band_hz
        mask = (freqs >= lo) & (freqs <= hi)
        feats[f"harm_{k}_energy"] = float(np.sum(spec[mask])) if np.any(mask) else 0.0
    return feats


def extract_signal_features(signal_xyz: np.ndarray, sample_rate: float, nominal_speed: float, cfg: CFG) -> Dict[str, float]:
    feats = {}
    axes = ["X", "Y", "Z"]

    for i, ax in enumerate(axes):
        x = signal_xyz[:, i]
        feats[f"{ax}_mean"] = float(np.mean(x))
        feats[f"{ax}_std"] = float(np.std(x))
        feats[f"{ax}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{ax}_min"] = float(np.min(x))
        feats[f"{ax}_max"] = float(np.max(x))
        feats[f"{ax}_ptp"] = float(np.ptp(x))
        feats[f"{ax}_skew"] = float(skew(x))
        feats[f"{ax}_kurtosis"] = float(kurtosis(x))

        hj_a, hj_m, hj_c = hjorth_params(x)
        feats[f"{ax}_hjorth_activity"] = hj_a
        feats[f"{ax}_hjorth_mobility"] = hj_m
        feats[f"{ax}_hjorth_complexity"] = hj_c

        tke = teager_kaiser_energy(x)
        feats[f"{ax}_tke_mean"] = float(np.mean(tke))
        feats[f"{ax}_tke_std"] = float(np.std(tke))

        for k, v in fft_features(x, sample_rate).items():
            feats[f"{ax}_{k}"] = v

        for k, v in harmonic_band_energy(x, sample_rate, nominal_speed, cfg.n_harmonics, cfg.harmonic_band_hz).items():
            feats[f"{ax}_{k}"] = v

    feats["corr_xy"] = safe_corr(signal_xyz[:, 0], signal_xyz[:, 1])
    feats["corr_xz"] = safe_corr(signal_xyz[:, 0], signal_xyz[:, 2])
    feats["corr_yz"] = safe_corr(signal_xyz[:, 1], signal_xyz[:, 2])

    mag = np.linalg.norm(signal_xyz, axis=1)
    feats["MAG_mean"] = float(np.mean(mag))
    feats["MAG_std"] = float(np.std(mag))
    feats["MAG_rms"] = float(np.sqrt(np.mean(mag ** 2)))
    feats["MAG_skew"] = float(skew(mag))
    feats["MAG_kurtosis"] = float(kurtosis(mag))
    for k, v in fft_features(mag, sample_rate).items():
        feats[f"MAG_{k}"] = v

    feats["sample_rate"] = float(sample_rate)
    feats["nominal_speed"] = float(nominal_speed)
    return feats


if HAS_TORCH:
    class MotorDataset(Dataset):
        def __init__(self, waveforms: np.ndarray, targets: Optional[np.ndarray] = None):
            self.waveforms = waveforms.astype(np.float32)
            self.targets = None if targets is None else targets.astype(np.float32)

        def __len__(self):
            return len(self.waveforms)

        def __getitem__(self, idx):
            x = self.waveforms[idx].transpose(1, 0)  # [L,3] -> [3,L]
            if self.targets is None:
                return torch.tensor(x, dtype=torch.float32)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

    class SmallWaveCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.head(x)
            return x.squeeze(1)

    class HuberLoss(nn.Module):
        def __init__(self, delta: float = 1.0):
            super().__init__()
            self.delta = delta

        def forward(self, pred, target):
            err = pred - target
            abs_err = torch.abs(err)
            quad = torch.clamp(abs_err, max=self.delta)
            lin = abs_err - quad
            return torch.mean(0.5 * quad ** 2 + self.delta * lin)


def train_cnn_regressor(X_train_wave, y_train, X_val_wave, y_val, cfg: CFG) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available.")

    device = cfg.cnn_device
    model = SmallWaveCNN().to(device)
    loss_fn = HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.cnn_lr,
        weight_decay=cfg.cnn_weight_decay
    )

    train_ds = MotorDataset(X_train_wave, y_train)
    val_ds = MotorDataset(X_val_wave, y_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.cnn_batch_size,
        shuffle=True,
        num_workers=cfg.cnn_num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.cnn_batch_size,
        shuffle=False,
        num_workers=cfg.cnn_num_workers
    )

    best_state = None
    best_val = float("inf")
    patience_count = 0

    for _ in range(cfg.cnn_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg.cnn_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit_full_cnn_and_predict(X_train_wave, y_train, X_test_wave, cfg: CFG) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available.")

    device = cfg.cnn_device
    model = SmallWaveCNN().to(device)
    loss_fn = HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.cnn_lr,
        weight_decay=cfg.cnn_weight_decay
    )

    train_ds = MotorDataset(X_train_wave, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.cnn_batch_size,
        shuffle=True,
        num_workers=cfg.cnn_num_workers
    )

    for _ in range(cfg.cnn_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_ds = MotorDataset(X_test_wave, None)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.cnn_batch_size,
        shuffle=False,
        num_workers=cfg.cnn_num_workers
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_catboost(cfg: CFG):
    return CatBoostRegressor(
        iterations=cfg.catboost_iterations,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_lr,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=False,
        random_seed=cfg.random_seed,
    )


def build_xgboost(cfg: CFG):
    return XGBRegressor(
        n_estimators=cfg.xgb_estimators,
        max_depth=cfg.xgb_depth,
        learning_rate=cfg.xgb_lr,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        objective="reg:squarederror",
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def build_random_forest(cfg: CFG):
    return RandomForestRegressor(
        n_estimators=cfg.rf_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )


def build_mlp(cfg: CFG):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=cfg.mlp_hidden,
            max_iter=cfg.mlp_max_iter,
            random_state=cfg.random_seed,
            early_stopping=True,
        )),
    ])


def load_single_signal(file_name: str, signal_dir: str, cfg: CFG) -> np.ndarray:
    path = robust_signal_path(file_name, signal_dir)
    df = pd.read_csv(path)
    required_cols = {"X", "Y", "Z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path} must contain X,Y,Z columns")
    arr = df[["X", "Y", "Z"]].values.astype(np.float32)
    arr = crop_or_pad_signal(arr, cfg.signal_length)
    arr = zscore_per_axis(arr)
    return arr


def build_dataset(meta_df: pd.DataFrame, signal_dir: str, cfg: CFG, with_target: bool = True):
    features = []
    waveforms = []
    file_names = []
    sample_rates = []
    nominal_speeds = []
    targets = []

    for _, row in meta_df.iterrows():
        file_name = str(row["file_name"])
        sample_rate = float(row["sample_rate"])
        nominal_speed = float(row["nominal_speed"])

        signal = load_single_signal(file_name, signal_dir, cfg)
        feat = extract_signal_features(signal, sample_rate, nominal_speed, cfg)

        features.append(feat)
        waveforms.append(signal)
        file_names.append(file_name)
        sample_rates.append(sample_rate)
        nominal_speeds.append(nominal_speed)

        if with_target:
            targets.append(float(row["running_speed"]))

    feat_df = pd.DataFrame(features)
    waveforms = np.stack(waveforms, axis=0)
    nominal_speeds = np.asarray(nominal_speeds, dtype=np.float32)

    out = {
        "file_names": np.asarray(file_names),
        "features_df": feat_df,
        "waveforms": waveforms,
        "sample_rate": np.asarray(sample_rates, dtype=np.float32),
        "nominal_speed": nominal_speeds,
    }
    if with_target:
        out["y_abs"] = np.asarray(targets, dtype=np.float32)
    return out


def fit_nonnegative_ridge(P: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """
    正确版本：
    先对 P 和 y 都中心化，再做 non-negative ridge。
    返回的 intercept 是按原始空间还原后的截距。
    """
    P_mean = P.mean(axis=0, keepdims=True)   # [1, M]
    y_mean = float(y.mean())

    Pc = P - P_mean
    yc = y - y_mean

    n_models = P.shape[1]
    A = np.vstack([Pc, np.sqrt(alpha) * np.eye(n_models)])
    bvec = np.concatenate([yc, np.zeros(n_models)])

    res = lsq_linear(A, bvec, bounds=(0, np.inf), lsmr_tol="auto", verbose=0)
    w = res.x
    intercept = y_mean - (P_mean @ w).item()

    return w, intercept


def interval_to_pretty_label(x) -> str:
    s = str(x)
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 2:
        try:
            a = float(parts[0])
            b = float(parts[1])
            return f"{a:.1f}-{b:.1f}"
        except Exception:
            return s
    return s


def save_fig_multi(fig, out_dir: str, stem: str, cfg: CFG, dpi: int = 300):
    ensure_dir(out_dir)
    if cfg.save_pdf_figures:
        fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    if cfg.save_png_figures:
        fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_fold_metrics_tables(
    fold_metrics: Dict[str, list],
    splits,
    y_true: np.ndarray,
    pred_stack: np.ndarray,
    output_dir: str
):
    stack_fold_metrics = []
    for _, va_idx in splits:
        stack_fold_metrics.append(compute_metrics(y_true[va_idx], pred_stack[va_idx]))

    local_fold_metrics = dict(fold_metrics)
    local_fold_metrics["stack"] = stack_fold_metrics

    rows = []
    for model_name, metrics_list in local_fold_metrics.items():
        for fold_id, m in enumerate(metrics_list, start=1):
            rows.append({
                "model": model_name,
                "fold": fold_id,
                "r2": m["r2"],
                "mae": m["mae"],
                "rmse": m["rmse"],
            })

    fold_df = pd.DataFrame(rows)
    fold_csv = os.path.join(output_dir, "fold_metrics.csv")
    fold_df.to_csv(fold_csv, index=False)

    summary_rows = []
    for model_name, g in fold_df.groupby("model"):
        summary_rows.append({
            "model": model_name,
            "r2_mean": g["r2"].mean(),
            "r2_std": g["r2"].std(ddof=1),
            "mae_mean": g["mae"].mean(),
            "mae_std": g["mae"].std(ddof=1),
            "rmse_mean": g["rmse"].mean(),
            "rmse_std": g["rmse"].std(ddof=1),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="mae_mean", ascending=True)
    summary_csv = os.path.join(output_dir, "main_results_mean_std.csv")
    summary_df.to_csv(summary_csv, index=False)

    return fold_df, summary_df


def export_latex_tables_from_csv(output_dir: str, paper_dir: str):
    ensure_dir(paper_dir)

    meanstd_path = os.path.join(output_dir, "main_results_mean_std.csv")
    error_path = os.path.join(output_dir, "error_summary.csv")
    binned_path = os.path.join(output_dir, "binned_mae_by_nominal.csv")

    if os.path.exists(meanstd_path):
        df = pd.read_csv(meanstd_path).copy()
        df["MAE"] = df.apply(lambda r: f"{r['mae_mean']:.4f} $\\pm$ {r['mae_std']:.4f}", axis=1)
        df["RMSE"] = df.apply(lambda r: f"{r['rmse_mean']:.4f} $\\pm$ {r['rmse_std']:.4f}", axis=1)
        df["$R^2$"] = df.apply(lambda r: f"{r['r2_mean']:.4f} $\\pm$ {r['r2_std']:.4f}", axis=1)
        tex_df = df[["model", "MAE", "RMSE", "$R^2$"]].rename(columns={"model": "Model"})
        tex = tex_df.to_latex(index=False, escape=False)
        with open(os.path.join(paper_dir, "table_main_results.tex"), "w", encoding="utf-8") as f:
            f.write(tex)

    if os.path.exists(error_path):
        df = pd.read_csv(error_path).copy()
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        with open(os.path.join(paper_dir, "table_error_summary.tex"), "w", encoding="utf-8") as f:
            f.write(tex)

    if os.path.exists(binned_path):
        df = pd.read_csv(binned_path).copy()
        df["nominal_bin"] = df["nominal_bin"].apply(interval_to_pretty_label)
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        with open(os.path.join(paper_dir, "table_binned_mae.tex"), "w", encoding="utf-8") as f:
            f.write(tex)


def generate_paper_figures(output_dir: str, paper_dir: str, cfg: CFG):
    ensure_dir(paper_dir)

    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    results_path = os.path.join(output_dir, "main_results.csv")
    meanstd_path = os.path.join(output_dir, "main_results_mean_std.csv")
    binned_path = os.path.join(output_dir, "binned_mae_by_nominal.csv")

    if not os.path.exists(oof_path):
        print("[WARN] oof_predictions.csv not found, skip figure generation.")
        return

    oof_df = pd.read_csv(oof_path)

    # 03_pred_vs_true_ensemble
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = oof_df["running_speed_true"].values
    y = oof_df["pred_stack"].values
    ax.scatter(x, y, s=18, alpha=0.65)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("True running speed")
    ax.set_ylabel("Predicted running speed")
    ax.set_title("Predicted vs. True (Stacked Ensemble)")
    ax.grid(alpha=0.25)
    save_fig_multi(fig, paper_dir, "03_pred_vs_true_ensemble", cfg)

    # 04_residual_hist
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    resid = oof_df["residual_stack"].values
    ax.hist(resid, bins=30, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (prediction - truth)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(alpha=0.20, axis="y")
    save_fig_multi(fig, paper_dir, "04_residual_hist", cfg)

    # 05_binned_mae_by_nominal
    if os.path.exists(binned_path):
        binned_df = pd.read_csv(binned_path).copy()
        binned_df["nominal_bin_pretty"] = binned_df["nominal_bin"].apply(interval_to_pretty_label)

        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.bar(binned_df["nominal_bin_pretty"], binned_df["mae"])
        ax.set_xlabel("Nominal-speed bin")
        ax.set_ylabel("MAE")
        ax.set_title("Binned MAE by Nominal Speed")
        ax.grid(alpha=0.20, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        save_fig_multi(fig, paper_dir, "05_binned_mae_by_nominal", cfg)

    # 06_residual_vs_nominal
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    ax.scatter(oof_df["nominal_speed"], oof_df["residual_stack"], s=16, alpha=0.55)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Nominal speed")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs. Nominal Speed")
    ax.grid(alpha=0.25)
    save_fig_multi(fig, paper_dir, "06_residual_vs_nominal", cfg)

    # 07_metric_compare
    metric_df = None
    if os.path.exists(meanstd_path):
        metric_df = pd.read_csv(meanstd_path).copy()
        metric_df = metric_df.sort_values(by="mae_mean", ascending=True)
        has_std = True
    elif os.path.exists(results_path):
        metric_df = pd.read_csv(results_path).copy()
        metric_df = metric_df.sort_values(by="mae", ascending=True)
        has_std = False

    if metric_df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
        labels = metric_df["model"].tolist()

        if has_std:
            axes[0].bar(labels, metric_df["mae_mean"], yerr=metric_df["mae_std"], capsize=3)
            axes[1].bar(labels, metric_df["rmse_mean"], yerr=metric_df["rmse_std"], capsize=3)
            axes[2].bar(labels, metric_df["r2_mean"], yerr=metric_df["r2_std"], capsize=3)
        else:
            axes[0].bar(labels, metric_df["mae"])
            axes[1].bar(labels, metric_df["rmse"])
            axes[2].bar(labels, metric_df["r2"])

        axes[0].set_title("MAE")
        axes[1].set_title("RMSE")
        axes[2].set_title("$R^2$")

        for ax in axes:
            ax.grid(alpha=0.20, axis="y")
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        save_fig_multi(fig, paper_dir, "07_metric_compare", cfg)


def train_and_evaluate(cfg: CFG):
    seed_everything(cfg.random_seed)
    ensure_dir(cfg.output_dir)

    if cfg.train_meta_path is None:
        cfg.train_meta_path = auto_detect_metadata("train")
    if cfg.test_meta_path is None:
        try:
            cfg.test_meta_path = auto_detect_metadata("test")
        except Exception:
            cfg.test_meta_path = None

    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"Train metadata: {cfg.train_meta_path}")
    print(f"Test metadata : {cfg.test_meta_path}")
    print(f"Train dir     : {cfg.train_signal_dir}")
    print(f"Test dir      : {cfg.test_signal_dir}")

    train_meta = read_table(cfg.train_meta_path)
    required_train = {"file_name", "sample_rate", "nominal_speed", "running_speed"}
    if not required_train.issubset(train_meta.columns):
        raise ValueError(f"Train metadata must contain columns: {required_train}")

    print("\nBuilding train dataset...")
    train_data = build_dataset(train_meta, cfg.train_signal_dir, cfg, with_target=True)

    X_feat = train_data["features_df"]
    X_wave = train_data["waveforms"]
    nominal = train_data["nominal_speed"]
    y_abs = train_data["y_abs"]
    y_train = transform_target(y_abs, nominal, cfg.target_mode)

    if cfg.use_stratified_cv:
        y_bins = bin_targets_for_stratification(y_abs, cfg.n_target_bins)
        splitter = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_seed
        )
        splits = list(splitter.split(X_feat, y_bins))
    else:
        splitter = KFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_seed
        )
        splits = list(splitter.split(X_feat))

    model_names: List[str] = []
    oof_dict: Dict[str, np.ndarray] = {}
    fold_metrics: Dict[str, list] = {}

    def register_model(name):
        model_names.append(name)
        oof_dict[name] = np.zeros(len(X_feat), dtype=np.float32)
        fold_metrics[name] = []

    if cfg.use_random_forest:
        register_model("rf")
    if cfg.use_mlp:
        register_model("mlp")
    if cfg.use_catboost and HAS_CATBOOST:
        register_model("catboost")
    if cfg.use_xgboost and HAS_XGBOOST:
        register_model("xgboost")
    if cfg.use_cnn and HAS_TORCH:
        register_model("cnn")

    if cfg.use_catboost and not HAS_CATBOOST:
        print("CatBoost not installed, skipping.")
    if cfg.use_xgboost and not HAS_XGBOOST:
        print("XGBoost not installed, skipping.")
    if cfg.use_cnn and not HAS_TORCH:
        print("PyTorch not installed, skipping CNN.")

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n========== Fold {fold + 1}/{cfg.n_splits} ==========")

        Xtr_feat = X_feat.iloc[tr_idx].reset_index(drop=True)
        Xva_feat = X_feat.iloc[va_idx].reset_index(drop=True)
        Xtr_wave = X_wave[tr_idx]
        Xva_wave = X_wave[va_idx]

        nominal_va = nominal[va_idx]
        ytr = y_train[tr_idx]
        yva = y_train[va_idx]
        yva_abs = y_abs[va_idx]

        if "rf" in oof_dict:
            model = build_random_forest(cfg)
            model.fit(Xtr_feat, ytr)
            pred_abs = invert_target(model.predict(Xva_feat), nominal_va, cfg.target_mode)
            oof_dict["rf"][va_idx] = pred_abs
            fold_metrics["rf"].append(compute_metrics(yva_abs, pred_abs))

        if "mlp" in oof_dict:
            model = build_mlp(cfg)
            model.fit(Xtr_feat, ytr)
            pred_abs = invert_target(model.predict(Xva_feat), nominal_va, cfg.target_mode)
            oof_dict["mlp"][va_idx] = pred_abs
            fold_metrics["mlp"].append(compute_metrics(yva_abs, pred_abs))

        if "catboost" in oof_dict:
            model = build_catboost(cfg)
            model.fit(Xtr_feat, ytr)
            pred_abs = invert_target(model.predict(Xva_feat), nominal_va, cfg.target_mode)
            oof_dict["catboost"][va_idx] = pred_abs
            fold_metrics["catboost"].append(compute_metrics(yva_abs, pred_abs))

        if "xgboost" in oof_dict:
            model = build_xgboost(cfg)
            model.fit(Xtr_feat, ytr)
            pred_abs = invert_target(model.predict(Xva_feat), nominal_va, cfg.target_mode)
            oof_dict["xgboost"][va_idx] = pred_abs
            fold_metrics["xgboost"].append(compute_metrics(yva_abs, pred_abs))

        if "cnn" in oof_dict:
            pred = train_cnn_regressor(Xtr_wave, ytr, Xva_wave, yva, cfg)
            pred_abs = invert_target(pred, nominal_va, cfg.target_mode)
            oof_dict["cnn"][va_idx] = pred_abs
            fold_metrics["cnn"].append(compute_metrics(yva_abs, pred_abs))

    main_metrics = {}
    for name in model_names:
        m = compute_metrics(y_abs, oof_dict[name])
        main_metrics[name] = m
        print(f"{name:10s} | R2={m['r2']:.6f} | MAE={m['mae']:.6f} | RMSE={m['rmse']:.6f}")

    # 只用配置中允许且当前已实际存在的模型做 stack
    stack_names = [n for n in cfg.stack_candidate_models if n in oof_dict]
    if len(stack_names) == 0:
        raise RuntimeError("No valid base models available for stacking. Check stack_candidate_models.")

    print(f"\nStack base models: {stack_names}")

    P = np.column_stack([oof_dict[name] for name in stack_names])
    stack_w, stack_b = fit_nonnegative_ridge(P, y_abs, alpha=cfg.stack_alpha)
    pred_stack = P @ stack_w + stack_b
    stack_metrics = compute_metrics(y_abs, pred_stack)
    print(f"{'stack':10s} | R2={stack_metrics['r2']:.6f} | MAE={stack_metrics['mae']:.6f} | RMSE={stack_metrics['rmse']:.6f}")

    fold_df, meanstd_df = export_fold_metrics_tables(
        fold_metrics=fold_metrics,
        splits=splits,
        y_true=y_abs,
        pred_stack=pred_stack,
        output_dir=cfg.output_dir,
    )

    oof_df = pd.DataFrame({
        "file_name": train_data["file_names"],
        "sample_rate": train_data["sample_rate"],
        "nominal_speed": nominal,
        "running_speed_true": y_abs,
    })
    for name in model_names:
        oof_df[f"pred_{name}"] = oof_dict[name]
    oof_df["pred_stack"] = pred_stack
    oof_df["residual_stack"] = pred_stack - y_abs
    if cfg.save_oof_csv:
        oof_df.to_csv(os.path.join(cfg.output_dir, "oof_predictions.csv"), index=False)

    rows = [{"model": name, **main_metrics[name]} for name in model_names]
    rows.append({"model": "stack", **stack_metrics})
    results_df = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
    results_df.to_csv(os.path.join(cfg.output_dir, "main_results.csv"), index=False)

    err_summary = summarize_errors(y_abs, pred_stack)
    pd.DataFrame([err_summary]).to_csv(os.path.join(cfg.output_dir, "error_summary.csv"), index=False)

    binned_df = binned_mae_by_nominal(y_abs, pred_stack, nominal, n_bins=4)
    binned_df.to_csv(os.path.join(cfg.output_dir, "binned_mae_by_nominal.csv"), index=False)

    if cfg.save_metrics_json:
        payload = {
            "config": asdict(cfg),
            "main_metrics": main_metrics,
            "stack_metrics": stack_metrics,
            "stack_base_models": stack_names,
            "stack_weights": {name: float(w) for name, w in zip(stack_names, stack_w)},
            "stack_intercept": float(stack_b),
            "error_summary": err_summary,
        }
        with open(os.path.join(cfg.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    test_pred_path = None
    if cfg.test_meta_path is not None and os.path.exists(cfg.test_signal_dir):
        print("\nBuilding test dataset...")
        test_meta = read_table(cfg.test_meta_path)
        required_test = {"file_name", "sample_rate", "nominal_speed"}
        if not required_test.issubset(test_meta.columns):
            raise ValueError(f"Test metadata must contain columns: {required_test}")

        test_data = build_dataset(test_meta, cfg.test_signal_dir, cfg, with_target=False)
        X_test_feat = test_data["features_df"]
        X_test_wave = test_data["waveforms"]
        nominal_test = test_data["nominal_speed"]

        full_preds: Dict[str, np.ndarray] = {}

        if cfg.use_random_forest and "rf" in model_names:
            model = build_random_forest(cfg)
            model.fit(X_feat, y_train)
            full_preds["rf"] = invert_target(model.predict(X_test_feat), nominal_test, cfg.target_mode)

        if cfg.use_mlp and "mlp" in model_names:
            model = build_mlp(cfg)
            model.fit(X_feat, y_train)
            full_preds["mlp"] = invert_target(model.predict(X_test_feat), nominal_test, cfg.target_mode)

        if cfg.use_catboost and HAS_CATBOOST and "catboost" in model_names:
            model = build_catboost(cfg)
            model.fit(X_feat, y_train)
            full_preds["catboost"] = invert_target(model.predict(X_test_feat), nominal_test, cfg.target_mode)

        if cfg.use_xgboost and HAS_XGBOOST and "xgboost" in model_names:
            model = build_xgboost(cfg)
            model.fit(X_feat, y_train)
            full_preds["xgboost"] = invert_target(model.predict(X_test_feat), nominal_test, cfg.target_mode)

        if cfg.use_cnn and HAS_TORCH and "cnn" in model_names:
            full_preds["cnn"] = invert_target(
                fit_full_cnn_and_predict(X_wave, y_train, X_test_wave, cfg),
                nominal_test,
                cfg.target_mode,
            )

        if not all(n in full_preds for n in stack_names):
            missing = [n for n in stack_names if n not in full_preds]
            raise RuntimeError(f"Missing test-time predictions for stack base models: {missing}")

        P_test = np.column_stack([full_preds[n] for n in stack_names])
        pred_stack_test = P_test @ stack_w + stack_b

        test_out = test_meta.copy()
        test_out["running_speed"] = pred_stack_test
        for n in sorted(full_preds.keys()):
            test_out[f"pred_{n}"] = full_preds[n]

        test_pred_path = os.path.join(cfg.output_dir, "final_predict_test.csv")
        if cfg.save_test_csv:
            test_out.to_csv(test_pred_path, index=False)

    return {
        "main_results_df": results_df,
        "error_summary": err_summary,
        "binned_df": binned_df,
        "stack_metrics": stack_metrics,
        "stack_names": stack_names,
        "fold_results_df": fold_df,
        "meanstd_df": meanstd_df,
        "test_pred_path": test_pred_path,
    }


def print_paper_ready_numbers(output_dir: str):
    results_path = os.path.join(output_dir, "main_results.csv")
    error_path = os.path.join(output_dir, "error_summary.csv")
    meanstd_path = os.path.join(output_dir, "main_results_mean_std.csv")
    metrics_json_path = os.path.join(output_dir, "metrics_summary.json")

    if os.path.exists(results_path):
        print("\n========== Main Results ==========")
        print(pd.read_csv(results_path).to_string(index=False))

    if os.path.exists(meanstd_path):
        print("\n========== Main Results (mean ± std source) ==========")
        print(pd.read_csv(meanstd_path).to_string(index=False))

    if os.path.exists(error_path):
        print("\n========== Error Summary ==========")
        print(pd.read_csv(error_path).to_string(index=False))

    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print("\n========== Stacking ==========")
        print("Base models :", payload.get("stack_base_models"))
        print("Weights     :", payload.get("stack_weights"))
        print("Intercept   :", payload.get("stack_intercept"))


if __name__ == "__main__":
    start = time.time()
    ensure_dir(CFG_OBJ.output_dir)

    with open(os.path.join(CFG_OBJ.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG_OBJ), f, indent=2, ensure_ascii=False)

    out = train_and_evaluate(CFG_OBJ)
    print_paper_ready_numbers(CFG_OBJ.output_dir)

    if CFG_OBJ.save_paper_assets:
        generate_paper_figures(CFG_OBJ.output_dir, CFG_OBJ.paper_dir, CFG_OBJ)
        if CFG_OBJ.export_latex_tables:
            export_latex_tables_from_csv(CFG_OBJ.output_dir, CFG_OBJ.paper_dir)
        print(f"\nPaper assets exported to: {CFG_OBJ.paper_dir}")

    print(f"\nDone in {time.time() - start:.2f} seconds.")
    if out["test_pred_path"] is not None:
        print(f"Test prediction saved to: {out['test_pred_path']}")