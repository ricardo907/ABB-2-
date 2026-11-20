#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, math, json, time, shutil, warnings, argparse, subprocess, random
from functools import partial
from typing import List, Dict, Optional, Tuple

print("[BOOT] main path =", os.path.abspath(__file__), flush=True)

# Keep the allocator setting compatible with older PyTorch releases.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import numpy as np
import pandas as pd
import torch

# 基础工具
def log(msg: str):
    print(msg, flush=True)
    try:
        os.makedirs("runtime", exist_ok=True)
        with open("runtime/log.txt", "a", encoding="utf-8") as f:
            f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg + "\n")
    except Exception:
        pass

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def has_ffmpeg() -> bool:
    return which("ffmpeg") is not None

def run(cmd: List[str]):
    log("[cmd] " + " ".join(cmd))
    subprocess.check_call(cmd)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# 显存自适应
def get_gpu_mem_gb() -> float:
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return float(p.total_memory) / (1024**3)
    except Exception:
        pass
    return 0.0

def auto_hparams(mem_gb: float):
    """返回 backbone/batch/img_size/seq_len/accum/workers"""
    if mem_gb >= 20:
        return dict(backbone="convnext_tiny", batch=24, img_size=288, seq_len=12, accum=1, workers=4)
    if mem_gb >= 12:
        return dict(backbone="convnext_tiny", batch=16, img_size=288, seq_len=12, accum=1, workers=4)
    if mem_gb >= 8:   # RTX 4060 8GB
        return dict(backbone="resnet18", batch=6, img_size=224, seq_len=10, accum=1, workers=2)
    if mem_gb >= 6:
        return dict(backbone="resnet18", batch=4, img_size=224, seq_len=8,  accum=1, workers=2)
    return dict(backbone="resnet18", batch=2, img_size=192, seq_len=8, accum=1, workers=0)

# ------------------------- 自动发现 6 个官方文件 -------------------------
def discover_files_recursive(root: str) -> Dict[str, Optional[str]]:
    def pick(patterns):
        for p in patterns:
            m = sorted(glob.glob(os.path.join(root, "**", p), recursive=True))
            if m: return m[0]
        return None
    return dict(
        pre_template = pick(["*Pre*Template*.xlsx"]),
        pre_video    = pick(["*Pre*noaudio*.MOV","*Pre*.MOV","*Pre*noaudio*.MP4","*Pre*.MP4"]),
        train_v1     = pick(["*Train*Angle1*noaudio*.MOV","*Train*Angle1*.MOV","*Train*Angle1*.MP4"]),
        train_v2     = pick(["*Train*Angle2*noaudio*.MOV","*Train*Angle2*.MOV","*Train*Angle2*.MP4"]),
        train_x1     = pick(["*Train*Angle1*.xlsx"]),
        train_x2     = pick(["*Train*Angle2*.xlsx"]),
    )

def require_all(files: Dict[str, Optional[str]]):
    miss = [k for k,v in files.items() if not v]
    if miss:
        raise FileNotFoundError("Missing required files: " + ", ".join(miss))

# ------------------------- 抽帧（启动前清空旧帧） -------------------------
def _resampled_frames(cap, fps: float):
    """Yield frames on a zero-based fixed-FPS grid from a constant-FPS video."""
    import cv2
    if fps <= 0:
        raise ValueError("fps must be positive")
    native = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(native) or native <= 0:
        raise RuntimeError("Cannot determine video FPS")
    src_idx = out_idx = 0
    last_frame = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # Each output timestamp takes the nearest source frame.
        boundary = (src_idx + 0.5) / native
        while out_idx / fps < boundary - 1e-9:
            yield fr
            out_idx += 1
        last_frame = fr
        src_idx += 1
    if last_frame is not None:
        duration = src_idx / native
        while out_idx / fps < duration - 1e-9:
            yield last_frame
            out_idx += 1

def extract_frames(video: str, out_dir: str, fps: float = 30.0) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    ensure_dir(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(frames_dir)
    # 清空旧帧，避免残留
    for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
        try: os.remove(f)
        except: pass

    if has_ffmpeg():
        out_pat = os.path.join(frames_dir, "%06d.jpg")
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",video,
             "-vf",f"setpts=PTS-STARTPTS,fps=fps={fps}:start_time=0:round=near",
             "-vsync","0","-start_number","0",out_pat])
    else:
        import cv2
        cap = cv2.VideoCapture(video)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video}")
        try:
            for out_idx, fr in enumerate(_resampled_frames(cap, fps)):
                path = os.path.join(frames_dir, f"{out_idx:06d}.jpg")
                if not cv2.imwrite(path, fr):
                    raise RuntimeError(f"Cannot write frame: {path}")
                if (out_idx + 1) % 500 == 0:
                    log(f"[extract] {video} -> {out_idx + 1} frames...")
        finally:
            cap.release()

    n = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if n < 10: raise RuntimeError("Too few frames; check codec/ffmpeg.")
    log(f"[extract] {video} -> {n} frames at {frames_dir}")
    return n

# ------------------------- Excel -> angles.csv -------------------------
def _guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower: return lower[name.lower()]
    for c in df.columns:
        key = c.lower().replace(" ","").replace("_","")
        for name in candidates:
            nkey = name.lower().replace(" ","").replace("_","")
            if key == nkey: return c
    return None

def convert_excel_to_angles(excel_path: str, out_csv: str, fps: float = 30.0) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    df = pd.read_excel(excel_path, sheet_name=0).dropna(axis=1, how="all")
    strict = ["Time(s)"] + [f"position(rad)_axis_{i}" for i in range(1,7)]
    if all(c in df.columns for c in strict):
        t_col = strict[0]
        angle_cols = strict[1:]
    else:
        t_col = _guess_col(df, ["t","time","time(s)","timestamp","sec","seconds"])
        angle_cols = []
        for i in range(1,7):
            col = _guess_col(df, [f"position(rad)_axis_{i}", f"axis{i}", f"joint{i}", f"j{i}"])
            if col is None:
                for c in df.columns:
                    key = c.lower().replace(" ","")
                    if key.endswith(f"axis{i}") or key.endswith(f"joint{i}") or key.endswith(f"j{i}"):
                        col = c; break
            angle_cols.append(col)
        if any(c is None for c in angle_cols):
            raise ValueError("Excel 列缺失（axis/joint/jX 任一都可）")
    if t_col is None:
        raise ValueError("Excel 缺少时间列，无法与视频帧可靠同步")

    # Excel may sample much faster than video. Interpolate its actual time
    # series onto the time grid of the frames that extraction produced.
    source = pd.DataFrame({"t": pd.to_numeric(df[t_col], errors="coerce")})
    for i, col in enumerate(angle_cols, start=1):
        source[f"j{i}"] = pd.to_numeric(df[col], errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan).dropna()
    if source.empty:
        raise ValueError(f"No valid time/angle rows in {excel_path}")
    source = source.groupby("t", as_index=False).mean().sort_values("t")
    source["t"] -= source["t"].iloc[0]

    frames_dir = os.path.join(os.path.dirname(out_csv), "frames")
    frame_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if frame_files:
        try:
            frame_ids = np.array(sorted(int(os.path.splitext(os.path.basename(p))[0])
                                        for p in frame_files), dtype=int)
        except ValueError as exc:
            raise ValueError(f"Frame filenames must be numeric in {frames_dir}") from exc
        if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
            raise ValueError(f"Frames must be numbered continuously from 0 in {frames_dir}")
    else:
        # Keep the standalone excel2csv command usable without extracted frames.
        frame_ids = np.arange(int(np.floor(source["t"].iloc[-1] * fps + 1e-9)) + 1)
        log(f"[excel] no extracted frames at {frames_dir}; using Excel duration")

    frame_times = frame_ids / float(fps)
    source_times = source["t"].to_numpy()
    sample_steps = np.diff(source_times)
    tolerance = min(0.5 / fps, 2 * float(np.median(sample_steps))) if len(sample_steps) else 0.0
    if len(frame_times) and frame_times[-1] > source_times[-1] + tolerance + 1e-9:
        raise ValueError(
            f"Video frames extend past Excel labels by {frame_times[-1] - source_times[-1]:.3f}s "
            f"(allowed {tolerance:.3f}s): {excel_path}"
        )
    out = pd.DataFrame({"frame": frame_ids, "t": frame_times})
    for i in range(1, 7):
        out[f"j{i}"] = np.interp(frame_times, source_times, source[f"j{i}"].to_numpy())
    ensure_dir(os.path.dirname(out_csv) or "."); out.to_csv(out_csv, index=False)
    log(f"[excel] {excel_path} -> {out_csv} rows={len(out)}")
    return len(out)

def assert_not_all_zero(csv_path: str):
    df = pd.read_csv(csv_path)
    if (df[["j1","j2","j3","j4","j5","j6"]].abs().sum().sum() < 1e-9):
        raise RuntimeError(f"{csv_path} 全 0 ！Excel 转换失败或列名不匹配。")

# ------------------------- ROI 自动裁剪 -------------------------
def find_roi(frames_dir: str, sample=120, margin=24) -> Tuple[int,int,int,int]:
    import cv2
    paths = sorted([os.path.join(frames_dir,f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not paths: return (0,0,0,0)
    step = max(1, len(paths)//sample)
    backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    mask = None
    for i in range(0, len(paths), step):
        img = cv2.imread(paths[i], cv2.IMREAD_COLOR)
        if img is None: continue
        fg = backsub.apply(img); fg = cv2.medianBlur(fg, 5)
        mask = fg if mask is None else cv2.bitwise_or(mask, fg)
    if mask is None: return (0,0,0,0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return (0,0,0,0)
    x,y,w,h = cv2.boundingRect(np.vstack(contours))
    H,W = cv2.imread(paths[0], cv2.IMREAD_COLOR).shape[:2]
    x0 = max(0, x - margin); y0 = max(0, y - margin)
    x1 = min(W, x+w+margin); y1 = min(H, y+h+margin)
    log(f"[roi] {(x0,y0,x1,y1)} on {W}x{H}")
    return (x0,y0,x1,y1)

# ------------------------- 预生成迷你帧缓存（ROI+resize 到 img_size） -------------------------
def prepare_miniframes(train_root: str, roi_map: Dict[str, Tuple[int,int,int,int]], img_size: int):
    import cv2
    for sub in sorted(os.listdir(train_root)):
        vdir = os.path.join(train_root, sub)
        fdir = os.path.join(vdir, "frames")
        if not os.path.isdir(fdir): 
            continue
        mdir = os.path.join(vdir, "mini")
        ensure_dir(mdir)
        roi = roi_map.get(sub, None)
        done = 0
        for fn in sorted(os.listdir(fdir)):
            if not fn.endswith(".jpg"): 
                continue
            src = os.path.join(fdir, fn)
            dst = os.path.join(mdir, fn)
            if os.path.exists(dst): 
                done += 1; 
                continue
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None: 
                continue
            if roi and roi[2] > roi[0] and roi[3] > roi[1]:
                x0,y0,x1,y1 = roi
                img = img[y0:y1, x0:x1]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            done += 1
            if done % 5000 == 0:
                log(f"[mini] {sub}: {done} frames...")
        log(f"[mini] {sub}: ready -> {mdir}")

# ------------------------- 模型与数据 -------------------------
def import_torch_bundle():
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from tqdm import tqdm
    import cv2
    return torch, nn, F, Dataset, DataLoader, models, tqdm, cv2

class VideoAnglesDataset(torch.utils.data.Dataset):
    """顶层 Dataset，保证每个样本返回固定 (seq_len, 6) 标签；支持 mini 缓存与滑窗步长"""
    def __init__(self, root, seq_len=12, img_size=288, roi_map=None, stride=1, prefer_mini=True,
                 augment=True):
        self.seq_len, self.img_size, self.stride = seq_len, img_size, max(1, int(stride))
        self.augment = augment
        self.samples = []  # (name, paths, frame-aligned labels)

        if os.path.isdir(root):
            for sub in sorted(os.listdir(root)):
                vdir = os.path.join(root, sub)
                fdir = os.path.join(vdir, "frames")
                mdir = os.path.join(vdir, "mini")
                base = mdir if prefer_mini and os.path.isdir(mdir) else fdir
                csvp = os.path.join(vdir, "angles.csv")
                if os.path.isdir(base) and os.path.isfile(csvp):
                    paths = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith(".jpg")])
                    if len(paths) < seq_len:
                        continue
                    df = pd.read_csv(csvp)

                    if "frame" not in df.columns:
                        raise RuntimeError(f"{csvp} 缺少 frame 列")
                    df["frame"] = df["frame"].astype(int)
                    for k in [f"j{i}" for i in range(1,7)]:
                        if k not in df.columns:
                            raise RuntimeError(f"{csvp} 缺少列 {k}")
                        df[k] = df[k].astype(np.float32)
                    df = df.groupby("frame")[["j1","j2","j3","j4","j5","j6"]].mean().sort_index()
                    frame_ids = np.array([int(os.path.splitext(os.path.basename(p))[0]) for p in paths])
                    if len(df):
                        label_ids = np.searchsorted(df.index.to_numpy(), frame_ids, side="right") - 1
                        labels = df.to_numpy(dtype=np.float32)[np.maximum(label_ids, 0)]
                    else:
                        raise RuntimeError(f"No angle labels in {csvp}")

                    self.samples.append((sub, paths, labels))

        # 滑窗索引（带步长）
        self.index = []
        for vid_idx, (_, paths, _) in enumerate(self.samples):
            n = len(paths)
            for s in range(0, n - seq_len + 1, self.stride):
                self.index.append((vid_idx, s))
        self.roi_map = roi_map or {}

    def _read(self, path, roi=None):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"读取帧失败: {path}")
        # Mini frames already contain the ROI. Apply it to full frames as well
        # so the default training path matches inference preprocessing.
        if roi and os.path.basename(os.path.dirname(path)) != "mini":
            x0, y0, x1, y1 = roi
            h, w = img.shape[:2]
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
            if x1 > x0 and y1 > y0:
                img = img[y0:y1, x0:x1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (img.shape[0], img.shape[1]) != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img

    def _augment(self, img):
        import cv2
        if random.random() < 0.5:
            a = 0.8 + 0.4*random.random(); b = int((random.random()-0.5)*20)
            img = np.clip(img*a + b, 0, 255).astype(np.uint8)
        if random.random() < 0.4:
            g = 0.9 + 0.4*random.random()
            img = np.clip(255.0*(img/255.0)**g, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3,3), 0)
        if random.random() < 0.2:
            H,W,_ = img.shape
            h = random.randint(H//10, H//5); w = random.randint(W//10, W//5)
            y = random.randint(0, H-h); x = random.randint(0, W-w)
            img[y:y+h, x:x+w] = random.randint(0,255)
        return img

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import torch as _torch

        vid_idx, s = self.index[i]
        name, paths, labels = self.samples[vid_idx]

        # 固定长度的帧序列
        seq_paths = paths[s:s+self.seq_len]
        frames = []
        for p in seq_paths:
            img = self._read(p, self.roi_map.get(name))
            if self.augment:
                img = self._augment(img)
            frames.append((img.astype(np.float32)/255.0).transpose(2,0,1))
        x = _torch.from_numpy(np.stack(frames, 0))  # (T, C, H, W)
        y = _torch.from_numpy(labels[s:s+self.seq_len].copy())  # (T, 6)

        # 最终形状断言
        if x.shape[0] != self.seq_len or y.shape != (self.seq_len, 6):
            fids = [os.path.splitext(os.path.basename(p))[0] for p in seq_paths]
            raise RuntimeError(f"shape mismatch: x{tuple(x.shape)} y{tuple(y.shape)} "
                               f"video={name} start={s} fids={fids[:3]}...{fids[-3:]}")

        return x, y

# ---- 顶层函数 collate（可序列化），统一裁剪/补齐到 seq_len ----
def pad_trim_collate(batch, seq_len: int):
    """
    batch: [(x(T,C,H,W), y(T,6)), ...]
    把每个样本的时间维统一成 seq_len：
      - 短 -> 用最后一帧/最后一行重复填充
      - 长 -> 截断
    """
    import torch as _t
    xs, ys = zip(*batch)
    T = int(seq_len)
    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        if x.shape[0] == 0:
            x = _t.zeros((1,) + tuple(x.shape[1:]), dtype=x.dtype)
        if x.shape[0] < T:
            x = _t.cat([x, x[-1:].repeat(T - x.shape[0], 1, 1, 1)], dim=0)
        elif x.shape[0] > T:
            x = x[:T]

        if y.shape[0] == 0:
            y = _t.zeros((1, 6), dtype=_t.float32)
        if y.shape[0] < T:
            y = _t.cat([y, y[-1:].repeat(T - y.shape[0], 1)], dim=0)
        elif y.shape[0] > T:
            y = y[:T]

        x_out.append(x); y_out.append(y)
    return _t.stack(x_out, 0), _t.stack(y_out, 0)

# ------------------------- 模型（含预训练权重） -------------------------
def build_model(backbone="convnext_tiny", d_model=512, nhead=8, layers=4, dropout=0.1, use_sincos=True, pretrained=True):
    torch, nn, F, Dataset, DataLoader, models, tqdm, cv2 = import_torch_bundle()

    # 兼容不同 torchvision 版本
    def _resnet18(pretrained_flag):
        if not pretrained_flag:
            return models.resnet18(weights=None)
        try:
            from torchvision.models import ResNet18_Weights
            return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                return models.resnet18(weights="IMAGENET1K_V1")
            except Exception:
                return models.resnet18(pretrained=True)

    def _convnext_tiny(pretrained_flag):
        if not pretrained_flag:
            return models.convnext_tiny(weights=None)
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights
            return models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                return models.convnext_tiny(weights="IMAGENET1K_V1")
            except Exception:
                return models.convnext_tiny(weights=None)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            if backbone == "convnext_tiny":
                m = _convnext_tiny(pretrained)
                feat = m.classifier[2].in_features
                m.classifier = nn.Identity()
                self.enc = m; self.out_dim = feat
            elif backbone == "resnet18":
                m = _resnet18(pretrained)
                feat = m.fc.in_features
                m.fc = nn.Identity()
                self.enc = m; self.out_dim = feat
            else:
                raise ValueError("backbone must be convnext_tiny or resnet18")
        def forward(self, x): return self.enc(x)

    class Temporal(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model) if d_in!=d_model else nn.Identity()
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model*4, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, z):
            z = self.proj(z); z = self.enc(z); return self.norm(z)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_sincos = use_sincos
            out = 12 if self.use_sincos else 6
            self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out))
        def forward(self, z):
            y = self.mlp(z)
            if self.use_sincos:
                B,T,_ = y.shape
                y = y.view(B,T,6,2)
                y = torch.nn.functional.normalize(y, dim=-1)
                y = y.view(B,T,12)
            return y

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.temporal = Temporal(self.backbone.out_dim)
            self.head = Head()
            self.use_sincos = use_sincos
        def forward(self, x):
            B,T,C,H,W = x.shape
            x = x.view(B*T,C,H,W)
            f = self.backbone(x)
            f = f.view(B,T,-1)
            z = self.temporal(f)
            y = self.head(z)
            return y

    return Regressor()

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------- Dataloaders（并行 + 预取 + pin_memory） -------------------------
def build_dataloaders(train_root: str, img_size=288, seq_len=12, batch=16, workers=2, roi_map=None,
                      stride=1, prefetch=4, pin_memory=True):
    _, _, _, _, DataLoader, _, _, _ = import_torch_bundle()
    ds = VideoAnglesDataset(train_root, seq_len=seq_len, img_size=img_size,
                            roi_map=roi_map or {}, stride=stride, augment=True)
    if len(ds) == 0: raise RuntimeError("Empty dataset. Check frames and angles.csv.")
    # With three or more videos, hold out one complete video. With the two
    # Angle1/Angle2 videos, use separate temporal splits so both motions train.
    # The boundary excludes windows that would share frames across splits.
    if len(ds.samples) >= 3:
        val_vid = len(ds.samples) - 1
        tr_ids = [i for i, (vid, _) in enumerate(ds.index) if vid != val_vid]
        va_ids = [i for i, (vid, _) in enumerate(ds.index) if vid == val_vid]
    else:
        tr_ids, va_ids = [], []
        for vid in range(len(ds.samples)):
            starts = sorted(s for v, s in ds.index if v == vid)
            n_val = max(1, math.ceil(len(starts) * 0.2))
            val_start = starts[-n_val]
            tr_ids.extend(i for i, (v, s) in enumerate(ds.index)
                          if v == vid and s + seq_len <= val_start)
            va_ids.extend(i for i, (v, s) in enumerate(ds.index)
                          if v == vid and s >= val_start)
    if not tr_ids or not va_ids:
        raise RuntimeError("Need more non-overlapping frames or another video for validation")

    val_ds = VideoAnglesDataset(train_root, seq_len=seq_len, img_size=img_size,
                                roi_map=roi_map or {}, stride=stride, augment=False)
    if val_ds.index != ds.index:
        raise RuntimeError("Training and validation frame indices differ")

    use_workers = int(workers) if workers is not None else 0
    extra = {}
    if use_workers > 0:
        extra["prefetch_factor"] = int(prefetch)

    from torch.utils.data import Subset
    cf = partial(pad_trim_collate, seq_len=seq_len)

    train_ld = DataLoader(Subset(ds, tr_ids), batch_size=batch, shuffle=True,
                          num_workers=use_workers, drop_last=False,
                          pin_memory=bool(pin_memory), persistent_workers=False,
                          collate_fn=cf, **extra)
    val_ld   = DataLoader(Subset(val_ds, va_ids), batch_size=batch, shuffle=False,
                          num_workers=use_workers, drop_last=False,
                          pin_memory=bool(pin_memory), persistent_workers=False,
                          collate_fn=cf, **extra)
    try:
        log(f"[dataloader] workers={use_workers} pin_mem={bool(pin_memory)} prefetch={extra.get('prefetch_factor', 0)}")
        log(f"[split] train={len(tr_ids)} val={len(va_ids)} videos={len(ds.samples)}")
        log(f"[collate] train={train_ld.collate_fn}  val={val_ld.collate_fn}")
    except Exception:
        pass
    return train_ld, val_ld

# ------------------------- Savitzky–Golay 平滑 -------------------------
def savgol_coeffs(window: int, poly: int):
    if window % 2 == 0 or window < 3: raise ValueError("window must be odd >=3")
    if poly >= window: raise ValueError("poly must be < window")
    half = window//2
    x = np.arange(-half, half+1, dtype=np.float64)[:,None]
    A = np.concatenate([x**k for k in range(0, poly+1)], axis=1)
    pinv = np.linalg.pinv(A)
    return pinv[0]

def savgol_filter_1d(y: np.ndarray, window: int, poly: int):
    c = savgol_coeffs(window, poly)
    half = window//2
    ypad = np.pad(y, ((half, half), (0,0)), mode="reflect")
    out = np.zeros_like(ypad, dtype=np.float64)
    for i in range(half, len(ypad)-half):
        seg = ypad[i-half:i+half+1]
        out[i] = (seg * c[:,None]).sum(axis=0)
    return out[half:-half].astype(np.float32)

# ------------------------- 训练（AMP / 梯度累积 / EMA / Cosine + 早停） -------------------------
def cosine_scheduler(optimizer, total_epochs, warmup=3):
    def lr_lambda(ep):
        if ep < warmup: return float(ep+1)/max(1,warmup)
        t = (ep-warmup)/max(1,total_epochs-warmup)
        return 0.5*(1+math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EMA:
    """仅跟踪浮点参数，跳过 Long/Int"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and sd[k].dtype == v.dtype and sd[k].shape == v.shape:
                sd[k].copy_(v)
        model.load_state_dict(sd, strict=False)

def train_model(train_root: str, img_size=288, seq_len=12,
                epochs=120, batch=16, base_lr=2e-4, wd=1e-4, warmup=3,
                amp=True, ema_decay=0.999, ckpt_dir="runtime/ckpts",
                mse_w=1.0, geo_w=0.5, tv_w=0.05, acc_w=0.02, roi_map=None,
                backbone="convnext_tiny", accum=1, workers=2,
                stride=1, prefetch=4, pin_memory=True, pretrained=True,
                patience: Optional[int]=10, min_delta: float=1e-4, target_mse: Optional[float]=None):

    set_seed(42)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    torch_, nn, F, Dataset, DataLoader, models, tqdm, cv2 = import_torch_bundle()
    train_ld, val_ld = build_dataloaders(train_root, img_size, seq_len, batch,
                                         workers=workers, roi_map=roi_map,
                                         stride=stride, prefetch=prefetch, pin_memory=pin_memory)

    model = build_model(backbone=backbone, d_model=512, nhead=8, layers=4,
                        dropout=0.1, use_sincos=True, pretrained=pretrained).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = next(model.parameters()).device
    log(f"[model] params={count_params(model)/1e6:.2f}M device={device}")

    from torch import amp as t_amp
    use_cuda = torch.cuda.is_available()
    scaler_cls = getattr(t_amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler = scaler_cls(enabled=amp and use_cuda)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    scheduler = cosine_scheduler(optimizer, epochs, warmup)
    ema = EMA(model, decay=ema_decay)

    def robust(x): return torch.sqrt(x*x + 1e-6)

    best_val = float("inf"); best_ep = 0
    ensure_dir(ckpt_dir); best_path = os.path.join(ckpt_dir, "best_pro.ckpt")
    no_improve = 0

    for ep in range(epochs):
        model.train(); meter=0.0; steps=0
        optimizer.zero_grad(set_to_none=True)
        accum_steps = max(1, int(accum))
        total_steps = len(train_ld)
        for step, (xb, yb) in enumerate(tqdm(train_ld, desc=f"train {ep+1}/{epochs}"), start=1):
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with t_amp.autocast(device_type='cuda', enabled=amp and use_cuda):
                pred = model(xb)  # (B,T,12)
                p = pred.view(pred.size(0), pred.size(1), 6, 2)
                angles = torch.atan2(p[...,0], p[...,1])  # (B,T,6)
                mse = (angles - yb).pow(2).mean()
                delta = torch.remainder(angles - yb + math.pi, 2*math.pi) - math.pi
                geo = (1 - torch.cos(delta)).mean()
                v = angles[:,1:,:] - angles[:,:-1,:]
                a = v[:,1:,:] - v[:,:-1,:]
                raw_loss = mse_w*mse + geo_w*geo + tv_w*robust(v).mean() + acc_w*(a*a).mean()
                group_size = min(accum_steps, total_steps - ((step - 1)//accum_steps)*accum_steps)
                loss = raw_loss / group_size

            scaler.scale(loss).backward()
            if step % accum_steps == 0 or step == total_steps:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            meter += raw_loss.item(); steps += 1

        scheduler.step()
        tr_loss = (meter/max(1,steps))
        log(f"[train] ep={ep+1} loss={tr_loss:.6f}")

        # ---- 验证：用 EMA 权重评估，但不影响训练权重 ----
        with torch.no_grad():
            model.eval()
            bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)

            vm=0.0; vc=0
            for xb, yb in val_ld:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                p = model(xb).view(xb.size(0), xb.size(1), 6, 2)
                ang = torch.atan2(p[...,0], p[...,1])
                vm += ((ang - yb)**2).mean().item(); vc += 1
            vm /= max(1, vc); log(f"[val] mse={vm:.6f}")

            model.load_state_dict(bak, strict=True)

            improved = (best_val - vm) > float(min_delta)
            if improved or best_val == float("inf"):
                best_val = vm; best_ep = ep+1; no_improve = 0
                # Include non-floating buffers (for example BatchNorm counters) too.
                checkpoint_state = {
                    k: ema.shadow.get(k, v).detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                torch.save({"model": checkpoint_state,
                            "val_mse": vm,
                            "cfg": dict(img_size=img_size, seq_len=seq_len, backbone=backbone)}, best_path)
                log(f"[ckpt] saved {best_path}")
            else:
                no_improve += 1

            if target_mse is not None and vm <= float(target_mse):
                log(f"[early-stop] target_mse={target_mse} reached at ep={ep+1} (val_mse={vm:.6f})")
                break
            if patience is not None and no_improve >= int(patience):
                log(f"[early-stop] no improvement for {no_improve} epochs (best={best_val:.6f} @ep{best_ep})")
                break

    return best_path

# ------------------------- 推理 & 多模态 TTA & 模板回填 -------------------------
def _frame_to_tensor(fr, img_size):
    import cv2
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fr = cv2.resize(fr, (img_size,img_size), interpolation=cv2.INTER_AREA)
    return (fr.astype(np.float32)/255.0).transpose(2,0,1)

def _apply_modal(fr, mode):
    import cv2
    if mode == "id": return fr
    if mode == "gray":
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if mode == "edges":
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); e = cv2.Canny(g, 64, 128); return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    if mode == "clahe":
        lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB); l,a,b = cv2.split(lab)
        cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge([cla,a,b]), cv2.COLOR_LAB2BGR)
    if mode == "bright":
        return np.clip(fr * 1.15 + 5, 0,255).astype(np.uint8)
    if mode == "gamma":
        g = 1.2; return np.clip(255.0*(fr/255.0)**g, 0,255).astype(np.uint8)
    if mode == "blur":
        return cv2.GaussianBlur(fr, (3,3), 0)
    return fr

def multi_modal_tta_predict(weights, video, img_size=288, seq_len=12, fps=30.0,
                            roi=None, tta_modes="id"):
    # 流式版本：不把整段视频读进内存
    torch, nn, F, Dataset, DataLoader, models, tqdm, cv2 = import_torch_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(weights, map_location=device, weights_only=True)
    cfg = ck.get("cfg", {})
    backbone = cfg.get("backbone", "convnext_tiny")
    seq_len = int(cfg.get("seq_len", seq_len))

    model = build_model(backbone=backbone, pretrained=False).to(device).eval()
    state = ck.get("model", None) or ck
    model.load_state_dict(state)

    modes = [m.strip().lower() for m in tta_modes.split(",") if m.strip()]

    def _stream_predict_mode(mode: str, skip: int = 0) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")

        preds_list = []
        buf = []

        try:
            with torch.no_grad():
                for frame_idx, fr in enumerate(_resampled_frames(cap, fps)):
                    if frame_idx < skip:  # shift TTA uses output-grid frames
                        continue
                    # ROI 只用于推理，不改变导出视频
                    if roi and roi[2] > roi[0] and roi[3] > roi[1]:
                        x0, y0, x1, y1 = roi
                        fr = fr[y0:y1, x0:x1]

                    if mode == "hflip":
                        fr = cv2.flip(fr, 1)
                    elif mode != "id":
                        fr = _apply_modal(fr, mode)

                    ten = _frame_to_tensor(fr, img_size)   # (C,H,W)
                    buf.append(ten)

                    if len(buf) == seq_len:
                        clip = np.stack(buf, 0)  # (T,C,H,W)
                        x = torch.from_numpy(clip[None, ...]).to(device)
                        y = model(x).view(1, seq_len, 6, 2)
                        ang = torch.atan2(y[..., 0], y[..., 1])[0].cpu().numpy()  # (T,6)
                        preds_list.append(ang)
                        buf.clear()

                if buf:
                    tail = np.stack(buf, 0)
                    pad = np.repeat(tail[-1][None, ...], seq_len - tail.shape[0], axis=0)
                    clip = np.concatenate([tail, pad], axis=0)
                    x = torch.from_numpy(clip[None, ...]).to(device)
                    y = model(x).view(1, seq_len, 6, 2)
                    ang = torch.atan2(y[..., 0], y[..., 1])[0].cpu().numpy()
                    preds_list.append(ang[:tail.shape[0]])
        finally:
            cap.release()
        if len(preds_list) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.concatenate(preds_list, axis=0).astype(np.float32)

    def _align_len(a: np.ndarray, N: int) -> np.ndarray:
        if a.shape[0] < N:
            if a.shape[0] == 0:
                return np.zeros((N, 6), dtype=np.float32)
            pad = np.repeat(a[-1:, :], N - a.shape[0], axis=0)
            return np.vstack([a, pad]).astype(np.float32)
        return a[:N].astype(np.float32)

    # 先跑 id 模式确定长度
    id_preds = _stream_predict_mode("id", skip=0)
    N = id_preds.shape[0]
    preds_pool = [id_preds]

    # 其他外观 TTA
    for m in ["hflip", "gray", "edges", "clahe", "bright", "gamma", "blur"]:
        if m in modes:
            p = _stream_predict_mode(m, skip=0)
            preds_pool.append(_align_len(p, N))

    # 位移 TTA
    if "shift1" in modes:
        p = _stream_predict_mode("id", skip=1)
        if p.shape[0] > 0:
            p = np.vstack([p[:1], p])
        preds_pool.append(_align_len(p, N))
    if "shift2" in modes:
        p = _stream_predict_mode("id", skip=2)
        if p.shape[0] > 0:
            p = np.vstack([p[:1].repeat(2, axis=0), p])
        preds_pool.append(_align_len(p, N))

    preds = np.mean(np.stack(preds_pool, axis=0), axis=0)
    return preds

def write_into_template(template_xlsx: str, out_xlsx: str, ts: np.ndarray, angles: np.ndarray, time_col="Time(s)"):
    from openpyxl import load_workbook
    if len(ts) == 0 or len(angles) == 0:
        raise ValueError("No predictions to write into template")
    wb = load_workbook(template_xlsx)
    if "Sheet1" not in wb.sheetnames:
        raise ValueError("Sheet1 not found")
    ws = wb["Sheet1"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    if time_col not in headers:
        raise ValueError(f"time column {time_col} not found")

    t_idx = headers.index(time_col) + 1
    axis_cols = []
    for i in range(1, 7):
        name = f"position(rad)_axis_{i}"
        if name not in headers:
            raise ValueError(f"missing {name}")
        axis_cols.append(headers.index(name) + 1)

    # 读模板时间
    t_vals = []
    for r in ws.iter_rows(min_row=2, min_col=t_idx, max_col=t_idx):
        v = r[0].value
        t_vals.append(np.nan if v is None else float(v))
    tpl = np.array(t_vals, dtype=np.float64)

    # 仅填充视频时间范围内的行
    mask_valid = (~np.isnan(tpl)) & (tpl >= ts.min()) & (tpl <= ts.max())
    if mask_valid.sum() == 0:
        ensure_dir(os.path.dirname(out_xlsx) or "."); wb.save(out_xlsx); return

    # ts/angles 排序去重
    order = np.argsort(ts)
    ts_sorted = ts[order]
    ang_sorted = angles[order]

    t_fill = tpl[mask_valid]
    rows_fill = np.where(mask_valid)[0]  # 0-based to row+2

    # 线性插值（不做边界外推）
    for ai, col in enumerate(axis_cols):
        y = np.interp(t_fill, ts_sorted, ang_sorted[:, ai])
        for rr, val in zip(rows_fill, y):
            ws.cell(row=rr + 2, column=col, value=float(val))

    ensure_dir(os.path.dirname(out_xlsx) or "."); wb.save(out_xlsx)
    log(f"[template] filled={int(mask_valid.sum())} -> {out_xlsx}")

def apply_calibrator(preds: np.ndarray, calib_path: str):
    if not calib_path or not os.path.isfile(calib_path): return preds
    d = np.load(calib_path); A,b = d["A"], d["b"]; return preds @ A + b

def predict_video(weights: str, video: str, img_size=288, seq_len=12, fps=30.0, roi=None,
                  tta="id", sg_win=11, sg_poly=3, calibrator=""):
    preds = multi_modal_tta_predict(weights, video, img_size, seq_len, fps, roi=roi, tta_modes=tta)
    # 短序列保护 + 平滑
    if sg_win and sg_win % 2 == 1 and sg_win >= 3:
        if len(preds) <= sg_win:
            sg_win = max(3, len(preds) - (1 - len(preds) % 2))
        if len(preds) > 2:
            preds = savgol_filter_1d(preds, window=sg_win, poly=min(sg_poly, sg_win - 1))
    preds = apply_calibrator(preds, calibrator)
    ts = np.array([i/float(fps) for i in range(len(preds))], dtype=np.float64)
    return ts, preds

def infer_to_template(weights: str, video: str, template_xlsx: str, out_xlsx: str,
                      img_size=288, seq_len=12, fps=30.0, roi=None,
                      tta="id",
                      sg_win=11, sg_poly=3, calibrator=""):
    ts, preds = predict_video(weights, video, img_size, seq_len, fps, roi, tta, sg_win, sg_poly, calibrator)
    write_into_template(template_xlsx, out_xlsx, ts, preds, time_col="Time(s)")
    return ts, preds

# ------------------------- 导出产物：CSV / 曲线图 / 叠字视频 -------------------------
def export_predictions_csv(ts: np.ndarray, preds: np.ndarray, path_csv: str):
    df = pd.DataFrame({
        "t": ts,
        "j1": preds[:,0], "j2": preds[:,1], "j3": preds[:,2],
        "j4": preds[:,3], "j5": preds[:,4], "j6": preds[:,5],
    })
    ensure_dir(os.path.dirname(path_csv) or ".")
    df.to_csv(path_csv, index=False)
    log(f"[export] csv -> {path_csv}")

def export_plot(ts: np.ndarray, preds: np.ndarray, path_png: str, title="Predicted Joint Angles"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(os.path.dirname(path_png) or ".")
    plt.figure(figsize=(12,6), dpi=150)
    labels = [f"j{i}" for i in range(1,7)]
    for i in range(6):
        plt.plot(ts, preds[:,i], label=labels[i])
    plt.xlabel("Time (s)"); plt.ylabel("Angle (rad)"); plt.title(title)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    log(f"[export] plot -> {path_png}")

def export_overlay_video(video_in: str, preds: np.ndarray, path_mp4: str,
                         max_fps: Optional[float]=None, pred_fps: Optional[float]=None):
    import cv2
    if len(preds) == 0:
        raise ValueError("No predictions to export")
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_in}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(native_fps) or native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot determine video FPS: {video_in}")
    fps_out = min(native_fps, float(max_fps)) if max_fps else native_fps
    prediction_fps = float(pred_fps) if pred_fps else native_fps
    if fps_out <= 0 or prediction_fps <= 0:
        cap.release()
        raise ValueError("Output and prediction FPS must be positive")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(os.path.dirname(path_mp4) or ".")
    writer = cv2.VideoWriter(path_mp4, fourcc, fps_out, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write video: {path_mp4}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for i, fr in enumerate(_resampled_frames(cap, fps_out)):
            t = i / fps_out
            pred_idx = min(int(round(t * prediction_fps)), len(preds) - 1)
            ang = preds[pred_idx]
            y0 = 40
            cv2.putText(fr, f"t={t:.2f}s", (20, y0), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
            for k in range(6):
                txt = f"j{k+1}: {ang[k]:+0.3f} rad"
                cv2.putText(fr, txt, (20, y0+30*(k+1)), font, 0.7, (50,255,50), 2, cv2.LINE_AA)
            writer.write(fr)
    finally:
        cap.release(); writer.release()
    log(f"[export] video -> {path_mp4}")

# ------------------------- 一键编排 -------------------------
def oneclick(args):
    if all([args.train_video1, args.train_video2, args.train_xlsx1, args.train_xlsx2, args.pre_video, args.pre_template]):
        files = dict(train_v1=args.train_video1, train_v2=args.train_video2,
                     train_x1=args.train_xlsx1, train_x2=args.train_xlsx2,
                     pre_video=args.pre_video, pre_template=args.pre_template)
    else:
        files = discover_files_recursive(os.getcwd())
    require_all(files); log("[discover] " + json.dumps(files, ensure_ascii=False))

    # 显存自适应（未指定或显式 --auto 时）
    if args.auto or not args.backbone:
        mem = get_gpu_mem_gb()
        rec = auto_hparams(mem)
        if not args.backbone: args.backbone = rec["backbone"]
        if args.batch == 16:     args.batch = rec["batch"]
        if args.img_size == 288: args.img_size = rec["img_size"]
        if args.seq_len == 12:   args.seq_len = rec["seq_len"]
        if args.accum == 1:      args.accum = rec["accum"]
        if args.workers == -1:   args.workers = rec["workers"]
        log(f"[autotune] mem={mem:.1f}GB -> {dict(backbone=args.backbone,batch=args.batch,img_size=args.img_size,seq_len=args.seq_len,accum=args.accum,workers=args.workers)}")
    else:
        if args.workers == -1: args.workers = 0 if os.name == "nt" else 4
        if not args.backbone: args.backbone = "convnext_tiny"

    # 1) 抽帧
    extract_frames(files["train_v1"], "data/train/Angle1", fps=args.fps)
    extract_frames(files["train_v2"], "data/train/Angle2", fps=args.fps)

    # 2) Excel -> CSV（并校验非全 0）
    convert_excel_to_angles(files["train_x1"], "data/train/Angle1/angles.csv", fps=args.fps)
    convert_excel_to_angles(files["train_x2"], "data/train/Angle2/angles.csv", fps=args.fps)
    assert_not_all_zero("data/train/Angle1/angles.csv")
    assert_not_all_zero("data/train/Angle2/angles.csv")

    # 3) ROI（按角度目录各自估计）
    roi1 = find_roi("data/train/Angle1/frames")
    roi2 = find_roi("data/train/Angle2/frames")
    roi_map = {"Angle1": roi1, "Angle2": roi2}
    log(f"[roi_map] {roi_map}")

    # 3.5) 预生成迷你帧
    if args.mini_cache:
        prepare_miniframes("data/train", roi_map, img_size=args.img_size)

    # 4) 训练（带早停）
    use_pretrained = not args.no_pretrained
    best = train_model("data/train", img_size=args.img_size, seq_len=args.seq_len,
                       epochs=args.epochs, batch=args.batch, base_lr=args.lr,
                       mse_w=args.mse_w, geo_w=args.geo_w, tv_w=args.tv_w, acc_w=args.acc_w,
                       amp=True, ema_decay=0.999, ckpt_dir="runtime/ckpts", roi_map=roi_map,
                       backbone=args.backbone, accum=args.accum, workers=args.workers,
                       stride=args.stride, prefetch=args.prefetch, pin_memory=True,
                       pretrained=use_pretrained, patience=args.patience,
                       min_delta=args.min_delta, target_mse=args.target_mse)

    # 5) 推理并写回模板（默认用 Angle1 的 ROI）+ 可选导出产物
    ensure_dir("runtime")
    out_xlsx = "runtime/Pre_Filled.xlsx"
    ts, preds = infer_to_template(best, files["pre_video"], files["pre_template"], out_xlsx,
                                  img_size=args.img_size, seq_len=args.seq_len, fps=args.fps,
                                  roi=roi1, tta=args.tta, sg_win=args.sg_win, sg_poly=args.sg_poly,
                                  calibrator=args.calibrator)
    log("DONE -> " + out_xlsx)

    # 导出
    if args.export_pred:
        export_predictions_csv(ts, preds, "runtime/Pre_Preds.csv")
    if args.export_plot:
        export_plot(ts, preds, "runtime/Pre_Preds.png", title="ABB RORD — Predicted Joint Angles")
    if args.export_video:
        export_overlay_video(files["pre_video"], preds, "runtime/Pre_Overlay.mp4",
                             max_fps=args.fps, pred_fps=args.fps)

# ------------------------- CLI -------------------------
def build_cli():
    p = argparse.ArgumentParser(description="ABB RORD Ultra One-Click (GPU Adaptive, Speed-up + Export + EarlyStop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("oneclick", help="full pipeline")
    s.add_argument("--train_video1", default=None)
    s.add_argument("--train_video2", default=None)
    s.add_argument("--train_xlsx1",  default=None)
    s.add_argument("--train_xlsx2",  default=None)
    s.add_argument("--pre_video",    default=None)
    s.add_argument("--pre_template", default=None)

    s.add_argument("--auto", action="store_true", help="根据显存自动调参")
    s.add_argument("--backbone", default="", help="convnext_tiny / resnet18（留空自动）")
    s.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    s.add_argument("--workers", type=int, default=-1, help="DataLoader 进程数")
    s.add_argument("--prefetch", type=int, default=4, help="DataLoader 预取批次数（workers>0 生效）")
    s.add_argument("--mini_cache", action="store_true", help="预生成迷你帧缓存以加速训练")
    s.add_argument("--no_pretrained", action="store_true", help="禁用 ImageNet 预训练权重（默认启用）")

    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--img_size", type=int, default=288)
    s.add_argument("--seq_len",  type=int, default=12)
    s.add_argument("--epochs",   type=int, default=120)
    s.add_argument("--batch",    type=int, default=16)
    s.add_argument("--lr",       type=float, default=2e-4)
    s.add_argument("--stride",   type=int, default=8, help="滑窗步长（越大越快）")

    s.add_argument("--mse_w", type=float, default=1.0)
    s.add_argument("--geo_w", type=float, default=0.5)
    s.add_argument("--tv_w",  type=float, default=0.05)
    s.add_argument("--acc_w", type=float, default=0.02)

    s.add_argument("--tta", default="id", help="可选测试增强，默认只解码一次；其他模式需单独验证效果")
    s.add_argument("--sg_win", type=int, default=11)
    s.add_argument("--sg_poly", type=int, default=3)
    s.add_argument("--calibrator", default="")

    # ✅ 新增导出开关
    s.add_argument("--export_pred", action="store_true", help="导出预测 CSV")
    s.add_argument("--export_video", action="store_true", help="导出覆盖角度的可视化视频")
    s.add_argument("--export_plot", action="store_true", help="导出六关节曲线图 PNG")

    # ✅ 新增早停
    s.add_argument("--patience", type=int, default=10, help="早停耐心")
    s.add_argument("--min_delta", type=float, default=1e-4, help="判定改进的最小幅度")
    s.add_argument("--target_mse", type=float, default=None, help="达到该 MSE 则提前停止（如 2e-2）")

    s2 = sub.add_parser("extract", help="extract frames")
    s2.add_argument("--video", required=True); s2.add_argument("--out_dir", required=True)
    s2.add_argument("--fps", type=float, default=30.0)

    s3 = sub.add_parser("excel2csv", help="convert training excel to angles.csv")
    s3.add_argument("--excel", required=True); s3.add_argument("--out", required=True)
    s3.add_argument("--fps", type=float, default=30.0)

    s4 = sub.add_parser("train", help="train only")
    s4.add_argument("--train_root", default="data/train")
    s4.add_argument("--img_size", type=int, default=288)
    s4.add_argument("--seq_len", type=int, default=12)
    s4.add_argument("--epochs", type=int, default=120)
    s4.add_argument("--batch", type=int, default=16)
    s4.add_argument("--lr", type=float, default=2e-4)
    s4.add_argument("--mse_w", type=float, default=1.0)
    s4.add_argument("--geo_w", type=float, default=0.5)
    s4.add_argument("--tv_w",  type=float, default=0.05)
    s4.add_argument("--acc_w", type=float, default=0.02)
    s4.add_argument("--backbone", default="convnext_tiny")
    s4.add_argument("--accum", type=int, default=1)
    s4.add_argument("--workers", type=int, default=-1)
    s4.add_argument("--prefetch", type=int, default=4)
    s4.add_argument("--stride", type=int, default=8)
    s4.add_argument("--no_pretrained", action="store_true")

    s5 = sub.add_parser("infer-template", help="infer and fill template")
    s5.add_argument("--weights", required=True)
    s5.add_argument("--video", required=True)
    s5.add_argument("--template", required=True)
    s5.add_argument("--out_xlsx", required=True)
    s5.add_argument("--fps", type=float, default=30.0)
    s5.add_argument("--img_size", type=int, default=288)
    s5.add_argument("--seq_len", type=int, default=12)
    s5.add_argument("--roi", default="")
    s5.add_argument("--tta", default="id")
    s5.add_argument("--sg_win", type=int, default=11)
    s5.add_argument("--sg_poly", type=int, default=3)
    s5.add_argument("--calibrator", default="")
    return p

def main():
    p = build_cli(); args = p.parse_args()
    if args.cmd == "oneclick":
        oneclick(args)
    elif args.cmd == "extract":
        extract_frames(args.video, args.out_dir, fps=args.fps)
    elif args.cmd == "excel2csv":
        convert_excel_to_angles(args.excel, args.out, fps=args.fps)
    elif args.cmd == "train":
        if args.workers == -1: args.workers = 0 if os.name == "nt" else 4
        train_model(args.train_root, img_size=args.img_size, seq_len=args.seq_len,
                    epochs=args.epochs, batch=args.batch, base_lr=args.lr,
                    mse_w=args.mse_w, geo_w=args.geo_w, tv_w=args.tv_w, acc_w=args.acc_w,
                    backbone=args.backbone, accum=args.accum, workers=args.workers,
                    stride=args.stride, prefetch=args.prefetch, pin_memory=True,
                    pretrained=not args.no_pretrained)
    elif args.cmd == "infer-template":
        roi = None
        if args.roi:
            try: roi = tuple(int(x) for x in args.roi.split(","))
            except: roi = None
        ts, preds = predict_video(args.weights, args.video, img_size=args.img_size, seq_len=args.seq_len, fps=args.fps,
                                  roi=roi, tta=args.tta, sg_win=args.sg_win, sg_poly=args.sg_poly, calibrator=args.calibrator)
        write_into_template(args.template, args.out_xlsx, ts, preds, time_col="Time(s)")
    else:
        raise ValueError("unknown cmd")

if __name__ == "__main__":
    main()
