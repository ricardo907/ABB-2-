#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, re, difflib, random, time, contextlib, warnings, pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from scipy.signal import stft
from scipy.stats import kurtosis
from scipy.linalg import solve_toeplitz

warnings.filterwarnings("ignore", category=FutureWarning)

# joblib（优先用于保存 .pkl)
try:
    import joblib
except Exception:
    joblib = None

# ======================
# 配置（可被命令行覆盖）
# ======================
CONFIG = dict(
    folds=5,
    seed=42,
    train_dir="train",
    test_dir="test",
    rebuild_cache=True,          # 首次 True
    cv="stratified",             # group / kfold / stratified
    cnn_cv="stratified",         # group / kfold / stratified
    target="residual",           # absolute / residual / relative
    extra_feats=True,            # 树模型高配（含小波包、谱峭度、倒谱峰、AR4 等）
    cnn_extra_phys=True,         # CNN 物理分支附加少量稳定统计
    cnn_epochs=50,
    cnn_batch=32,
    cnn_lr=3e-4,                 # 更稳的默认学习率
    max_len=8192,
    f_bins=129,
    t_frames=160,
    no_amp=False,
    workers=None,                # 自动
    xgb_gpu=False,
    cat_gpu=False,
    tune_xgb=0,                  # 关闭（>0 开启 optuna）
    tune_cat=0,
    export_onnx=True,
    onnx_dir="export_onnx",

    # ---- CNN 数据增强 & 归一化（新增）----
    cnn_aug_gain=False,
    cnn_gain_min=0.85,
    cnn_gain_max=1.20,
    cnn_aug_noise=False,
    cnn_noise_std=0.01,
    cnn_norm="zscore",           # zscore | minmax
)

# 依赖（树/Optuna/小波）
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
try:
    import pywt
except Exception:
    pywt = None
try:
    import optuna
except Exception:
    optuna = None

# Torch（CNN)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ========== 工具 ==========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _auto_locate_csv(start_dir: str, name: str) -> Optional[str]:
    p0 = os.path.join(start_dir, name)
    if os.path.isfile(p0): return p0
    for d,_,fs in os.walk(start_dir):
        if name in fs: return os.path.join(d, name)
    return None

def norm_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\.csv$", "", s, flags=re.IGNORECASE)
    s = s.lower()
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s

def build_index(root: str) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for d,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".csv"):
                base = os.path.splitext(f)[0]
                idx.setdefault(norm_name(base), []).append(os.path.join(d,f))
    return idx

def smart_pick(paths: List[str]) -> str:
    def key(p):
        parts = p.replace("\\","/").split("/")
        return (len(p), len(parts), p)
    return sorted(paths, key=key)[0]

def resolve_one(query: str, idx_norm: Dict[str, List[str]]):
    q_raw = str(query); q = norm_name(q_raw)
    if q in idx_norm:
        paths = idx_norm[q]
        return smart_pick(paths), "exact", 1.0, (len(paths)>1)
    subs = []
    for nb, paths in idx_norm.items():
        if q and (q in nb or nb in q):
            subs.extend(paths)
    if len(subs)==1:
        return subs[0], "substring", 1.0, False
    elif len(subs)>1:
        return smart_pick(subs), "substring", 0.9, True
    names = list(idx_norm.keys())
    best = difflib.get_close_matches(q, names, n=3, cutoff=0.75)
    if best:
        nb = best[0]; score = difflib.SequenceMatcher(None, q, nb).ratio()
        paths = idx_norm[nb]
        return smart_pick(paths), "fuzzy", float(score), (len(paths)>1 or len(best)>1)
    return None, "none", 0.0, False

def align_meta(meta: pd.DataFrame, data_root: str, split_name="train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx_norm = build_index(data_root)
    recs = []; fixed = meta.copy(); out_names = []; miss = 0; amb = 0
    for fn in meta["file_name"].astype(str):
        path, tp, sc, am = resolve_one(fn, idx_norm)
        if path is None:
            miss += 1; out_names.append(fn)
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            out_names.append(base); amb += int(am)
        recs.append({"file_name_original": fn, "matched_path": path or "", "match_type": tp, "score": sc, "ambiguous": bool(am)})
    fixed["file_name"] = out_names
    map_df = pd.DataFrame(recs)
    map_df.to_csv(f"file_map_{split_name}.csv", index=False)
    fixed.to_csv(f"{split_name}.fixed.csv", index=False)
    print(f"[map/{split_name}] 未匹配: {miss} | 可疑歧义: {amb} -> file_map_{split_name}.csv, {split_name}.fixed.csv")
    if miss:
        print("  ！有未匹配条目，请打开映射表检查（match_type=none）")
    return fixed, map_df


# ========== 数据 & 特征 ==========
def load_signal(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    for c in ("X","Y","Z"):
        if c not in df.columns: raise ValueError(f"{path} 缺少列 {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["X","Y","Z"]].fillna(0.0).values.astype(np.float32)

def to_numeric_3cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("X","Y","Z"):
        if c not in df.columns: raise ValueError(f"缺少列 {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0.0)

def hjorth_axis(x: np.ndarray) -> Tuple[float,float,float]:
    x = x.astype(np.float32)
    dx = np.diff(x, prepend=x[0])
    ddx = np.diff(dx, prepend=dx[0])
    var0 = float(np.var(x)+1e-12)
    var1 = float(np.var(dx)+1e-12)
    var2 = float(np.var(ddx)+1e-12)
    activity = var0
    mobility = math.sqrt(var1/var0)
    complexity = math.sqrt((var2/var1) / (var1/var0))
    return activity, mobility, complexity

def tkeo_axis(x: np.ndarray) -> Tuple[float,float]:
    x = x.astype(np.float32)
    psi = x[1:-1]**2 - x[:-2]*x[2:]
    return float(np.mean(psi)), float(np.std(psi))

def cepstrum_peak_axis(x: np.ndarray, fs: float, fmin=10.0, fmax=200.0) -> Tuple[float,float]:
    X = np.fft.rfft(x); logmag = np.log(np.abs(X)+1e-12)
    c = np.fft.irfft(logmag)
    qmin = int(max(1, np.floor(fs/float(fmax))))
    qmax = int(min(len(c)-1, np.ceil(fs/float(fmin))))
    if qmax <= qmin: return 0.0, 0.0
    seg = c[qmin:qmax]
    i = int(np.argmax(seg)); amp = float(seg[i]); q = (qmin+i)/fs
    f0 = 1.0/q if q>0 else 0.0
    return float(f0), amp

def spectral_kurtosis_axis(x: np.ndarray, fs: float) -> Tuple[float,float]:
    nper = _auto_nperseg(fs); nover = nper//2
    f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nover, boundary=None)
    S = np.abs(Z).astype(np.float32) + 1e-12
    sk = kurtosis(S, axis=1, fisher=False, bias=False)
    idx = int(np.nanargmax(sk))
    return float(np.nanmax(sk)), float(f[idx] if 0<=idx<len(f) else 0.0)

def ar_yule_walker_axis(x: np.ndarray, order=4) -> List[float]:
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')[len(x)-1:len(x)+order]
    R = solve_toeplitz((r[:-1], r[:-1]), r[1:]) if r[0] > 1e-12 else np.zeros(order)
    return [float(v) for v in -R]

def extract_time_features(df: pd.DataFrame) -> dict:
    df = to_numeric_3cols(df); feats = {}
    for axis in ("X","Y","Z"):
        x = df[axis].values.astype(float)
        feats[f"{axis}_mean"] = x.mean()
        feats[f"{axis}_std"]  = x.std()
        feats[f"{axis}_max"]  = x.max()
        feats[f"{axis}_min"]  = x.min()
        feats[f"{axis}_ptp"]  = float(np.ptp(x))
        feats[f"{axis}_rms"]  = float(np.sqrt(np.mean(np.square(x))))
        s = pd.Series(x); feats[f"{axis}_skew"] = float(s.skew()); feats[f"{axis}_kurt"] = float(s.kurt())
        act,mob,cpx = hjorth_axis(x); tkeo_m, tkeo_s = tkeo_axis(x)
        feats[f"{axis}_hj_activity"]=act; feats[f"{axis}_hj_mobility"]=mob; feats[f"{axis}_hj_complexity"]=cpx
        feats[f"{axis}_tkeo_mean"]=tkeo_m; feats[f"{axis}_tkeo_std"]=tkeo_s
    return feats

def extract_freq_features(df: pd.DataFrame, fs: float, extra=False) -> dict:
    df = to_numeric_3cols(df); feats = {}; n = len(df)
    for axis in ("X","Y","Z"):
        x = df[axis].values.astype(float)
        X = np.fft.rfft(x); freqs = np.fft.rfftfreq(n, d=1.0/float(fs))
        mag = np.abs(X) + 1e-12; total = mag.sum()
        centroid = float((freqs*mag).sum()/total); feats[f"{axis}_spectral_centroid"] = centroid
        bandwidth = float(np.sqrt(((freqs-centroid)**2 * mag).sum()/total)); feats[f"{axis}_spectral_bandwidth"] = bandwidth
        cum = np.cumsum(mag); thr = 0.85*cum[-1]; idx = int(np.searchsorted(cum, thr)); idx = min(idx, len(freqs)-1)
        feats[f"{axis}_spectral_rolloff"] = float(freqs[idx])
        norm = mag/total; entropy = float(-(norm*np.log(norm)).sum()); feats[f"{axis}_spectral_entropy"] = entropy
        if extra:
            f0c, camp = cepstrum_peak_axis(x, fs); skv, skf = spectral_kurtosis_axis(x, fs); ar4 = ar_yule_walker_axis(x, 4)
            feats[f"{axis}_cep_f0"] = f0c; feats[f"{axis}_cep_amp"] = camp
            feats[f"{axis}_sk_value"] = skv; feats[f"{axis}_sk_freq"] = skf
            for i,a in enumerate(ar4,1): feats[f"{axis}_ar{i}"] = a
        if len(mag)>1:
            ids = np.argsort(mag[1:])[-3:] + 1
            for i, k in enumerate(ids):
                feats[f"{axis}_peak{i+1}_freq"] = float(freqs[k]); feats[f"{axis}_peak{i+1}_mag"]  = float(mag[k])
        else:
            for i in range(3): feats[f"{axis}_peak{i+1}_freq"] = 0.0; feats[f"{axis}_peak{i+1}_mag"]  = 0.0
    return feats

def extract_corr_features(df: pd.DataFrame) -> dict:
    df = to_numeric_3cols(df)
    x,y,z = [df[c].values.astype(float) for c in ("X","Y","Z")]
    return {"corr_xy": float(np.corrcoef(x,y)[0,1]),
            "corr_xz": float(np.corrcoef(x,z)[0,1]),
            "corr_yz": float(np.corrcoef(y,z)[0,1])}

def wavelet_packet_energy_axis(x: np.ndarray, level=3, wavelet="db4") -> List[float]:
    if pywt is None:
        return [0.0]*(2**level)
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, maxlevel=level)
    leaves = [node.path for node in wp.get_level(level, order='natural')]
    energies = []; total = 1e-12
    for p in leaves:
        v = np.asarray(wp[p].data, dtype=np.float32)
        e = float(np.sum(v*v))
        energies.append(e); total += e
    energies = [e/total for e in energies]
    return energies

def build_feature_matrix(meta: pd.DataFrame, data_root: str, extra=False) -> pd.DataFrame:
    index = {}
    for d,_,fs in os.walk(data_root):
        for f in fs:
            if f.lower().endswith(".csv"): index.setdefault(os.path.splitext(f)[0], os.path.join(d,f))
    feats_list, missing = [], []
    for _, row in meta.iterrows():
        fn = str(row["file_name"]); path = index.get(fn)
        if path is None: missing.append(fn); continue
        df = pd.read_csv(path)
        feats = {}
        feats.update(extract_time_features(df))
        feats.update(extract_freq_features(df, float(row["sample_rate"]), extra=extra))
        feats.update(extract_corr_features(df))
        if extra:
            for axis in ("X","Y","Z"):
                w = wavelet_packet_energy_axis(df[axis].values.astype(np.float32))
                for i,v in enumerate(w): feats[f"{axis}_wp{i}"] = v
        feats["sample_rate"]   = float(row["sample_rate"])
        feats["nominal_speed"] = float(row["nominal_speed"])
        feats_list.append(feats)
    if missing:
        prev = ", ".join(missing[:5])
        raise FileNotFoundError(f"[features] 缺失 {len(missing)} 个文件（示例：{prev}）。请先运行映射或检查 file_name。")
    return pd.DataFrame(feats_list).astype(float)


# ===== STFT/物理（CNN) =====
def _auto_nperseg(fs: float) -> int:
    target = max(32.0, fs/8.0); p = int(round(math.log2(target)))
    return 2 ** min(10, max(7, p))

def _resample_freq_axis(S: np.ndarray, new_len: int) -> np.ndarray:
    F, T = S.shape
    if F == new_len: return S
    x_old = np.linspace(0,1,F, dtype=np.float32); x_new = np.linspace(0,1,new_len, dtype=np.float32)
    out = np.empty((new_len, T), dtype=np.float32)
    for t in range(T): out[:,t] = np.interp(x_new, x_old, S[:,t])
    return out

def stft_image(arr: np.ndarray, fs: float, f_bins: int=129, t_frames: int=128) -> np.ndarray:
    imgs = []; nperseg = _auto_nperseg(fs); noverlap = nperseg//2
    for a in range(3):
        f, t, Z = stft(arr[:,a], fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
        S = np.log1p(np.abs(Z).astype(np.float32)); S = _resample_freq_axis(S, f_bins)
        T = S.shape[1]
        if T >= t_frames: s = (T - t_frames)//2; S = S[:, s:s+t_frames]
        else: pad = np.zeros((S.shape[0], t_frames), dtype=np.float32); pad[:,:T] = S; S = pad
        imgs.append(S)
    return np.stack(imgs, axis=0)

def physics_features(arr: np.ndarray, fs: float, nominal_speed: float, harmonics: int = 5, extra=False) -> np.ndarray:
    feats = []
    N = len(arr)
    f_nom = float(nominal_speed)/60.0 if nominal_speed and nominal_speed>0 else 0.0
    axes_fft = []
    for a in range(3):
        X = np.fft.rfft(arr[:, a]); freqs = np.fft.rfftfreq(N, d=1.0/float(fs))
        mag = np.abs(X) + 1e-12; axes_fft.append((freqs, mag))
    bw = max(1.0, 0.02*(f_nom if f_nom>0 else fs/10))
    for a in range(3):
        freqs, mag = axes_fft[a]
        for k in range(1, harmonics+1):
            fc = k*f_nom
            if fc <= 0: feats.append(0.0); continue
            m = (freqs>=fc-bw) & (freqs<=fc+bw); feats.append(float(mag[m].sum()) if np.any(m) else 0.0)
        for b in (50.0,100.0,200.0): feats.append(float(mag[freqs<=b].sum()))
    x=arr[:,0]; y=arr[:,1]; z=arr[:,2]
    feats += [float(np.corrcoef(x,y)[0,1]), float(np.corrcoef(x,z)[0,1]), float(np.corrcoef(y,z)[0,1])]
    feats += [float(nominal_speed), float(fs)]
    if extra:
        feats += [float(np.std(arr[:,0])), float(np.std(arr[:,1])), float(np.std(arr[:,2]))]
    return np.array(feats, dtype=np.float32)


# ===== 目标变换 =====
def transform_target(y_abs: np.ndarray, base_nominal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "residual":
        return y_abs - base_nominal
    elif mode == "relative":
        return (y_abs - base_nominal) / (base_nominal + 1e-6)
    return y_abs

def invert_target(y_model: np.ndarray, base_nominal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "residual":
        return y_model + base_nominal
    elif mode == "relative":
        return y_model * (base_nominal + 1e-6) + base_nominal
    return y_model


# ===== CV =====
def _make_strat_bins(y_abs, n_bins=10):
    return pd.qcut(pd.Series(y_abs), q=min(n_bins, max(2, len(np.unique(y_abs))//2)),
                   labels=False, duplicates="drop").astype(int).values

def get_splitter_indices(X, y_abs, folds, seed, cv_mode, groups=None):
    if cv_mode == "group":
        if groups is None: raise ValueError("group 模式需要 groups")
        splitter = GroupKFold(n_splits=folds)
        return list(splitter.split(X, y_abs, groups=groups))
    elif cv_mode == "stratified":
        bins = _make_strat_bins(y_abs, n_bins=10)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(X, bins))
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(X))


# ===== 树模型 & OOF =====
def get_cat_model(use_gpu=False):
    if CatBoostRegressor is None: raise RuntimeError("catboost 未安装")
    params = dict(iterations=1500, learning_rate=0.03, depth=8,
                  loss_function="RMSE", l2_leaf_reg=3, subsample=0.8,
                  random_seed=42, verbose=False)
    if use_gpu: params.update(task_type="GPU", devices="0")
    return CatBoostRegressor(**params)

def get_xgb_model(use_gpu=False):
    if XGBRegressor is None: raise RuntimeError("xgboost 未安装")
    params = dict(n_estimators=1200, max_depth=8, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, n_jobs=0)
    if use_gpu: params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:       params.update(tree_method="hist")
    return XGBRegressor(**params)

def kfold_oof_tree(model_getter,
                   X: pd.DataFrame,
                   y_abs: np.ndarray,
                   base_nominal_train: np.ndarray,
                   folds: int,
                   seed: int,
                   cv_mode: str,
                   groups_for_group_cv: Optional[np.ndarray],
                   name: str,
                   target_mode: str):
    y_model_full = transform_target(y_abs, base_nominal_train, mode=target_mode)
    splits = get_splitter_indices(X, y_abs, folds, seed, cv_mode, groups_for_group_cv)
    oof_abs = np.zeros(len(X), dtype=float); scores=[]
    for i,(tr,va) in enumerate(splits,1):
        mdl = model_getter(); mdl.fit(X.iloc[tr], y_model_full[tr])
        pred_model = mdl.predict(X.iloc[va])
        pred_abs = invert_target(pred_model, base_nominal_train[va], mode=target_mode)
        oof_abs[va] = pred_abs
        r2 = r2_score(y_abs[va], pred_abs)
        print(f"[{name}] Fold {i} R^2 = {r2:.6f} | val std={y_abs[va].std():.3f}  range=[{y_abs[va].min():.1f},{y_abs[va].max():.1f}]")
        scores.append(r2)
    print(f"[{name}] Mean R^2 = {np.mean(scores):.6f} (± {np.std(scores):.6f})")
    mdl_full = model_getter(); mdl_full.fit(X, y_model_full)
    return oof_abs, mdl_full


# ===== Optuna（可选）=====
def tune_xgb_params(X, y_abs, base_nominal, groups, use_gpu=False, n_trials=30, seed=42, target_mode="residual"):
    if optuna is None:
        raise RuntimeError("需要安装 optuna：pip install optuna")
    y_model = transform_target(y_abs, base_nominal, target_mode)
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 1600),
            max_depth=trial.suggest_int("max_depth", 5, 12),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 0.0, 10.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0)
        )
        if use_gpu: params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        else:       params.update(tree_method="hist")
        splitter = GroupKFold(n_splits=3).split(X, y_model, groups=groups) if groups is not None \
                   else KFold(n_splits=3, shuffle=True, random_state=seed).split(X)
        scores=[]
        for tr,va in splitter:
            mdl = XGBRegressor(random_state=seed, n_jobs=0, **params)
            mdl.fit(X.iloc[tr], y_model[tr])
            pv = mdl.predict(X.iloc[va])
            scores.append(r2_score(y_model[va], pv))
        return -float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print("[tune_xgb] best params:", best, "| best score≈", -study.best_value)
    return best

def tune_cat_params(X, y_abs, base_nominal, groups, use_gpu=False, n_trials=30, seed=42, target_mode="residual"):
    if optuna is None:
        raise RuntimeError("需要安装 optuna：pip install optuna")
    y_model = transform_target(y_abs, base_nominal, target_mode)
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 800, 2000),
            depth=trial.suggest_int("depth", 6, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            loss_function="RMSE",
            random_seed=seed,
            verbose=False
        )
        if use_gpu: params.update(task_type="GPU", devices="0")
        splitter = GroupKFold(n_splits=3).split(X, y_model, groups=groups) if groups is not None \
                   else KFold(n_splits=3, shuffle=True, random_state=seed).split(X)
        scores=[]
        for tr,va in splitter:
            mdl = CatBoostRegressor(**params)
            mdl.fit(X.iloc[tr], y_model[tr])
            pv = mdl.predict(X.iloc[va])
            scores.append(r2_score(y_model[va], pv))
        return -float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print("[tune_cat] best params:", best, "| best score≈", -study.best_value)
    return best


# ========== CNN ==========
if TORCH_OK:
    def _amp_ctx(enabled: bool, device: str):
        if not enabled or device != "cuda":
            return contextlib.nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()

    def _grad_scaler(enabled: bool, device: str):
        if not enabled or device != "cuda":
            class _Dummy:
                def scale(self, x): return x
                def step(self, opt): opt.step()
                def update(self): pass
            return _Dummy()
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler()
        return torch.cuda.amp.GradScaler()

    class HybridDataset(Dataset):
        def __init__(self, meta: pd.DataFrame, data_root: str, max_len: int, f_bins: int, t_frames: int,
                     y_abs: Optional[np.ndarray]=None, target_mode: str="absolute",
                     extra_phys=False, is_train=False,
                     aug_gain=False, gain_range=(0.85,1.2),
                     aug_noise=False, noise_std=0.01,
                     norm_type="zscore", target_scale: float = 1.0):
            self.meta = meta.reset_index(drop=True)
            self.max_len, self.f_bins, self.t_frames = max_len, f_bins, t_frames
            self.target_mode = target_mode
            self.extra_phys = extra_phys
            self.is_train = is_train
            self.aug_gain = aug_gain
            self.gain_range = gain_range
            self.aug_noise = aug_noise
            self.noise_std = float(noise_std)
            self.norm_type = norm_type
            self.target_scale = float(target_scale)
            self.y_abs = None if y_abs is None else np.asarray(y_abs, dtype=np.float32)
            self.index = {}
            for d,_,fs in os.walk(data_root):
                for f in fs:
                    if f.lower().endswith(".csv"):
                        self.index.setdefault(os.path.splitext(f)[0], os.path.join(d,f))
        def __len__(self): return len(self.meta)

        def __getitem__(self, idx):
            row = self.meta.iloc[idx]
            fn = str(row["file_name"]); fs = float(row["sample_rate"]); nom = float(row["nominal_speed"])
            path = self.index.get(fn)
            if path is None: raise FileNotFoundError(f"找不到 {fn}.csv")
            arr = load_signal(path)

            # 简单增强
            if self.is_train and self.aug_gain:
                g = np.random.uniform(*self.gain_range)
                arr = arr * float(g)
            if self.is_train and self.aug_noise and self.noise_std > 0:
                arr = arr + np.random.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)

            # 时间对齐/补零
            n = len(arr)
            if n > self.max_len:
                s = (n - self.max_len)//2; arr = arr[s:s+self.max_len]
            elif n < self.max_len:
                pad = np.zeros((self.max_len,3), dtype=np.float32); pad[:n] = arr; arr = pad

            # 时间域标准化（逐轴）
            if self.norm_type == "zscore":
                m = arr.mean(axis=0, keepdims=True); s = arr.std(axis=0, keepdims=True)+1e-6
                arrn = (arr-m)/s
            else:
                mn = arr.min(axis=0, keepdims=True); mx = arr.max(axis=0, keepdims=True)
                arrn = (arr-mn)/(mx-mn+1e-6); arrn = (arrn-0.5)*2.0

            x_time = torch.from_numpy(arrn.T.copy())
            x_spec = torch.from_numpy(stft_image(arrn, fs, f_bins=self.f_bins, t_frames=self.t_frames).copy())

            # 物理分支：log1p + z-score（防爆炸）
            xp = physics_features(arr, fs, nom, extra=self.extra_phys).astype(np.float32)
            xp = np.sign(xp) * np.log1p(np.abs(xp))
            mu, sigma = xp.mean(dtype=np.float32), xp.std(dtype=np.float32)
            xp = (xp - mu) / (sigma + 1e-6)
            x_phys = torch.from_numpy(xp.copy())

            # 目标：缩放后返回（模型输出在缩放空间）
            if self.y_abs is None:
                y_scaled = 0.0
            else:
                y_model = transform_target(np.array([self.y_abs[idx]], dtype=np.float32),
                                           np.array([nom], dtype=np.float32),
                                           self.target_mode)[0]
                y_scaled = y_model / self.target_scale
            return x_time, x_spec, x_phys, torch.tensor(y_scaled, dtype=torch.float32), torch.tensor(nom, dtype=torch.float32)

    class Conv1DBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k=9, s=2, p=4):
            super().__init__(); self.conv = nn.Conv1d(in_ch,out_ch,kernel_size=k,stride=s,padding=p)
            self.bn = nn.BatchNorm1d(out_ch); self.act = nn.ReLU(inplace=True)
        def forward(self,x): return self.act(self.bn(self.conv(x)))

    class Conv2DBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k=(3,3), s=(1,2), p=(1,1)):
            super().__init__(); self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=k,stride=s,padding=p)
            self.bn = nn.BatchNorm2d(out_ch); self.act = nn.ReLU(inplace=True)
        def forward(self,x): return self.act(self.bn(self.conv(x)))

    class HybridNet(nn.Module):
        def __init__(self, phys_dim: int):
            super().__init__()
            self.time = nn.Sequential(Conv1DBlock(3,32), Conv1DBlock(32,64), Conv1DBlock(64,128))
            self.time_gap = nn.AdaptiveAvgPool1d(1); self.time_proj = nn.Linear(128,128)
            self.spec = nn.Sequential(Conv2DBlock(3,32), Conv2DBlock(32,64), Conv2DBlock(64,128))
            self.spec_gap = nn.AdaptiveAvgPool2d((1,1)); self.spec_proj = nn.Linear(128,128)
            self.phys_mlp = nn.Sequential(nn.Linear(phys_dim,128), nn.ReLU(True), nn.Linear(128,64), nn.ReLU(True))
            self.phys_proj = nn.Linear(64,128)
            self.gate = nn.Sequential(nn.Linear(384,64), nn.ReLU(True), nn.Linear(64,3))
            self.head = nn.Sequential(nn.Linear(128,64), nn.ReLU(True), nn.Linear(64,1))
        def forward(self, xt, xs, xp):
            ht = self.time_proj(self.time_gap(self.time(xt)).squeeze(-1))
            hs = self.spec_proj(self.spec_gap(self.spec(xs)).flatten(1))
            hp = self.phys_proj(self.phys_mlp(xp))
            H = torch.cat([ht,hs,hp],dim=1); w = torch.softmax(self.gate(H),dim=1)
            fused = w[:,0:1]*ht + w[:,1:2]*hs + w[:,2:3]*hp
            out = self.head(fused).squeeze(-1)
            return out

    def train_cnn_oof(train_meta: pd.DataFrame, test_meta: pd.DataFrame,
                      train_root: str, test_root: str,
                      y_abs: np.ndarray, base_nominal_train: np.ndarray, base_nominal_test: np.ndarray,
                      folds=5, seed=42, cv_mode="group",
                      max_len=8192, f_bins=129, t_frames=128,
                      epochs=25, batch=64, lr=1e-3, patience=6, amp=True, num_workers=0, pin_memory=False,
                      extra_phys=False, target_mode="residual",
                      aug_gain=False, gain_min=0.8, gain_max=1.25,
                      aug_noise=False, noise_std=0.01, norm_type="zscore"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 目标统一缩放（robust）
        y_model_all = transform_target(y_abs, base_nominal_train, target_mode).astype(np.float32)
        mad = np.median(np.abs(y_model_all - np.median(y_model_all))) + 1e-6
        target_scale = float(max(1.4826*mad, y_model_all.std() + 1e-6))
        print(f"[cnn] target_scale = {target_scale:.3f}")

        # 先构造一次，拿 phys_dim
        ds_tmp = HybridDataset(train_meta.iloc[:1], train_root, max_len, f_bins, t_frames, y_abs=None,
                               target_mode=target_mode, extra_phys=extra_phys, is_train=False,
                               aug_gain=False, aug_noise=False, norm_type=norm_type, target_scale=target_scale)
        _,_,xp,_,_ = ds_tmp[0]; phys_dim = int(xp.numel())

        os.makedirs("innov_runs", exist_ok=True)
        with open("innov_runs/last_meta.json","w",encoding="utf-8") as f:
            json.dump({"phys_dim": phys_dim, "max_len": max_len, "f_bins": f_bins, "t_frames": t_frames,
                       "extra_phys": bool(extra_phys)}, f, ensure_ascii=False, indent=2)

        # 切分
        if cv_mode == "group":
            groups = train_meta["nominal_speed"].round().astype(int).values
            splits = get_splitter_indices(train_meta, y_abs, folds, seed, "group", groups)
        elif cv_mode == "stratified":
            splits = get_splitter_indices(train_meta, y_abs, folds, seed, "stratified", None)
        else:
            splits = get_splitter_indices(train_meta, y_abs, folds, seed, "kfold", None)

        oof_abs = np.zeros(len(train_meta), dtype=float); scores=[]
        test_accum = []

        for fold,(tr,va) in enumerate(splits,1):
            # 每折重新初始化模型 & 优化器
            model = HybridNet(phys_dim=phys_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
            loss_fn = nn.HuberLoss(delta=1.0)
            scaler = _grad_scaler(amp, device)

            ds_tr = HybridDataset(train_meta.iloc[tr], train_root, max_len, f_bins, t_frames,
                                  y_abs=y_abs[tr], target_mode=target_mode, extra_phys=extra_phys, is_train=True,
                                  aug_gain=aug_gain, gain_range=(gain_min,gain_max),
                                  aug_noise=aug_noise, noise_std=noise_std, norm_type=norm_type,
                                  target_scale=target_scale)
            ds_va = HybridDataset(train_meta.iloc[va], train_root, max_len, f_bins, t_frames,
                                  y_abs=y_abs[va], target_mode=target_mode, extra_phys=extra_phys, is_train=False,
                                  aug_gain=False, aug_noise=False, norm_type=norm_type,
                                  target_scale=target_scale)
            dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

            best_mae = float("inf"); best_state=None; noimp=0

            for ep in range(1, epochs+1):
                model.train()
                for xt,xs,xp,y_scaled,nom in dl_tr:
                    xt,xs,xp,y_scaled = xt.to(device), xs.to(device), xp.to(device), y_scaled.to(device)
                    opt.zero_grad(set_to_none=True)
                    with _amp_ctx(amp, device):
                        out_scaled = model(xt,xs,xp)
                        loss = loss_fn(out_scaled, y_scaled)
                    scaler.scale(loss).backward()
                    if hasattr(scaler, "unscale_"): scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt); scaler.update()
                sch.step()

                # 验证（以 MAE 早停）
                model.eval(); pv_abs=[]; yt_abs=[]
                with torch.no_grad():
                    for xt,xs,xp,y_scaled,nom in dl_va:
                        xt,xs,xp = xt.to(device), xs.to(device), xp.to(device)
                        with _amp_ctx(amp, device):
                            out_scaled = model(xt,xs,xp)
                        pred_model = (out_scaled.cpu().numpy()) * target_scale
                        pred_abs   = invert_target(pred_model, nom.numpy(), target_mode)
                        pv_abs.append(pred_abs.reshape(-1))

                        y_model_true = (y_scaled.numpy()) * target_scale
                        y_true_abs   = invert_target(y_model_true, nom.numpy(), target_mode)
                        yt_abs.append(y_true_abs.reshape(-1))
                pv_abs = np.concatenate(pv_abs); yt_abs = np.concatenate(yt_abs)
                val_mae = mean_absolute_error(yt_abs, pv_abs); val_r2 = r2_score(yt_abs, pv_abs)
                print(f"[cnn] Fold {fold} | epoch {ep:02d}  val MAE={val_mae:.3f} | R^2={val_r2:.6f}")

                if val_mae + 1e-6 < best_mae:
                    best_mae = val_mae; best_state = model.state_dict(); noimp=0
                else:
                    noimp+=1
                    if noimp>=patience:
                        print("  early stop"); break

            if best_state is not None:
                model.load_state_dict(best_state)
                ckpt = f"innov_runs/hybrid_fold{fold}.pt"
                torch.save(best_state, ckpt); print(f"  saved best -> {ckpt}")

            # OOF
            pv_abs=[]
            with torch.no_grad():
                for xt,xs,xp,y_scaled,nom in dl_va:
                    xt,xs,xp = xt.to(device), xs.to(device), xp.to(device)
                    with _amp_ctx(amp, device):
                        out_scaled = model(xt,xs,xp)
                    pred_model = (out_scaled.cpu().numpy()) * target_scale
                    pred_abs   = invert_target(pred_model, nom.numpy(), target_mode)
                    pv_abs.append(pred_abs.reshape(-1))
            pv_abs = np.concatenate(pv_abs); oof_abs[va] = pv_abs
            r2 = r2_score(y_abs[va], pv_abs); scores.append(r2)
            print(f"[cnn] Fold {fold} best R^2={r2:.6f}")

            # Test 推断
            ds_te = HybridDataset(test_meta, test_root, max_len, f_bins, t_frames,
                                  y_abs=None, target_mode=target_mode, extra_phys=extra_phys, is_train=False,
                                  aug_gain=False, aug_noise=False, norm_type=norm_type,
                                  target_scale=target_scale)
            dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            te_abs=[]
            with torch.no_grad():
                for xt,xs,xp,_,nom in dl_te:
                    xt,xs,xp = xt.to(device), xs.to(device), xp.to(device)
                    with _amp_ctx(amp, device):
                        out_scaled = model(xt,xs,xp)
                    pred_model = (out_scaled.cpu().numpy()) * target_scale
                    pred_abs   = invert_target(pred_model, nom.numpy(), target_mode)
                    te_abs.append(pred_abs.reshape(-1))
            te_abs = np.concatenate(te_abs)
            test_accum.append(te_abs)

        print(f"[cnn] Mean R^2 = {np.mean(scores):.6f} (± {np.std(scores):.6f})")
        test_pred_abs = np.mean(np.stack(test_accum,axis=0),axis=0)
        return oof_abs, test_pred_abs
else:
    def train_cnn_oof(*args, **kwargs):
        raise RuntimeError("未安装 torch，无法训练 CNN")


# ===== 缓存 =====
def _cache_paths(train=True, extra=False):
    base = "cache_X_train" if train else "cache_X_test"
    suffix = "_ext" if extra else ""
    return base + suffix + ".parquet", base + suffix + ".csv"

def cache_exists(train=True, extra=False):
    pq,_ = _cache_paths(train, extra); return os.path.isfile(pq)

def read_cache(train=True, extra=False):
    pq, csv = _cache_paths(train, extra)
    try:
        return pd.read_parquet(pq)
    except Exception:
        if os.path.isfile(csv): return pd.read_csv(csv)
        raise

def write_cache(df: pd.DataFrame, train=True, extra=False):
    pq, csv = _cache_paths(train, extra)
    try:
        df.to_parquet(pq, index=False); print(f"[cache] 写入 {pq} ({len(df)}×{df.shape[1]})")
    except Exception as e:
        print("[cache] parquet 写入失败，改写 csv：", e); df.to_csv(csv, index=False)


# ===== ONNX 导出（Cat/XGB/CNN) =====
def _export_xgb_onnx(xgb_model, onnx_dir, n_features):
    ok = False
    try:
        import onnxmltools
        from onnxmltools.convert.xgboost import convert as convert_xgb
        from onnxconverter_common.data_types import FloatTensorType
        onx = convert_xgb(xgb_model, initial_types=[('input', FloatTensorType([None, int(n_features)]))])
        outp = os.path.join(onnx_dir, "xgboost_model.onnx")
        onnxmltools.utils.save_model(onx, outp)
        print("[ONNX] XGBoost ->", outp); ok = True
    except Exception:
        pass
    if not ok:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            onx = convert_sklearn(xgb_model, initial_types=[('input', FloatTensorType([None, int(n_features)]))])
            outp = os.path.join(onnx_dir, "xgboost_model.onnx")
            with open(outp, "wb") as f:
                f.write(onx.SerializeToString())
            print("[ONNX] XGBoost ->", outp); ok = True
        except Exception as e:
            print("[ONNX][warn] XGBoost 导出失败：", e)

def export_onnx(cat_model, xgb_model, onnx_dir, n_features_tree):
    os.makedirs(onnx_dir, exist_ok=True)
    try:
        if cat_model is not None:
            outp = os.path.join(onnx_dir, "catboost_model.onnx")
            cat_model.save_model(outp, format="onnx"); print("[ONNX] CatBoost ->", outp)
    except Exception as e:
        print("[ONNX][warn] CatBoost 导出失败：", e)
    try:
        if xgb_model is not None and n_features_tree is not None:
            _export_xgb_onnx(xgb_model, onnx_dir, n_features_tree)
    except Exception as e:
        print("[ONNX][warn] XGBoost 导出失败：", e)
    if TORCH_OK:
        meta_path = os.path.join("innov_runs","last_meta.json")
        ckpts = [os.path.join("innov_runs",f) for f in os.listdir("innov_runs")] if os.path.isdir("innov_runs") else []
        ckpts = [c for c in ckpts if c.endswith(".pt") and "hybrid_fold" in os.path.basename(c)]
        ckpt = sorted(ckpts)[0] if ckpts else None
        if ckpt and os.path.isfile(meta_path):
            try:
                meta = json.load(open(meta_path,"r",encoding="utf-8"))
                phys_dim = int(meta["phys_dim"]); L=int(meta["max_len"]); FB=int(meta["f_bins"]); TF=int(meta["t_frames"])
                net = HybridNet(phys_dim=phys_dim)
                sd = torch.load(ckpt, map_location='cpu'); net.load_state_dict(sd, strict=False); net.eval()
                xt = torch.randn(1,3,L, dtype=torch.float32)
                xs = torch.randn(1,3,FB,TF, dtype=torch.float32)
                xp = torch.randn(1,phys_dim, dtype=torch.float32)
                outp = os.path.join(onnx_dir, "hybrid_cnn.onnx")
                torch.onnx.export(net, (xt,xs,xp), outp,
                                  input_names=['xt','xs','xp'], output_names=['y'],
                                  dynamic_axes={'xt':{0:'batch'}, 'xs':{0:'batch'}, 'xp':{0:'batch'}, 'y':{0:'batch'}},
                                  opset_version=13)
                print("[ONNX] CNN ->", outp)
            except Exception as e:
                print("[ONNX][warn] CNN 导出失败：", e)
        else:
            print("[ONNX] 跳过 CNN：未检测到 ckpt 或 meta。")


# ===== 最终融合模型导出（Ridge）=====
def _ensure_models_dir():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def export_final_ensemble(rr: Ridge, base_model_names: List[str]):
    """
    导出最终融合模型（Ridge）到：
      models/最终融合模型.pkl
      models/最终融合模型.onnx
      models/最终融合模型.meta.json
    """
    model_dir = _ensure_models_dir()

    # 1) 保存 pkl（优先 joblib）
    pkl_path = os.path.join(model_dir, "最终融合模型.pkl")
    payload = {"model": rr, "base_models": base_model_names}
    try:
        if joblib is not None:
            joblib.dump(payload, pkl_path)
        else:
            with open(pkl_path, "wb") as f:
                pickle.dump(payload, f)
        print("[MODEL] Ridge 融合已保存 ->", pkl_path)
    except Exception as e:
        print("[MODEL][warn] 保存 pkl 失败：", e)

    # 2) 保存 ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onx = convert_sklearn(rr, initial_types=[("input", FloatTensorType([None, len(base_model_names)]))])
        onnx_path = os.path.join(model_dir, "最终融合模型.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print("[MODEL] ONNX 导出成功 ->", onnx_path)
    except Exception as e:
        print("[MODEL][warn] ONNX 导出失败（安装 skl2onnx 后重试）：", e)

    # 3) 保存 meta
    meta = {
        "base_models_order": base_model_names,
        "coef_raw": rr.coef_.tolist(),
        "intercept": float(rr.intercept_),
        "note": "输入须为按 base_models_order 排列的基模型预测值列向量；输出为最终预测"
    }
    meta_path = os.path.join(model_dir, "最终融合模型.meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("[MODEL] meta 信息已保存 ->", meta_path)
    except Exception as e:
        print("[MODEL][warn] 保存 meta 失败：", e)


# ===== CLI & 主流程 =====
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="ABB Cup")
    ap.add_argument("--folds", type=int, default=CONFIG["folds"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--test_csv",  default=None)
    ap.add_argument("--train_dir", default=CONFIG["train_dir"])
    ap.add_argument("--test_dir",  default=CONFIG["test_dir"])
    ap.add_argument("--rebuild_cache", action="store_true", default=CONFIG["rebuild_cache"])
    ap.add_argument("--extra_feats", action="store_true", default=CONFIG["extra_feats"])
    ap.add_argument("--cv", choices=["group","kfold","stratified"], default=CONFIG["cv"])
    ap.add_argument("--cnn_cv", choices=["group","kfold","stratified"], default=CONFIG["cnn_cv"])
    ap.add_argument("--target", choices=["absolute","residual","relative"], default=CONFIG["target"])
    ap.add_argument("--cnn_epochs", type=int, default=CONFIG["cnn_epochs"])
    ap.add_argument("--cnn_batch", type=int, default=CONFIG["cnn_batch"])
    ap.add_argument("--cnn_lr", type=float, default=CONFIG["cnn_lr"])
    ap.add_argument("--max_len", type=int, default=CONFIG["max_len"])
    ap.add_argument("--f_bins", type=int, default=CONFIG["f_bins"])
    ap.add_argument("--t_frames", type=int, default=CONFIG["t_frames"])
    ap.add_argument("--no_amp", action="store_true", default=CONFIG["no_amp"])
    ap.add_argument("--workers", type=int, default=CONFIG["workers"] if CONFIG["workers"] is not None else -1)
    ap.add_argument("--cnn_extra_phys", action="store_true", default=CONFIG["cnn_extra_phys"])
    ap.add_argument("--xgb_gpu", action="store_true", default=CONFIG["xgb_gpu"])
    ap.add_argument("--cat_gpu", action="store_true", default=CONFIG["cat_gpu"])
    ap.add_argument("--tune_xgb", type=int, default=CONFIG["tune_xgb"])
    ap.add_argument("--tune_cat", type=int, default=CONFIG["tune_cat"])
    ap.add_argument("--export_onnx", action="store_true", default=CONFIG["export_onnx"])
    ap.add_argument("--onnx_dir", default=CONFIG["onnx_dir"])

    # CNN 增强/归一化
    ap.add_argument("--cnn_aug_gain", action="store_true", default=CONFIG["cnn_aug_gain"])
    ap.add_argument("--cnn_gain_min", type=float, default=CONFIG["cnn_gain_min"])
    ap.add_argument("--cnn_gain_max", type=float, default=CONFIG["cnn_gain_max"])
    ap.add_argument("--cnn_aug_noise", action="store_true", default=CONFIG["cnn_aug_noise"])
    ap.add_argument("--cnn_noise_std", type=float, default=CONFIG["cnn_noise_std"])
    ap.add_argument("--cnn_norm", choices=["zscore","minmax"], default=CONFIG["cnn_norm"])

    args = ap.parse_args()
    if args.workers == -1:
        args.workers = None
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = args.train_csv or _auto_locate_csv(script_dir, "train.csv")
    test_csv  = args.test_csv  or _auto_locate_csv(script_dir, "test.csv")
    if not train_csv or not os.path.isfile(train_csv): print("ERROR: 找不到 train.csv"); sys.exit(1)
    if not test_csv or not os.path.isfile(test_csv):   print("ERROR: 找不到 test.csv"); sys.exit(1)
    if not os.path.isdir(args.train_dir) or not os.path.isdir(args.test_dir):
        print("ERROR: 缺少 train/ 或 test/ 目录"); sys.exit(1)

    print(f"[path] train_csv={train_csv}")
    print(f"[path] test_csv ={test_csv}")
    print(f"[path] train_dir={args.train_dir}")
    print(f"[path] test_dir ={args.test_dir}")

    t0=time.perf_counter()
    raw_train = pd.read_csv(train_csv); raw_test  = pd.read_csv(test_csv)
    need_cols = {"file_name","sample_rate","nominal_speed"}
    low = set(map(str.lower, raw_train.columns))
    if not need_cols.issubset(low): print("ERROR: train.csv 缺少必要列 file_name / sample_rate / nominal_speed"); sys.exit(1)
    low = set(map(str.lower, raw_test.columns))
    if not need_cols.issubset(low): print("ERROR: test.csv 缺少必要列 file_name / sample_rate / nominal_speed"); sys.exit(1)

    train_fixed, _ = align_meta(raw_train, args.train_dir, "train")
    test_fixed,  _ = align_meta(raw_test,  args.test_dir,  "test")
    print(f"[timer] 文件映射耗时：{time.perf_counter()-t0:.2f}s"); t0=time.perf_counter()

    # 特征（树）—— 不过滤 0 样本
    if (not args.rebuild_cache) and cache_exists(train=True, extra=args.extra_feats):
        print("[cache] 载入训练特征缓存 …"); X = read_cache(train=True, extra=args.extra_feats)
    else:
        print("[step] building train features ...")
        X = build_feature_matrix(train_fixed, args.train_dir, extra=args.extra_feats); write_cache(X, train=True, extra=args.extra_feats)
    y_abs = train_fixed["running_speed"].astype(float).reset_index(drop=True).values
    base_train = train_fixed["nominal_speed"].astype(float).reset_index(drop=True).values

    if (not args.rebuild_cache) and cache_exists(train=False, extra=args.extra_feats):
        print("[cache] 载入测试特征缓存 …"); X_test = read_cache(train=False, extra=args.extra_feats)
    else:
        print("[step] building test features ...")
        X_test = build_feature_matrix(test_fixed, args.test_dir, extra=args.extra_feats); write_cache(X_test, train=False, extra=args.extra_feats)
    base_test = test_fixed["nominal_speed"].astype(float).reset_index(drop=True).values
    print(f"[timer] 特征准备耗时：{time.perf_counter()-t0:.2f}s"); t0=time.perf_counter()

    # 自动调参（可选）
    xgb_overrides = None; cat_overrides = None
    groups_for_tune = train_fixed["nominal_speed"].round().astype(int).values if args.cv=="group" else None
    if args.tune_xgb>0 and XGBRegressor is not None:
        try:
            xgb_overrides = tune_xgb_params(X, y_abs, base_train, groups_for_tune, use_gpu=args.xgb_gpu,
                                            n_trials=args.tune_xgb, seed=args.seed, target_mode=args.target)
        except Exception as e:
            print("[warn] XGB 调参失败，使用默认参数：", e)
    if args.tune_cat>0 and CatBoostRegressor is not None:
        try:
            cat_overrides = tune_cat_params(X, y_abs, base_train, groups_for_tune, use_gpu=args.cat_gpu,
                                            n_trials=args.tune_cat, seed=args.seed, target_mode=args.target)
        except Exception as e:
            print("[warn] CatBoost 调参失败，使用默认参数：", e)

    def _get_cat():
        m = get_cat_model(use_gpu=args.cat_gpu)
        if cat_overrides: m.set_params(**cat_overrides)
        return m
    def _get_xgb():
        m = get_xgb_model(use_gpu=args.xgb_gpu)
        if xgb_overrides: m.set_params(**xgb_overrides)
        return m

    # 树模型 OOF
    oof_dict, preds_dict = {}, {}
    groups = train_fixed["nominal_speed"].round().astype(int).values if args.cv=="group" else None

    cat_full = None
    try:
        print("\n[CatBoost] 5-fold OOF ...")
        oof_c_abs, cat_full = kfold_oof_tree(_get_cat, X, y_abs, base_train,
                                             folds=args.folds, seed=args.seed,
                                             cv_mode=args.cv, groups_for_group_cv=groups, name="cat",
                                             target_mode=args.target)
        pd.DataFrame({"oof_pred": oof_c_abs}).to_csv("oof_cat.csv", index=False)
        pred_model = cat_full.predict(X_test)
        preds_dict["cat"] = invert_target(pred_model, base_test, args.target)
        pd.DataFrame({"running_speed": preds_dict["cat"]}).to_csv("predictions_cat.csv", index=False)
        oof_dict["cat"] = oof_c_abs
    except Exception as e:
        print("[warn] CatBoost 跳过：", e)
    print(f"[timer] CatBoost 耗时：{time.perf_counter()-t0:.2f}s"); t0=time.perf_counter()

    xgb_full = None
    try:
        print("\n[XGBoost] 5-fold OOF ...")
        oof_x_abs, xgb_full = kfold_oof_tree(_get_xgb, X, y_abs, base_train,
                                             folds=args.folds, seed=args.seed,
                                             cv_mode=args.cv, groups_for_group_cv=groups, name="xgb",
                                             target_mode=args.target)
        pd.DataFrame({"oof_pred": oof_x_abs}).to_csv("oof_xgb.csv", index=False)
        pred_model = xgb_full.predict(X_test)
        preds_dict["xgb"] = invert_target(pred_model, base_test, args.target)
        pd.DataFrame({"running_speed": preds_dict["xgb"]}).to_csv("predictions_xgb.csv", index=False)
        oof_dict["xgb"] = oof_x_abs
    except Exception as e:
        print("[warn] XGBoost 跳过：", e)
    print(f"[timer] XGBoost 耗时：{time.perf_counter()-t0:.2f}s"); t0=time.perf_counter()

    # CNN
    if TORCH_OK:
        try:
            print("\n[Hybrid CNN] 5-fold OOF ...")
            nw = args.workers if args.workers is not None else max(0, min(8, (os.cpu_count() or 1)-1))
            oof_h_abs, pred_h_abs = train_cnn_oof(
                train_fixed, test_fixed, args.train_dir, args.test_dir,
                y_abs=y_abs, base_nominal_train=base_train, base_nominal_test=base_test,
                folds=args.folds, seed=args.seed, cv_mode=args.cnn_cv,
                max_len=args.max_len, f_bins=args.f_bins, t_frames=args.t_frames,
                epochs=args.cnn_epochs, batch=args.cnn_batch, lr=args.cnn_lr, patience=6,
                amp=(not args.no_amp), num_workers=nw, pin_memory=torch.cuda.is_available(),
                extra_phys=args.cnn_extra_phys, target_mode=args.target,
                aug_gain=args.cnn_aug_gain, gain_min=args.cnn_gain_min, gain_max=args.cnn_gain_max,
                aug_noise=args.cnn_aug_noise, noise_std=args.cnn_noise_std, norm_type=args.cnn_norm
            )
            pd.DataFrame({"oof_pred": oof_h_abs}).to_csv("oof_cnn.csv", index=False)
            pd.DataFrame({"running_speed": pred_h_abs}).to_csv("predictions_cnn.csv", index=False)
            oof_dict["cnn"] = oof_h_abs; preds_dict["cnn"] = pred_h_abs
        except Exception as e:
            print("[warn] CNN 训练失败，跳过：", e)
    else:
        print("\n[info] 未安装 torch，跳过 CNN。")
    print(f"[timer] CNN/融合前准备耗时：{time.perf_counter()-t0:.2f}s"); t0=time.perf_counter()

    # 融合（绝对转速）
    if len(oof_dict) == 0:
        print("\n[ERROR] 没有可用基模型，无法融合与提交。"); sys.exit(1)
    names = list(oof_dict.keys())
    oof_mat = np.vstack([oof_dict[n] for n in names]).T

    rr = Ridge(alpha=1.0, fit_intercept=True, positive=True)
    rr.fit(oof_mat, y_abs)

    # 显示归一化权重（便于 PPT）
    w_raw = rr.coef_
    w_norm = w_raw / (w_raw.sum() + 1e-12)
    b = rr.intercept_
    weights_show = {n: float(wi) for n, wi in zip(names, w_norm)}
    print("\n[blend] normalized weights (display only):", weights_show, "| intercept:", float(b))

    with open("blend_weights.json","w",encoding="utf-8") as f:
        json.dump({
            "weights_raw": {n: float(w) for n, w in zip(names, w_raw)},
            "weights_normalized": {n: float(w) for n, w in zip(names, w_norm)},
            "intercept": float(b),
            "order": names
        }, f, ensure_ascii=False, indent=2)

    # 用 rr.predict 确保与导出一致
    mat_test = np.vstack([preds_dict[n] for n in names]).T
    pred_ens_abs = rr.predict(mat_test)

    # 导出最终融合模型
    export_final_ensemble(rr, names)

    # 输出提交
    pd.DataFrame({"running_speed": pred_ens_abs}).to_csv("predictions_ensemble.csv", index=False)
    submit = test_fixed[["file_name"]].copy(); submit["running_speed"] = pred_ens_abs
    submit.to_csv("predictions.csv", index=False)  # 两列（文件名 + 预测）
    submit_full = test_fixed[["file_name","sample_rate","nominal_speed"]].copy()
    submit_full["running_speed"] = pred_ens_abs
    submit_full.to_csv("predictions_submit.csv", index=False)  # 比赛提交表格
    print("[OK] 提交文件：predictions.csv / predictions_submit.csv / predictions_ensemble.csv")
    print("[done] 融合模型已导出至 models/（pkl + onnx + meta）")

    # Cat/XGB/CNN ONNX
    if args.export_onnx:
        cat_nfeat = X.shape[1] if 'X' in locals() else None
        export_onnx(cat_full, xgb_full, args.onnx_dir, cat_nfeat)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print("\nERROR:", e, file=sys.stderr)
        print("提示：原始 CSV 包含列 X,Y,Z；train/test 的 file_name 不带 .csv 后缀。", file=sys.stderr)
        sys.exit(1)
