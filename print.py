# -*- coding: utf-8 -*-
"""
原始信号可视化（时域 + STFT 频谱图）
- 支持按文件名或按索引取样本
- 若不指定，自动各取 1 个“停机(=0)”样本 + 1 个“非停机”样本
- 输出图片到 figs/ 目录
"""
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','Arial']
plt.rcParams['axes.unicode_minus'] = False

def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\.csv$", "", s)
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s

def _auto_nperseg(fs: float) -> int:
    # 与 predict.py 的策略一致
    import math
    target = max(32.0, fs/8.0)
    p = int(round(math.log2(target)))
    return 2 ** min(10, max(7, p))  # 128~1024

def _find_file(base_name: str, roots):
    base_norm = _norm_name(base_name)
    for root in roots:
        if not os.path.isdir(root): continue
        for d,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(".csv"):
                    b = _norm_name(os.path.splitext(f)[0])
                    if b == base_norm:
                        return os.path.join(d, f)
    return None

def _load_xyz(path: str):
    df = pd.read_csv(path)
    for c in ("X","Y","Z"):
        if c not in df.columns:
            raise ValueError(f"{path} 缺少列 {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    arr = df[["X","Y","Z"]].values.astype(np.float32)
    return arr

def _plot_one(row, data_path, outdir="figs", max_seconds=10.0):
    os.makedirs(outdir, exist_ok=True)
    fn = str(row["file_name"])
    fs = float(row["sample_rate"])
    ns = int(min(len(pd.read_csv(data_path)), int(max_seconds*fs)))  # 最多画前 max_seconds 秒（图更清晰）
    arr = _load_xyz(data_path)[:ns]

    t = np.arange(len(arr))/fs
    # --- 图1：三轴时域 ---
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.plot(t, arr[:,0], lw=0.7, label="X")
    ax.plot(t, arr[:,1], lw=0.7, label="Y")
    ax.plot(t, arr[:,2], lw=0.7, label="Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform | {fn} | fs={fs:g}Hz")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{fn}_wave.png"), dpi=200)
    plt.close(fig)

    # --- 图2：三轴 STFT 频谱图 ---
    nper = _auto_nperseg(fs)
    nover = nper//2
    fig, axes = plt.subplots(1,3, figsize=(12,3.6), sharey=True)
    for i,ax in enumerate(axes):
        f, tt, Z = stft(arr[:,i], fs=fs, nperseg=nper, noverlap=nover, boundary=None)
        S = np.log1p(np.abs(Z).astype(np.float32))
        im = ax.imshow(S, origin="lower", aspect="auto",
                       extent=[tt.min(), tt.max(), f.min(), f.max()])
        ax.set_title(["X","Y","Z"][i])
        ax.set_xlabel("Time (s)")
        if i==0: ax.set_ylabel("Freq (Hz)")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("log(1+|STFT|)")
    fig.suptitle(f"STFT Spectrograms | {fn} | nperseg={nper}")
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(os.path.join(outdir, f"{fn}_stft.png"), dpi=200)
    plt.close(fig)

def _load_meta(split_csv, fixed_csv=None):
    # 优先使用 *.fixed.csv（若由 predict.py 生成过）
    if fixed_csv and os.path.isfile(fixed_csv):
        df = pd.read_csv(fixed_csv)
    else:
        df = pd.read_csv(split_csv)
    # 统一列名
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"file_name","sample_rate","nominal_speed"}
    assert need.issubset(set(df.columns)), f"{split_csv} 需包含 {need}"
    return df

def main():
    ap = argparse.ArgumentParser("信号可视化")
    ap.add_argument("--train_csv", default="train.csv")
    ap.add_argument("--test_csv",  default="test.csv")
    ap.add_argument("--train_fixed", default="train.fixed.csv")
    ap.add_argument("--test_fixed",  default="test.fixed.csv")
    ap.add_argument("--train_dir", default="train")
    ap.add_argument("--test_dir",  default="test")
    ap.add_argument("--split", choices=["train","test","auto"], default="auto")
    ap.add_argument("--file", help="文件名（不带 .csv）")
    ap.add_argument("--idx", type=int, help="按索引选择样本（配合 --split）")
    ap.add_argument("--max_seconds", type=float, default=10.0, help="最多绘制前 N 秒数据")
    args = ap.parse_args()

    # 载入元数据
    train = _load_meta(args.train_csv, args.train_fixed) if os.path.isfile(args.train_csv) else None
    test  = _load_meta(args.test_csv,  args.test_fixed)  if os.path.isfile(args.test_csv)  else None

    targets = []

    if args.file:
        # 在 train / test 中查找该 file_name
        base = _norm_name(args.file)
        for df, tag in [(train,'train'), (test,'test')]:
            if df is None: continue
            cand = df[df['file_name'].map(_norm_name) == base]
            if len(cand):
                targets.append((tag, cand.iloc[0]))
                break
        if not targets:
            raise RuntimeError(f"在 train/test 中未找到文件名：{args.file}")
    elif args.idx is not None and args.split in ("train","test"):
        df = train if args.split=="train" else test
        if df is None: raise RuntimeError(f"缺少 {args.split}.csv")
        targets.append((args.split, df.iloc[args.idx]))
    else:
        # 自动：各取一个 停机(=0) + 一个非停机（来自 train）
        if train is None: raise RuntimeError("自动模式需要存在 train.csv")
        zero = train[train.get('running_speed', pd.Series([])).fillna(0)==0]
        nonz = train[train.get('running_speed', pd.Series([])).fillna(0)!=0]
        if len(zero):  targets.append(('train', zero.iloc[len(zero)//2]))
        if len(nonz):  targets.append(('train', nonz.iloc[len(nonz)//2]))
        if not targets:
            # 兜底随便拿两个
            k = min(2, len(train))
            for i in range(k): targets.append(('train', train.iloc[i]))

    # 绘图
    for tag, row in targets:
        base_name = str(row['file_name'])
        roots = [args.train_dir] if tag=='train' else [args.test_dir]
        # 同时在另一侧也搜一下以防用户放在一个目录里
        roots += [args.test_dir, args.train_dir]
        p = _find_file(base_name, roots)
        if not p:
            raise FileNotFoundError(f"找不到数据文件：{base_name}.csv（在 {roots} 中搜索）")
        print(f"[draw] {tag} -> {base_name}  ({p})")
        _plot_one(row, p, outdir="figs", max_seconds=args.max_seconds)

    print("✅ 已输出：figs/*_wave.png 与 figs/*_stft.png")

if __name__ == "__main__":
    main()
