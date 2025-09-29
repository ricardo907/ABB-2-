#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, json, math, subprocess
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

# ===================== 用户可配 =====================
FOLDS = 5
SEED = 42
CV_MODE = "stratified"   # "group" / "kfold" / "stratified"
GROUP_COL = "nominal_speed"  # group 模式用哪个列分组
OUTDIR = "figs_part5"
LOG_PATH = "train.log"

# 可选：当没有 train.log 时，是否由本脚本一键触发训练并 tee 到 train.log
AUTO_RUN_TRAIN_IF_NO_LOG = False
PREDICT_CMD = ["python", "predict.py", "--rebuild_cache"]  # 你自己的训练命令
# ===================================================

plt.rcParams["figure.dpi"] = 150

def _enable_cn_font():
    cands = ["Microsoft YaHei","SimHei","PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Source Han Sans CN"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in cands:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    else:
        local_font = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Regular.otf")
        if os.path.isfile(local_font):
            fm.fontManager.addfont(local_font)
            matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

_enable_cn_font()

def _pick_first_exist(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def load_inputs():
    meta_path = _pick_first_exist(["train.fixed.csv","train.csv"])
    if meta_path is None:
        raise FileNotFoundError("未找到 train.fixed.csv 或 train.csv")
    meta = pd.read_csv(meta_path)
    meta.columns = [c.lower() for c in meta.columns]
    need = {"running_speed","nominal_speed"}
    if not need.issubset(set(meta.columns)):
        raise ValueError("train(.fixed).csv 缺少 running_speed/nominal_speed 列")
    return meta

def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)

# ---------------- CV 分布两张图 ----------------
def _make_strat_bins(y, n_bins=10):
    s = pd.Series(y)
    q = min(n_bins, max(2, len(np.unique(y))//2))
    return pd.qcut(s, q=q, labels=False, duplicates="drop").astype(int).values

def _get_splits(meta, cv_mode):
    y = meta["running_speed"].astype(float).values
    if cv_mode == "group":
        groups = meta[GROUP_COL].round().astype(int).values
        splitter = GroupKFold(n_splits=FOLDS).split(meta, y, groups)
    elif cv_mode == "stratified":
        bins = _make_strat_bins(y, 10)
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(meta, bins)
    else:
        splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(meta)
    return list(splitter)

def fig_cv_nominal_bins(meta):
    splits = _get_splits(meta, CV_MODE)
    nom = meta["nominal_speed"].astype(float).values
    # 10 分位数分箱
    edges = np.unique(np.quantile(nom, np.linspace(0,1,10)))
    if len(edges) < 3:
        edges = np.unique(np.linspace(nom.min(), nom.max(), 4))
    labels = [f"[{edges[i]:.0f},{edges[i+1]:.0f}]" for i in range(len(edges)-1)]
    counts = np.zeros((FOLDS, len(edges)-1), dtype=int)
    for fi,(_, val_idx) in enumerate(splits):
        idx = np.clip(np.searchsorted(edges, nom[val_idx], side="right")-1, 0, len(edges)-2)
        for b in idx:
            counts[fi, b] += 1

    plt.figure(figsize=(8,4.5))
    bottom = np.zeros(len(edges)-1)
    for fi in range(FOLDS):
        plt.bar(range(len(edges)-1), counts[fi], bottom=bottom, label=f"Fold {fi+1}")
        bottom += counts[fi]
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.title(f"各折样本在 nominal_speed 分箱分布 | CV={CV_MODE}, folds={FOLDS}")
    plt.legend(ncol=min(FOLDS,5), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cv_nominal_bins.png"))
    plt.close()

def fig_cv_zero_ratio(meta):
    splits = _get_splits(meta, CV_MODE)
    y = meta["running_speed"].astype(float).values
    ratios = []
    for _, val_idx in splits:
        v = y[val_idx]
        ratios.append((v==0).mean()*100.0)
    plt.figure()
    plt.bar(range(1,FOLDS+1), ratios)
    plt.xticks(range(1,FOLDS+1), [f"Fold {i}" for i in range(1,FOLDS+1)])
    plt.ylabel("0 值占比 (%)")
    plt.title("各折 0-转速样本占比")
    plt.ylim(0, max(5.0, np.max(ratios)*1.2))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cv_zero_ratio.png"))
    plt.close()

# ---------------- 学习曲线（解析 train.log） ----------------
LOG_PAT = re.compile(r"\[cnn\]\s*Fold\s*(\d+)\s*\|\s*epoch\s*(\d+)\s*.*?val MAE=([0-9\.]+)\s*\|\s*R\^2=([\-0-9\.]+)")

def maybe_run_train():
    if not AUTO_RUN_TRAIN_IF_NO_LOG:
        return
    if not os.path.isfile("predict.py"):
        print("[warn] 没找到 predict.py，无法自动触发训练。")
        return
    print("[info] 未检测到 train.log，自动启动训练并写入 train.log …")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        # 同时回显到控制台
        p = subprocess.Popen(PREDICT_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
    print("[info] 训练完成，已生成 train.log")

def parse_log(log_path):
    if not os.path.isfile(log_path):
        return None
    data = []
    with open(log_path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            m = LOG_PAT.search(line)
            if m:
                fold = int(m.group(1))
                epoch = int(m.group(2))
                mae = float(m.group(3))
                r2 = float(m.group(4))
                data.append((fold, epoch, mae, r2))
    if not data:
        return None
    df = pd.DataFrame(data, columns=["fold","epoch","val_mae","val_r2"])
    # 每折按 epoch 排序并去重
    df = df.sort_values(["fold","epoch"]).drop_duplicates(["fold","epoch"])
    return df

def plot_learning_curves(df):
    # 保存明细
    csv_path = os.path.join(OUTDIR, "learning_curve.csv")
    df.to_csv(csv_path, index=False)

    # MAE
    plt.figure()
    for g,sub in df.groupby("fold"):
        plt.plot(sub["epoch"], sub["val_mae"], alpha=0.5, label=f"Fold {int(g)}")
    # 中位线
    med = df.groupby("epoch")["val_mae"].median()
    plt.plot(med.index, med.values, linewidth=2.5, label="Median", zorder=10)
    plt.xlabel("epoch"); plt.ylabel("val MAE")
    plt.title("学习曲线（验证 MAE）")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "learning_curve_mae.png"))
    plt.close()

    # R^2
    plt.figure()
    for g,sub in df.groupby("fold"):
        plt.plot(sub["epoch"], sub["val_r2"], alpha=0.5, label=f"Fold {int(g)}")
    med = df.groupby("epoch")["val_r2"].median()
    plt.plot(med.index, med.values, linewidth=2.5, label="Median", zorder=10)
    plt.xlabel("epoch"); plt.ylabel("val R²")
    plt.title("学习曲线（验证 R²）")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "learning_curve_r2.png"))
    plt.close()

def main():
    ensure_outdir(OUTDIR)
    meta = load_inputs()

    # 1) CV 分布两张图
    fig_cv_nominal_bins(meta)
    fig_cv_zero_ratio(meta)

    # 2) 学习曲线（需要 train.log）
    if not os.path.isfile(LOG_PATH):
        maybe_run_train()

    df = parse_log(LOG_PATH)
    if df is None or df.empty:
        print(f"[warn] 未能从 {LOG_PATH} 解析到 CNN 学习曲线（检查日志是否包含行形如："
              "[cnn] Fold 1 | epoch 12  val MAE=... | R^2=... ）")
    else:
        plot_learning_curves(df)

    print("[done] 已输出到：", os.path.abspath(OUTDIR))
    print("  - cv_nominal_bins.png")
    print("  - cv_zero_ratio.png")
    print("  - learning_curve_mae.png（如有日志）")
    print("  - learning_curve_r2.png（如有日志）")
    print("  - learning_curve.csv（如有日志）")

if __name__ == "__main__":
    main()
