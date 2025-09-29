#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

OUTDIR = "figs_part6"
os.makedirs(OUTDIR, exist_ok=True)

def _pick_first_exist(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def load_ground_truth():
    meta_path = _pick_first_exist(["train.fixed.csv", "train.csv"])
    if meta_path is None:
        raise FileNotFoundError("train.fixed.csv / train.csv 未找到（请先跑主训练脚本）")
    meta = pd.read_csv(meta_path)
    meta.columns = [c.lower() for c in meta.columns]
    if "running_speed" not in meta.columns:
        raise ValueError("元数据缺少 running_speed 列")
    return meta["running_speed"].astype(float).values

def load_oof_predictions():
    files = {
        "cat": "oof_cat.csv",
        "xgb": "oof_xgb.csv",
        "cnn": "oof_cnn.csv",
    }
    out = {}
    for name, fn in files.items():
        if os.path.isfile(fn):
            df = pd.read_csv(fn)
            col = "oof_pred" if "oof_pred" in df.columns else df.columns[0]
            out[name] = pd.to_numeric(df[col], errors="coerce").values.astype(float)
    if not out:
        raise FileNotFoundError("未发现 oof_*.csv（oof_cat.csv / oof_xgb.csv / oof_cnn.csv）")
    return out

def make_bar_r2(y, oof_dict):
    names, r2s, maes = [], [], []
    for k, v in oof_dict.items():
        n = min(len(y), len(v))
        yy, pp = y[:n], v[:n]
        r2s.append(r2_score(yy, pp))
        maes.append(mean_absolute_error(yy, pp))
        names.append(k.upper())
    order = np.argsort(r2s)[::-1]
    names = [names[i] for i in order]
    r2s   = [r2s[i]   for i in order]
    maes  = [maes[i]  for i in order]

    plt.figure(figsize=(7,4))
    xs = np.arange(len(names))
    plt.bar(xs, r2s)
    for i, (x, r, m) in enumerate(zip(xs, r2s, maes)):
        plt.text(x, r, f"R²={r:.3f}\nMAE={m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(xs, names)
    plt.ylabel("OOF R²")
    plt.title("OOF R² by Submodel")
    plt.tight_layout()
    outp = os.path.join(OUTDIR, "oof_r2_bar_submodels.png")
    plt.savefig(outp, dpi=200)
    plt.close()
    print("保存：", outp)

    # 返回最佳模型名（按 R²）
    best_idx = int(np.argmax(r2s))
    return names[best_idx].lower()

def make_scatter_best(y, oof_dict, best_name):
    pred = oof_dict[best_name]
    n = min(len(y), len(pred))
    yy, pp = y[:n], pred[:n]
    r2 = r2_score(yy, pp)
    mae = mean_absolute_error(yy, pp)

    lim_lo = float(min(yy.min(), pp.min()))
    lim_hi = float(max(yy.max(), pp.max()))
    pad = 0.02 * (lim_hi - lim_lo + 1e-6)
    lo, hi = lim_lo - pad, lim_hi + pad

    plt.figure(figsize=(6,6))
    plt.scatter(yy, pp, s=10, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Ground Truth (running_speed)")
    plt.ylabel(f"OOF Prediction ({best_name.upper()})")
    plt.title(f"OOF Scatter | {best_name.upper()}  (R²={r2:.3f}, MAE={mae:.1f})")
    plt.tight_layout()
    outp = os.path.join(OUTDIR, f"{best_name}_scatter_oof.png")
    plt.savefig(outp, dpi=200)
    plt.close()
    print("保存：", outp)

def main():
    y = load_ground_truth()
    oof = load_oof_predictions()
    best = make_bar_r2(y, oof)
    make_scatter_best(y, oof, best_name=best)

if __name__ == "__main__":
    main()
