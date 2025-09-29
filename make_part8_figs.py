#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "figs_part8"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 150

# --- 一键中文字体（复制到脚本最上方）---
def use_chinese_font(add_font_path: str | None = None):
    """
    让 Matplotlib 正常显示中文：
    - 优先使用你传入的字体文件 add_font_path（.ttf/.ttc）
    - 否则自动在系统已装字体中按常见中文字体名轮询
    - 统一修正负号显示（axes.unicode_minus=False）
    """
    import os
    import matplotlib
    import matplotlib.font_manager as fm

    # 常见中文字体名（按优先级）
    preferred = [
        "Microsoft YaHei",          # 微软雅黑（Win）
        "SimHei",                   # 黑体（Win）
        "SimSun",                   # 宋体（Win）
        "PingFang SC",              # 苹方（macOS）
        "Hiragino Sans GB",         # 冬青黑（macOS）
        "WenQuanYi Micro Hei",      # 文泉驿微米黑（Linux）
        "Noto Sans CJK SC",         # 谷歌思源黑体
        "Source Han Sans CN",       # 思源黑体
    ]

    chosen = None

    
    if add_font_path and os.path.exists(add_font_path):
        try:
            fm.fontManager.addfont(add_font_path)
            chosen = fm.FontProperties(fname=add_font_path).get_name()
        except Exception:
            chosen = None

   
    if not chosen:
        available = {f.name for f in fm.fontManager.ttflist}
        for name in preferred:
            if name in available:
                chosen = name
                break

    
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
  
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 便于确认
    print(f"[font] chinese font -> {chosen or '未找到（将显示英文；建议提供 .ttf/.ttc 路径）'}")

use_chinese_font()


# ---------- 工具 ----------
def safe_lowess(x, y, frac=0.2):
    """优先用 statsmodels.lowess；若无则用简单滑窗中位数近似."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xy = np.c_[x, y]
        xy = xy[np.argsort(xy[:,0])]
        sm = lowess(xy[:,1], xy[:,0], frac=frac, return_sorted=True)
        return sm[:,0], sm[:,1]
    except Exception:
        # simple moving median
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        win = max(5, int(len(xs)*frac))
        yy = []
        for i in range(len(xs)):
            L = max(0, i - win//2)
            R = min(len(xs), i + win//2 + 1)
            yy.append(np.median(ys[L:R]))
        return xs, np.array(yy)

def bin_by_quantile(v, q=8):
    edges = np.unique(np.quantile(v, np.linspace(0,1,q+1)))
    if len(edges) < 3:  # 兜底
        edges = np.unique(np.linspace(v.min(), v.max(), q+1))
    return edges

# ---------- 读入数据 ----------
meta = pd.read_csv("train.fixed.csv")
y = meta["running_speed"].astype(float).values
nom = meta["nominal_speed"].astype(float).values

def read_oof(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # 兼容列名 oof_pred / running_speed
    for c in ["oof_pred","running_speed","pred"]:
        if c in df.columns:
            return df[c].values.astype(float)
    raise ValueError(f"{path} 列名不识别")

oof_cat = read_oof("oof_cat.csv")
oof_xgb = read_oof("oof_xgb.csv")
oof_cnn = read_oof("oof_cnn.csv")

# 用融合权重重建“最终模型”的 OOF（b + Σ w_i · oof_i）
with open("blend_weights.json","r",encoding="utf-8") as f:
    bw = json.load(f)
order    = bw["order"]                      # e.g. ["cat","xgb","cnn"]
w_raw    = [bw["weights_raw"][k] for k in order]
b        = float(bw["intercept"])

oof_map = {"cat":oof_cat, "xgb":oof_xgb, "cnn":oof_cnn}
stack = []
names = []
for k in order:
    if oof_map[k] is None:
        # 缺哪个就用0，不让脚本崩（也会在图标题里提示）
        stack.append(np.zeros_like(y))
        names.append(k+"(MISSING)")
    else:
        stack.append(oof_map[k])
        names.append(k)
Xoof = np.vstack(stack).T  # (N, K)
w = np.array(w_raw, dtype=float)

y_pred = b + Xoof.dot(w)
res = y - y_pred
abs_res = np.abs(res)

print("[info] 重建融合 OOF 完成：", dict(zip(names, w_raw)), " | b=", b)

# ---------- 图 1：残差直方图 ----------
plt.figure()
bins = 60
plt.hist(res, bins=bins)
mu, med = float(np.mean(res)), float(np.median(res))
mad = float(np.median(np.abs(res - med)))  # robust
plt.axvline(mu, linestyle="--")
plt.axvline(med, linestyle=":")
plt.title(f"残差分布（融合 OOF） | mean={mu:.1f}, median={med:.1f}, MAD={mad:.1f}")
plt.xlabel("residual = y - y_pred")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residual_hist.png"))
plt.close()

# ---------- 图 2：残差 vs nominal_speed（带平滑） ----------
plt.figure()
plt.scatter(nom, res, s=10, alpha=0.5)
xs, ys = safe_lowess(nom, res, frac=0.2)
plt.plot(xs, ys, linestyle="--")
plt.title("残差 vs nominal_speed（含平滑趋势）")
plt.xlabel("nominal_speed")
plt.ylabel("residual")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residual_vs_nominal.png"))
plt.close()

# ---------- 图 3：按 nominal 分箱的 MAE ----------
edges = bin_by_quantile(nom, q=8)
labels, maes, counts = [], [], []
for i in range(len(edges)-1):
    lo, hi = edges[i], edges[i+1]
    m = (nom >= lo) & (nom < hi if i < len(edges)-2 else nom <= hi)
    if m.sum() == 0: 
        continue
    labels.append(f"[{lo:.0f},{hi:.0f}]")
    maes.append(float(np.mean(abs_res[m])))
    counts.append(int(m.sum()))
plt.figure(figsize=(10,4))
plt.bar(range(len(maes)), maes)
plt.xticks(range(len(maes)), labels, rotation=30)
for i,(mae,cnt) in enumerate(zip(maes,counts)):
    plt.text(i, mae, f"MAE={mae:.1f}\n(n={cnt})", ha="center", va="bottom")
plt.title("按 nominal_speed 分箱的 MAE（融合 OOF）")
plt.xlabel("nominal_speed bins")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "mae_by_nominal_bins.png"))
plt.close()

# ---------- 图 4：2D 误差热力图（nominal × actual） ----------
nbin = 8
H_nom, x_edges = np.histogram(nom, bins=bin_by_quantile(nom, q=nbin))
H_act, y_edges = np.histogram(y,   bins=bin_by_quantile(y, q=nbin))

# 计算每个二维格子的 |res| 平均值
grid = np.zeros((len(y_edges)-1, len(x_edges)-1), dtype=float)  # rows: actual, cols: nominal
count = np.zeros_like(grid)

for i in range(len(x_edges)-1):
    cx = (nom >= x_edges[i]) & (nom < x_edges[i+1] if i < len(x_edges)-2 else nom <= x_edges[i+1])
    for j in range(len(y_edges)-1):
        ry = (y   >= y_edges[j]) & (y   < y_edges[j+1] if j < len(y_edges)-2 else y   <= y_edges[j+1])
        m = cx & ry
        if m.any():
            grid[j,i] = float(np.mean(abs_res[m]))
            count[j,i] = int(m.sum())

plt.figure(figsize=(6,5))
im = plt.imshow(grid, origin="lower", aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(x_edges)-1), [f"{x_edges[i]:.0f}" for i in range(len(x_edges)-1)], rotation=45)
plt.yticks(range(len(y_edges)-1), [f"{y_edges[i]:.0f}" for i in range(len(y_edges)-1)])
plt.xlabel("nominal_speed bin left-edge")
plt.ylabel("actual (running_speed) bin left-edge")
plt.title("2D 误差热图：mean(|residual|)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "error_heatmap_2d.png"))
plt.close()

print("[done] 输出目录：", os.path.abspath(OUTDIR))
for f in ["residual_hist.png","residual_vs_nominal.png","mae_by_nominal_bins.png","error_heatmap_2d.png"]:
    print(" -", f)
