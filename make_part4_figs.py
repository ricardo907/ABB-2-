#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 统一图像风格
plt.rcParams["figure.dpi"] = 150
mpl.rcParams["axes.unicode_minus"] = False  # 负号不显示方块

# ========= 中文字体启用（新增）=========
def enable_chinese_font(preferred=None, font_path=None, fonts_dir="fonts"):
    """
    优先顺序：
    1) 传入的 font_path
    2) 项目 fonts/ 目录下的第一款 .ttf/.otf
    3) 系统已安装的常见中文字体列表（preferred）
    """
    candidates = preferred or [
        "Microsoft YaHei",          # Windows 微软雅黑
        "SimHei",                   # 黑体
        "PingFang SC",              # macOS 苹方
        "Hiragino Sans GB",         # 冬青黑体
        "Source Han Sans SC",       # 思源黑体
        "Noto Sans CJK SC",         # Noto 中文
        "WenQuanYi Micro Hei",      # 文泉驿微米黑
    ]

    # 1) 指定路径
    use_name = None
    if font_path and os.path.isfile(font_path):
        font_manager.fontManager.addfont(font_path)
        use_name = font_manager.FontProperties(fname=font_path).get_name()

    # 2) 项目内 fonts/
    if use_name is None and os.path.isdir(fonts_dir):
        for f in os.listdir(fonts_dir):
            if f.lower().endswith((".ttf", ".otf")):
                fp = os.path.join(fonts_dir, f)
                try:
                    font_manager.fontManager.addfont(fp)
                    use_name = font_manager.FontProperties(fname=fp).get_name()
                    break
                except Exception:
                    pass

    # 3) 系统常见字体
    if use_name is None:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                use_name = name
                break

    if use_name:
        mpl.rcParams["font.sans-serif"] = [use_name]
        print(f"[font] 使用中文字体：{use_name}")
    else:
        print("[font] 未找到中文字体，将使用英文标题以避免方块。")


# ========= 现有工具函数 =========
def _pick_first_exist(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def load_inputs():
    # 训练元数据（优先使用对齐后的）
    meta_path = _pick_first_exist([
        "train.fixed.csv",   # 主脚本生成
        "train.csv"
    ])
    if meta_path is None:
        raise FileNotFoundError("未找到 train.fixed.csv 或 train.csv")

    # 特征缓存（主脚本生成）
    feat_path = _pick_first_exist([
        "cache_X_train_ext.parquet",
        "cache_X_train.parquet",
        "cache_X_train_ext.csv",
        "cache_X_train.csv"
    ])
    if feat_path is None:
        raise FileNotFoundError("未找到特征缓存（cache_X_train*.parquet/csv）——请先运行主训练脚本")

    # 读取
    meta = pd.read_csv(meta_path)
    if feat_path.endswith(".parquet"):
        X = pd.read_parquet(feat_path)
    else:
        X = pd.read_csv(feat_path)

    # 关键列检查
    need = {"running_speed","nominal_speed"}
    low = set(map(str.lower, meta.columns))
    if not need.issubset(low):
        raise ValueError("train(.fixed).csv 缺少 running_speed/nominal_speed 列")

    # 统一列名小写访问
    meta.columns = [c.lower() for c in meta.columns]
    return meta, X, os.path.abspath("figs_part4")

def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

# ========= 出图函数 =========
def fig_target_hist(meta, outdir):
    y = pd.to_numeric(meta["running_speed"], errors="coerce").values.astype(float)
    y = np.nan_to_num(y, nan=0.0)
    zeros = int((y == 0).sum())
    nonzeros = int((y != 0).sum())

    plt.figure()
    # 自适应上限，避免长尾压扁图形（排除最极端top 1%后确定显示上限）
    if (y > 0).any():
        upper = float(np.quantile(y[y > 0], 0.99))
    else:
        upper = 1.0
    bins = 50
    plt.hist(np.clip(y, 0, upper), bins=bins)
    title_cn = f"Running Speed 分布（含0） | zeros={zeros}, non-zeros={nonzeros}"
    plt.title(title_cn)
    plt.xlabel("running_speed")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "target_hist.png"), bbox_inches="tight")
    plt.close()

def fig_target_box_by_nominal(meta, outdir, n_bins=10):
    y = pd.to_numeric(meta["running_speed"], errors="coerce").values.astype(float)
    nom = pd.to_numeric(meta["nominal_speed"], errors="coerce").values.astype(float)
    y = np.nan_to_num(y, nan=0.0)
    nom = np.nan_to_num(nom, nan=0.0)

    # 按 nominal 分箱（分位数），每箱做一个 box
    q = np.linspace(0, 1, n_bins+1)
    edges = np.unique(np.quantile(nom, q))
    # 保障至少2箱
    if len(edges) < 3:
        edges = np.unique(np.linspace(nom.min(), nom.max(), 4))

    groups, labels = [], []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        m = (nom >= lo) & (nom <= hi if i == len(edges)-2 else nom < hi)
        if m.sum() == 0:
            continue
        groups.append(y[m])
        labels.append(f"[{lo:.0f},{hi:.0f}]")

    plt.figure(figsize=(max(6, len(groups)*0.6), 4))
    plt.boxplot(groups, showfliers=False)
    plt.xticks(range(1, len(labels)+1), labels, rotation=30)
    plt.title("Running Speed 按 nominal_speed 分箱的箱线图（含0样本）")
    plt.xlabel("nominal_speed bins")
    plt.ylabel("running_speed")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "target_box_by_nominal.png"), bbox_inches="tight")
    plt.close()

def fig_corr_heatmap_topk(X, outdir, k=50):
    # 选择方差最大的 K 个数值特征
    num_cols = X.select_dtypes(include=[np.number]).copy()
    num_cols = num_cols.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    var = num_cols.var().sort_values(ascending=False)
    top_cols = var.index[:min(k, len(var))]

    corr = num_cols[top_cols].corr().values

    plt.figure(figsize=(6,5))
    im = plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"特征相关性热力图（Top-{len(top_cols)} 方差特征）")
    # 只标记稀疏的刻度，避免过密
    step = max(1, len(top_cols)//10)
    idxs = np.arange(0, len(top_cols), step)
    plt.xticks(idxs, [top_cols[i] for i in idxs], rotation=90, fontsize=7)
    plt.yticks(idxs, [top_cols[i] for i in idxs], fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_heatmap_top50.png"), bbox_inches="tight")
    plt.close()

def fig_some_feature_hists(X, outdir, num=6):
    # 挑几个直观的统计量画分布
    candidates = [c for c in X.columns if any(k in c.lower() for k in ["_rms","_std","_spectral_centroid","_spectral_entropy","corr_","_peak1_mag"])]
    cols = [c for c in candidates if np.issubdtype(X[c].dtype, np.number)]
    cols = cols[:min(num, len(cols))]
    if not cols:
        return

    for c in cols:
        v = pd.to_numeric(X[c], errors="coerce").values
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        upper = np.quantile(v, 0.99)
        plt.figure()
        plt.hist(np.clip(v, v.min(), upper), bins=50)
        plt.title(f"特征分布: {c}")
        plt.xlabel(c)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"feat_hist_{c}.png"), bbox_inches="tight")
        plt.close()

# ========= 主流程 =========
def main():
    # 优先使用项目 fonts/ 中的字体；没有就自动找系统中文字体
    # 可按需改成：enable_chinese_font(font_path="fonts/SourceHanSansSC-Regular.otf")
    enable_chinese_font()

    meta, X, outdir = load_inputs()
    ensure_outdir(outdir)
    print("[OK] 数据载入完成。输出目录：", outdir)

    fig_target_hist(meta, outdir)
    fig_target_box_by_nominal(meta, outdir)
    fig_corr_heatmap_topk(X, outdir, k=50)
    fig_some_feature_hists(X, outdir, num=6)

    print("已生成：")
    for f in ["target_hist.png","target_box_by_nominal.png","corr_heatmap_top50.png"]:
        print("  -", os.path.join(outdir, f))
    print("以及若干 feat_hist_*.png（若有筛中候选特征）")

if __name__ == "__main__":
    main()
