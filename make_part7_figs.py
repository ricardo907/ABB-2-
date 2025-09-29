#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# ---- 字体：尽力避免中文“方框” ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUTDIR = "figs_part7"
os.makedirs(OUTDIR, exist_ok=True)

def _pick_first(*cands):
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None

def load_meta_csv():
    p = _pick_first("train.fixed.csv", "train.csv")
    if p is None:
        raise FileNotFoundError("未找到 train.fixed.csv 或 train.csv")
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    if "running_speed" not in df.columns:
        raise ValueError("csv 缺少 running_speed 列")
    return df

def load_oof(name):
    p = _pick_first(f"oof_{name}.csv")
    if p is None:
        return None
    s = pd.read_csv(p)
    col = [c for c in s.columns if c.lower().startswith("oof")]
    if not col:
        raise ValueError(f"{p} 中未找到 oof 列")
    return s[col[0]].astype(float).values

def load_weights():
    """
    返回：order(list[str]), coef(np.array), intercept(float)
    优先 blend_weights.json；否则读 models/最终融合模型.meta.json
    """
    p1 = _pick_first("blend_weights.json")
    p2 = _pick_first(os.path.join("models","最终融合模型.meta.json"))

    if p1:
        j = json.load(open(p1,"r",encoding="utf-8"))
        if "order" in j and "weights_raw" in j:
            order = j["order"]
            coef = np.array([j["weights_raw"][k] for k in order], dtype=float)
            intercept = float(j.get("intercept", 0.0))
            return order, coef, intercept

    if p2:
        j = json.load(open(p2,"r",encoding="utf-8"))
        order = j["base_models_order"]
        coef  = np.array(j["coef_raw"], dtype=float)
        intercept = float(j.get("intercept", 0.0))
        return order, coef, intercept

    raise FileNotFoundError("未找到 blend_weights.json 或 models/最终融合模型.meta.json")

def metric(y, pred):
    return r2_score(y, pred), mean_absolute_error(y, pred)

def annotate_bars(ax, vals, fmt="{:.3f}", dy=0.01):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for i, v in enumerate(vals):
        ax.text(i, v + span*dy, fmt.format(v), ha="center", va="bottom", fontsize=10)

def plot_blend_weights(order, coef, intercept):
    w = np.array(coef, dtype=float)
    wn = w / (w.sum() + 1e-12)  # 仅展示归一化，便于讲解
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.bar(order, wn)
    ax.set_title("融合权重（归一化显示）")
    ax.set_ylabel("weight (normalized)")
    for i, (n, v) in enumerate(zip(order, wn)):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.text(0.02, 0.95, f"Intercept ≈ {intercept:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.7"))
    plt.tight_layout()
    out = os.path.join(OUTDIR, "blend_weights.png")
    plt.savefig(out); plt.close()
    print("  -", out)

def plot_metric_bar(metrics_dict):
    """
    metrics_dict: {name: (r2, mae)}
    生成一张图，左轴 R² 柱，右侧小字标注 MAE。
    """
    names = list(metrics_dict.keys())
    r2s   = [metrics_dict[k][0] for k in names]
    maes  = [metrics_dict[k][1] for k in names]

    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(111)
    bars = ax.bar(names, r2s)
    ax.set_ylim(0, max(1.0, max(r2s)*1.05))
    ax.set_ylabel("OOF R²")
    ax.set_title("OOF：单模 vs 融合 指标对比（R²；柱顶标注 MAE）")
    # 在柱顶标注 MAE
    for i,(b,m) in enumerate(zip(bars, maes)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"MAE={m:.1f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "metric_bar_single_vs_ens.png")
    plt.savefig(out); plt.close()
    print("  -", out)

def plot_ablation(stage_names, stage_r2s, stage_maes):
    """
    stage_* 长度一致，例如：
      stage_names = ["CAT", "CAT+CNN", "CAT+CNN+XGB"]
    """
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(111)
    bars = ax.bar(stage_names, stage_r2s)
    ax.set_ylim(0, max(1.0, max(stage_r2s)*1.05))
    ax.set_ylabel("OOF R²")
    ax.set_title("融合消融对比（柱顶标注 MAE）")
    for b,m in zip(bars, stage_maes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"MAE={m:.1f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "ablation_bar.png")
    plt.savefig(out); plt.close()
    print("  -", out)

def main():
    print("[part7] 读取数据 …")
    meta = load_meta_csv()
    y = meta["running_speed"].astype(float).values

    oofs = {}
    for name in ["cat","xgb","cnn"]:
        v = load_oof(name)
        if v is not None:
            if len(v) != len(y):
                raise ValueError(f"oof_{name}.csv 长度和训练集不一致：{len(v)} vs {len(y)}")
            oofs[name] = v

    if not oofs:
        raise RuntimeError("未找到任何 oof_*.csv，无法作图")

    # 权重与截距
    order, coef, intercept = load_weights()

    # 只用有 OOF 的模型参与重建融合 OOF
    used_names, used_coefs = [], []
    for n, c in zip(order, coef):
        if n in oofs:
            used_names.append(n)
            used_coefs.append(c)
    used_coefs = np.array(used_coefs, dtype=float)

    # 构造融合 OOF
    if len(used_names) == 0:
        raise RuntimeError("权重里所有基模型都没有 OOF 文件")
    mat = np.vstack([oofs[n] for n in used_names]).T
    oof_ens = mat.dot(used_coefs) + float(intercept)

    # 计算指标：各单模 + 融合
    metrics = {}
    for n, pred in oofs.items():
        metrics[n.upper()] = metric(y, pred)
    metrics["Ensemble"] = metric(y, oof_ens)

    print("[part7] 生成图片：")
    # 1) 融合权重（按原始顺序显示权重，如果某个模型没 OOF 也一并显示，便于讲解）
    plot_blend_weights(order, coef, intercept)

    # 2) 单模 vs 融合 指标条形图
    #    为了统一顺序：CAT / XGB / CNN / Ensemble（若不存在则自动跳过）
    disp_order = [n for n in ["CAT","XGB","CNN"] if n in metrics] + ["Ensemble"]
    metrics_sorted = {k: metrics[k] for k in disp_order}
    plot_metric_bar(metrics_sorted)

    # 3) 消融：按“权重从大到小”的模型逐步加入
    #    仅使用存在 OOF 的模型
    abl_pairs = sorted([(n, c) for n, c in zip(order, coef) if n in oofs],
                       key=lambda x: abs(x[1]), reverse=True)
    stage_names, stage_r2s, stage_maes = [], [], []
    if abl_pairs:
        acc_pred = np.zeros_like(y, dtype=float) + float(intercept)
        names_so_far = []
        for n, c in abl_pairs:
            acc_pred = acc_pred + c * oofs[n]
            names_so_far.append(n.upper())
            nm = "+".join(names_so_far)
            r2, m = metric(y, acc_pred)
            stage_names.append(nm)
            stage_r2s.append(r2)
            stage_maes.append(m)
        plot_ablation(stage_names, stage_r2s, stage_maes)
    else:
        print("[warn] 无可用于消融的子模型（可能缺少 OOF），跳过 ablation_bar.png")

    print(f"[done] 已输出到：{os.path.abspath(OUTDIR)}")
    print("      - blend_weights.png")
    print("      - metric_bar_single_vs_ens.png")
    print("      - ablation_bar.png（如有）")

if __name__ == "__main__":
    main()
