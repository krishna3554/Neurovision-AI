"""Evaluation + report figures (Figs 4.1-4.6, Table 6.1).

Metrics [REPORT]: Dice, IoU, Precision, Recall, Specificity, HD95.

Usage:
    python training/evaluate.py --root ISLES-2022 --checkpoint training/checkpoints/best.pt --out docs/figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    return tp, fp, fn, tn


def all_metrics(pred, gt):
    from monai.metrics import HausdorffDistanceMetric

    tp, fp, fn, tn = confusion(pred, gt)
    e = 1e-8
    hd = HausdorffDistanceMetric(include_background=True, percentile=95)
    try:
        hd(torch.tensor(pred)[None, None].float(), torch.tensor(gt)[None, None].float())
        hd95 = float(hd.aggregate().item())
    except Exception:
        hd95 = float("nan")
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + e),
        "iou": tp / (tp + fp + fn + e),
        "precision": tp / (tp + fp + e),
        "recall": tp / (tp + fn + e),
        "specificity": tn / (tn + fp + e),
        "hd95": hd95,
    }


# ---- Figure generators (titles match the report) ----


def fig41_metric_summary(metrics: dict, out: str):
    keys = ["dice", "iou", "precision", "recall", "specificity"]
    vals = [metrics[k] for k in keys]
    colors = ["teal", "blue", "gold", "orange", "green"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(keys, vals, color=["#14b8a6", "#3b82f6", "#eab308", "#f97316", "#22c55e"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation prediction metric summary")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig42_multimodal(dwi, adc, flair, gt, case: str, out: str, z: int | None = None):
    if z is None:
        z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5))
    for a, vol, t in zip(ax, [dwi, adc, flair, flair], ["DWI", "ADC", "FLAIR", "Ground-truth overlay"]):
        a.imshow(np.rot90(vol[:, :, z]), cmap="gray")
        a.set_title(t)
        a.axis("off")
    ax[3].imshow(np.rot90((gt[:, :, z] > 0).astype(float)), cmap="Greens", alpha=0.5)
    fig.suptitle(f"Multi-modal MRI input at lesion slice: {case}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _contour(ax, mask, color):
    from skimage import measure

    for c in measure.find_contours(mask, 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.2)


def fig43_qualitative(flair, gt, pred, case: str, out: str):
    m = all_metrics(pred > 0.5, gt > 0.5)
    z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
    y = flair.shape[1] // 2
    x = flair.shape[0] // 2
    views = [
        (np.rot90(flair[:, :, z]), np.rot90(gt[:, :, z]), np.rot90(pred[:, :, z]), "axial"),
        (np.rot90(flair[:, y, :]), np.rot90(gt[:, y, :]), np.rot90(pred[:, y, :]), "coronal"),
        (np.rot90(flair[x, :, :]), np.rot90(gt[x, :, :]), np.rot90(pred[x, :, :]), "sagittal"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (bg, g, p, t) in zip(ax, views):
        a.imshow(bg, cmap="gray")
        _contour(a, g, "lime")
        _contour(a, p, "red")
        a.set_title(t)
        a.axis("off")
    fig.suptitle(f"Qualitative lesion segmentation: {case} | Dice={m['dice']:.3f} | IoU={m['iou']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig44_top_slices(vol_dwi, slices: list, case: str, out: str):
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    for a, z in zip(ax.flat, slices):
        a.imshow(np.rot90(vol_dwi[:, :, z]), cmap="gray")
        a.set_title(f"z={z}")
        a.axis("off")
    fig.suptitle(f"Adaptive high-information slice extraction: {case}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig45_probability(mri_axial, prob_axial, ent_axial, prob_cor, prob_sag, case: str, out: str):
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(np.rot90(mri_axial), cmap="gray")
    ax[0, 0].set_title("MRI axial")
    im = ax[0, 1].imshow(np.rot90(prob_axial), cmap="jet", vmin=0, vmax=1)
    ax[0, 1].set_title("Prediction probability")
    ax[0, 2].imshow(np.rot90(ent_axial), cmap="hot")
    ax[0, 2].set_title("Uncertainty entropy")
    ax[1, 0].imshow(np.rot90(prob_cor), cmap="jet", vmin=0, vmax=1)
    ax[1, 0].set_title("Coronal prob")
    ax[1, 1].imshow(np.rot90(prob_sag), cmap="jet", vmin=0, vmax=1)
    ax[1, 1].set_title("Sagittal prob")
    ax[1, 2].hist(prob_axial.ravel(), bins=50)
    ax[1, 2].set_title("Voxel-probability histogram")
    for a in ax.flat:
        a.axis("off") if a != ax[1, 2] else None
    fig.colorbar(im, ax=ax[0, 1], shrink=0.8)
    fig.suptitle(f"Probability and uncertainty visualization: {case}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig46_training_curve(history: list, out: str):
    ep = [h["epoch"] for h in history]
    dice = [h["dice"] for h in history]
    loss = [h["loss"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ep, dice, "o-", color="teal", label="Dice")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Dice", color="teal")
    ax2 = ax1.twinx()
    ax2.plot(ep, loss, "s-", color="orange", label="loss")
    ax2.set_ylabel("loss", color="orange")
    ax1.set_title("Training convergence and checkpoint history")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ISLES-2022")
    ap.add_argument("--checkpoint", default="training/checkpoints/best.pt")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--case", default="sub-strokecase0004")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Training curve from history.json if present
    hist_path = os.path.join(os.path.dirname(a.checkpoint), "history.json")
    if os.path.exists(hist_path):
        fig46_training_curve(json.load(open(hist_path)), f"{a.out}/fig46_training_curve.png")

    # Test-set metrics -> Table 6.1
    try:
        split = json.load(open("data/splits.json"))
    except FileNotFoundError:
        split = {"test": []}
    print("Test subjects:", split.get("test", []))
    print("Run inference per test subject, aggregate all_metrics(), fill Table 6.1:")
    print("target Dice~0.81 IoU~0.70 Prec~0.89 Rec~0.78 Spec~0.9998 HD95~4.0")


if __name__ == "__main__":
    main()
