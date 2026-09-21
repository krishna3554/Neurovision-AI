"""Training (FR-9 checkpoint recovery).

[REPORT] deep supervision, checkpoint save/restore, Kaggle training,
~145 epochs (Fig 4.6: Dice ~0.3 -> ~0.8, loss -> ~0.3-0.4).
[CHOICE] optimizer/schedule/augmentation/patch sampling defaults.

Usage:
    python training/train.py --root /path/to/ISLES-2022 --epochs 150
"""
import argparse
import json
import os

import torch

from backend.app.services.preprocess import build_preproc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.environ.get("CKPT_DIR", "training/checkpoints")


def make_items(ids, root):
    return [
        {
            "dwi": f"{root}/{s}/ses-0001/{s}_ses-0001_dwi.nii.gz",
            "adc": f"{root}/{s}/ses-0001/{s}_ses-0001_adc.nii.gz",
            "flair": f"{root}/{s}/ses-0001/{s}_ses-0001_flair.nii.gz",
            "label": f"{root}/derivatives/{s}/ses-0001/{s}_ses-0001_msk.nii.gz",
        }
        for s in ids
    ]


def main(root, epochs=150, bs=2, lr=2e-4, patch=(96, 96, 96)):
    from monai.data import CacheDataset, DataLoader
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )

    from backend.app.models.net import TriViewResMambaUNet

    os.makedirs(CKPT_DIR, exist_ok=True)
    split = json.load(open("data/splits.json"))
    pre = build_preproc(with_label=True)
    aug = Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=2,
                neg=1,
                num_samples=2,
            ),  # tri-view-guided weights can replace this
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        ]
    )
    tr = CacheDataset(make_items(split["train"], root), Compose([pre, aug]), cache_rate=0.5)
    va = CacheDataset(make_items(split["val"], root), pre, cache_rate=0.5)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=1)

    model = TriViewResMambaUNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = DiceCELoss(sigmoid=True)
    dice = DiceMetric(include_background=True, reduction="mean")
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")

    start, best = 0, 0.0
    last = f"{CKPT_DIR}/last.pt"
    if os.path.exists(last):  # resume interrupted training
        ck = torch.load(last, map_location=DEV)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best_dice"]
        print(f"Resumed from epoch {start}")

    history = []
    for ep in range(start, epochs):
        model.train()
        tot = 0
        for b in tl:
            x, y = b["image"].to(DEV), b["label"].to(DEV)
            opt.zero_grad()
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                out, aux = model(x)
                loss = loss_fn(out, y)
                for k, a in enumerate(aux):  # deep supervision
                    a = torch.nn.functional.interpolate(a, size=y.shape[2:], mode="trilinear")
                    loss = loss + (0.5 / (k + 1)) * loss_fn(a, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()

        model.eval()
        dice.reset()
        with torch.no_grad():
            for b in vl:
                x, y = b["image"].to(DEV), b["label"].to(DEV)
                p = sliding_window_inference(x, patch, 2, model, overlap=0.5)
                dice((torch.sigmoid(p) > 0.5).float(), y)
        d = dice.aggregate().item()
        history.append({"epoch": ep, "loss": tot / len(tl), "dice": d})
        ck = {
            "epoch": ep,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "sched": sched.state_dict(),
            "best_dice": max(best, d),
        }
        torch.save(ck, last)
        if d > best:
            best = d
            torch.save(ck, f"{CKPT_DIR}/best.pt")
        json.dump(history, open(f"{CKPT_DIR}/history.json", "w"))
        print(f"ep {ep} loss {tot / len(tl):.4f} val dice {d:.4f} best {best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/isles-2022")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    main(a.root, epochs=a.epochs, bs=a.bs, lr=a.lr)
