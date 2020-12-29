"""What the classifier does with shapes it was never trained on, and whether a simple check can tell.

The CNN always answers with one of its 7 classes; it has no "none of these". This script measures how sure it is on the 2,000
held-out images (shapes it knows) and on 1,200 images of six shapes it never saw (octagon, the stop-sign shape; heptagon;
ellipse; plus sign; arrow; crescent), drawn with the same backgrounds, colours and noise as the training images. It then
scores one rule for flagging an unfamiliar input: trust the answer only when the CNN is at least 90% sure AND the contour
geometry (src/traditional.py, on the thresholded mask) names the same shape.

    python ood_check.py [<lab_output_dir>]

With an output directory it also writes, for the portfolio Lab's parity check: geometry.json (Python's threshold-path results
on the Lab's 63 held-out PNGs and on 12 unfamiliar ones) and the 12 unfamiliar images as PNG.
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import random_split

from src.dataset import SHAPE_CLASSES, ShapeDataset
from src.traditional import classify_shape_by_geometry, preprocess_image

MODEL = os.path.join("models", "shape_detector.onnx")
UNSEEN = ["octagon", "heptagon", "ellipse", "plus", "arrow", "crescent"]
PER_UNSEEN = 200
CONFIDENT = 0.90
SIZE = 64


def unseen_image(kind: str) -> np.ndarray:
    """One 64 x 64 image of a shape the model never saw, with src/dataset.py's background, colour and noise recipe."""
    bg = random.randint(220, 255)
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * bg
    if random.random() < 0.3:
        for y in range(SIZE):
            img[y] = np.clip(img[y] * (1.0 - 0.15 * (y / SIZE)), 0, 255).astype(np.uint8)
    color = tuple(random.randint(0, 130) for _ in range(3))
    cx, cy = random.randint(int(SIZE * 0.4), int(SIZE * 0.6)), random.randint(int(SIZE * 0.4), int(SIZE * 0.6))
    r = random.randint(int(SIZE * 0.16), int(SIZE * 0.32))
    angle = random.randint(0, 360)

    def poly(points):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cv2.fillPoly(img, [cv2.transform(pts, m).reshape(-1, 2).astype(np.int32)], color)

    if kind in ("octagon", "heptagon"):
        n = 8 if kind == "octagon" else 7
        poly([[cx + r * np.cos(i * 2 * np.pi / n), cy + r * np.sin(i * 2 * np.pi / n)] for i in range(n)])
    elif kind == "ellipse":
        cv2.ellipse(img, (cx, cy), (r, int(r * random.uniform(0.45, 0.65))), angle, 0, 360, color, -1)
    elif kind == "plus":
        t = r * 0.35
        poly([[cx - t, cy - r], [cx + t, cy - r], [cx + t, cy - t], [cx + r, cy - t], [cx + r, cy + t], [cx + t, cy + t],
              [cx + t, cy + r], [cx - t, cy + r], [cx - t, cy + t], [cx - r, cy + t], [cx - r, cy - t], [cx - t, cy - t]])
    elif kind == "arrow":
        t = r * 0.3
        poly([[cx - r, cy - t], [cx, cy - t], [cx, cy - r], [cx + r, cy], [cx, cy + r], [cx, cy + t], [cx - r, cy + t]])
    elif kind == "crescent":
        cv2.circle(img, (cx, cy), r, color, -1)
        shift = int(r * 0.55)
        ox, oy = int(cx + shift * np.cos(np.radians(angle))), int(cy + shift * np.sin(np.radians(angle)))
        cv2.circle(img, (ox, oy), int(r * 0.85), (bg, bg, bg), -1)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() > 0.6:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img  # treated as RGB: the colour is random per channel, so the channel order does not matter


def geometry(rgb: np.ndarray) -> dict:
    """The threshold path of src/traditional.py: grayscale, 5 x 5 blur, Otsu, close + open, then the largest contour's shape."""
    mask = preprocess_image(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"label": "unknown", "vertices": 0, "area": 0.0, "contours": 0, "white": int((mask > 0).sum())}
    big = max(contours, key=cv2.contourArea)
    label, approx = classify_shape_by_geometry(big)
    return {"label": label, "vertices": int(len(approx)), "area": float(cv2.contourArea(big)), "contours": len(contours),
            "white": int((mask > 0).sum())}


def cnn(session, name, rgb: np.ndarray) -> np.ndarray:
    logits = session.run(None, {name: np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]})[0][0]
    p = np.exp(logits - logits.max())
    return p / p.sum()


def main(out_dir=None) -> None:
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    full = ShapeDataset(os.path.join("data", "shapes"))
    train_size = int(0.8 * len(full))
    _, val = random_split(full, [train_size, len(full) - train_size], generator=torch.Generator().manual_seed(42))
    known, pieces, share = [], [], []
    for idx in val.indices:
        item = full.annotations[idx]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(full.images_dir, item["filename"])), cv2.COLOR_BGR2RGB)
        p, g = cnn(session, name, rgb), geometry(rgb)
        known.append((item["label_name"], SHAPE_CLASSES[int(p.argmax())], float(p.max()), g["label"]))
        pieces.append(g["contours"])
        share.append(g["area"] / max(g["white"], 1))

    random.seed(7)
    np.random.seed(7)
    unseen = {k: [] for k in UNSEEN}
    images = {k: [] for k in UNSEEN}
    for k in UNSEEN:
        for _ in range(PER_UNSEEN):
            rgb = unseen_image(k)
            p = cnn(session, name, rgb)
            unseen[k].append((SHAPE_CLASSES[int(p.argmax())], float(p.max()), geometry(rgb)["label"]))
            images[k].append(rgb)

    def trusted(pred, conf, geo):
        return conf >= CONFIDENT and geo == pred

    kc = np.array([c for _, _, c, _ in known])
    print(f"Held-out (known shapes), n={len(known)}")
    print(f"  CNN at least {CONFIDENT:.0%} sure: {np.mean(kc >= CONFIDENT):.1%}; median confidence {np.median(kc):.4f}")
    acc = [t == p for t, p, c, g in known if trusted(p, c, g)]
    print(f"  trusted by the rule: {np.mean([trusted(p, c, g) for _, p, c, g in known]):.1%}, correct when trusted: {np.mean(acc):.2%}")
    print(f"  separate pieces in the mask: at most {max(pieces)}, {np.mean(np.array(pieces) == 1):.1%} have exactly 1; "
          f"largest piece's area / white pixels: 1st percentile {np.percentile(share, 1):.3f}")
    print(f"Unseen shapes, {PER_UNSEEN} each")
    print("  shape | CNN's most common answer | CNN >= 90% sure | trusted by the rule")
    allc, allt = [], []
    for k in UNSEEN:
        preds = [p for p, _, _ in unseen[k]]
        common = max(set(preds), key=preds.count)
        conf = np.array([c for _, c, _ in unseen[k]])
        t = [trusted(p, c, g) for p, c, g in unseen[k]]
        allc += list(conf >= CONFIDENT)
        allt += t
        print(f"  {k} | {common} ({preds.count(common) / len(preds):.0%}) | {np.mean(conf >= CONFIDENT):.1%} | {np.mean(t):.1%}")
    print(f"  all unseen | | {np.mean(allc):.1%} | {np.mean(allt):.1%}")

    if out_dir:
        lab = json.load(open(os.path.join(out_dir, "samples.json")))
        held = []
        for s in lab["samples"]:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(out_dir, s["file"])), cv2.COLOR_BGR2RGB)
            held.append({"file": s["file"], "geometry": geometry(rgb)})
        extra = []
        for k in UNSEEN:
            for i in range(2):
                rgb = images[k][i]
                fname = f"unseen_{k}_{i}.png"
                cv2.imwrite(os.path.join(out_dir, fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                p = cnn(session, name, rgb)
                extra.append({"file": fname, "label": k, "probs": [round(float(v), 6) for v in p],
                              "predicted": SHAPE_CLASSES[int(p.argmax())], "geometry": geometry(rgb)})
        with open(os.path.join(out_dir, "geometry.json"), "w") as f:
            json.dump({"note": "src/traditional.py threshold path: preprocess_image, largest external contour, "
                               "classify_shape_by_geometry. Written by ood_check.py.",
                       "confident": CONFIDENT, "held_out_max_pieces": int(max(pieces)), "held_out": held, "unseen": extra}, f, indent=1)
        print(f"geometry.json and {len(extra)} unseen images written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
