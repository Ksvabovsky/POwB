import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def iter_pairs(images_root: Path, masks_root: Path):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    for img_path in images_root.rglob("*"):
        if not img_path.is_file() or img_path.suffix not in exts:
            continue
        rel = img_path.relative_to(images_root)
        mask_path = masks_root / rel
        if mask_path.exists():
            yield img_path, mask_path, rel


def resize_max_side(img: np.ndarray, max_side: int, is_mask: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp)


def detect_face_rect(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def rect_from_face_or_center(image: np.ndarray):
    h, w = image.shape[:2]
    face = detect_face_rect(image)
    if face is None:
        # Simple fallback: central rectangle.
        rw = int(0.45 * w)
        rh = int(0.55 * h)
        rx = (w - rw) // 2
        ry = (h - rh) // 2
        return rx, ry, rw, rh

    x, y, fw, fh = face
    mx = int(0.25 * fw)
    my = int(0.35 * fh)
    rx = max(0, x - mx)
    ry = max(0, y - my)
    rw = min(w - rx, fw + 2 * mx)
    rh = min(h - ry, fh + 2 * my)
    return rx, ry, rw, rh


def segment_grabcut(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    rect = rect_from_face_or_center(image)

    gc_mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, gc_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    pred = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    # Light mask smoothing.
    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel, iterations=1)
    return pred


def to_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8)


def iou_dice(pred: np.ndarray, gt: np.ndarray):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    p = int(pred_b.sum())
    g = int(gt_b.sum())
    iou = 1.0 if union == 0 else inter / union
    dice = 1.0 if (p + g) == 0 else (2.0 * inter) / (p + g)
    return iou, dice, p, g


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def run(images_root: Path, masks_root: Path, output_root: Path, max_side: int):
    pred_root = output_root / "predicted_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    for img_path, gt_path, rel in iter_pairs(images_root, masks_root):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            skipped += 1
            continue

        image = resize_max_side(image, max_side, is_mask=False)
        gt = resize_max_side(gt, max_side, is_mask=True)

        pred = segment_grabcut(image)
        gt_bin = to_binary(gt)

        iou, dice, pred_pixels, gt_pixels = iou_dice(pred, gt_bin)

        pred_img = (pred * 255).astype(np.uint8)
        out_mask_path = pred_root / rel.with_suffix(".png")
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask_path), pred_img)

        rows.append(
            {
                "image": str(rel).replace("/", "\\"),
                "iou": f"{iou:.6f}",
                "dice": f"{dice:.6f}",
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "width": image.shape[1],
                "height": image.shape[0],
                "prediction_mask": str(out_mask_path).replace("/", "\\"),
            }
        )

    if not rows:
        raise RuntimeError("Nie znaleziono par obraz-maska.")

    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "iou", "dice", "pred_pixels", "gt_pixels", "width", "height", "prediction_mask"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["image"]))

    ious = [float(r["iou"]) for r in rows]
    dices = [float(r["dice"]) for r in rows]
    summary_path = output_root / "summary.txt"
    summary = [
        "Segmentation summary (OpenCV GrabCut)",
        f"images_processed={len(rows)}",
        f"images_skipped={skipped}",
        f"resize_max_side={max_side}",
        f"iou_avg={np.mean(ious):.6f}",
        f"iou_median={median(ious):.6f}",
        f"iou_min={np.min(ious):.6f}",
        f"iou_max={np.max(ious):.6f}",
        f"dice_avg={np.mean(dices):.6f}",
        f"dice_median={median(dices):.6f}",
        f"dice_min={np.min(dices):.6f}",
        f"dice_max={np.max(dices):.6f}",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    sorted_rows = sorted(rows, key=lambda r: float(r["iou"]))
    worst5 = sorted_rows[:5]
    best5 = list(reversed(sorted_rows[-5:]))

    print(f"Przetworzono obrazow: {len(rows)}")
    print(f"Pominieto obrazow: {skipped}")
    print(f"IoU srednie: {np.mean(ious):.4f}")
    print(f"Dice srednie: {np.mean(dices):.4f}")
    print(f"CSV: {metrics_path}")
    print(f"Podsumowanie: {summary_path}")
    print(f"Maski predykcji: {pred_root}")
    print("")
    print("Najgorsze 5 (IoU):")
    for r in worst5:
        print(f"- {r['image']} | IoU={float(r['iou']):.4f} | Dice={float(r['dice']):.4f}")
    print("")
    print("Najlepsze 5 (IoU):")
    for r in best5:
        print(f"- {r['image']} | IoU={float(r['iou']):.4f} | Dice={float(r['dice']):.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Prosta segmentacja twarzy gotowa metoda: OpenCV GrabCut.")
    p.add_argument("--images-root", type=Path, default=Path("data/images"))
    p.add_argument("--masks-root", type=Path, default=Path("data/masked_images"))
    p.add_argument("--output-root", type=Path, default=Path("output/segmentation"))
    p.add_argument("--max-side", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.images_root, args.masks_root, args.output_root, args.max_side)

