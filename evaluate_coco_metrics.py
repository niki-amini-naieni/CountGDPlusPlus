#!/usr/bin/env python3
import argparse, json, math, sys
from collections import defaultdict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    print("pycocotools is required. Install with: pip install pycocotools", file=sys.stderr)
    raise

STAT_NAMES = [
    "AP @[.50:.95]", "AP @0.50", "AP @0.75",
    "AP (small)", "AP (medium)", "AP (large)",
    "AR @1", "AR @10", "AR @100",
    "AR (small)", "AR (medium)", "AR (large)"
]

def print_stats(coco_eval, title=None):
    if title:
        print(f"\n{title}")
    coco_eval.summarize()
    for name, val in zip(STAT_NAMES, coco_eval.stats):
        print(f"{name:>14}: {val:.3f}")

def evaluate_coco(gt_json, pred_json, iou_type="bbox"):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate(); ev.accumulate(); print_stats(ev, "COCO Evaluation")
    return coco_gt, coco_dt

def compute_counting_errors(coco_gt, pred_json_path):
    # Sets for quick membership tests
    img_ids = sorted(coco_gt.getImgIds())
    cat_ids = sorted(coco_gt.getCatIds())
    img_set, cat_set = set(img_ids), set(cat_ids)

    # ---- GT counts per (image, category)
    gt_counts = defaultdict(int)
    for ann in coco_gt.dataset.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in img_set and cat_id in cat_set:
            gt_counts[(img_id, cat_id)] += 1

    # ---- Prediction counts per (image, category) 
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    pred_counts = defaultdict(int)
    for det in preds:
        img_id = det["image_id"]
        cat_id = det["category_id"]
        if img_id in img_set and cat_id in cat_set:
            pred_counts[(img_id, cat_id)] += 1

    # ---- Aggregate errors across ALL (image, category) pairs identified in GT json
    total_abs = 0.0
    total_sq = 0.0
    N = len(gt_counts.keys())
    for (img_id, cat_id) in gt_counts:
        diff = pred_counts[(img_id, cat_id)] - gt_counts[(img_id, cat_id)]
        e = abs(diff)
        total_abs += e
        total_sq += e * e

    mae = total_abs / N if N > 0 else float("nan")
    rmse = math.sqrt(total_sq / N) if N > 0 else float("nan")
    return mae, rmse, N, len(img_ids), len(cat_ids)

def main():
    p = argparse.ArgumentParser(description="COCO evaluation + counting error metrics")
    p.add_argument("--gt", required=True, help="Path to COCO ground-truth JSON (e.g., instances_val2017.json)")
    p.add_argument("--pred", required=True, help="Path to predictions JSON in COCO results format")
    p.add_argument("--iou-type", default="bbox", choices=["bbox", "segm", "keypoints"],
                   help="COCO evaluation type for AP/AR")
    p.add_argument("--no-coco-eval", action="store_true",
                   help="Skip COCO AP/AR evaluation and only compute counting errors")
    args = p.parse_args()

    if args.no_coco_eval:
        coco_gt = COCO(args.gt)
    else:
        coco_gt, _ = evaluate_coco(args.gt, args.pred, iou_type=args.iou_type)

    mae, rmse, N, n_img, n_cat = compute_counting_errors(coco_gt, args.pred)
    print("\nCounting Metrics (over ALL image × category pairs)")
    print(f"Images: {n_img} | Categories: {n_cat} | Pairs: {N}")
    print(f"Mean Absolute Counting Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Counting Error (RMSE): {rmse:.4f}")

if __name__ == "__main__":
    main()
