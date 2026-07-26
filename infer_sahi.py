#!/usr/bin/env python3
"""
SAHI (Slicing Aided Hyper Inference) wrapper for CountGD++.

Large images (e.g. drone orthophotos, satellite tiles) contain objects that are
too small to detect reliably when the full image is downscaled to the model's
input resolution (~800 px). This script slices each image into overlapping
square tiles, runs CountGD++ on every tile, maps the predicted boxes back to
original-image coordinates, and merges cross-tile duplicates with NMS.

Usage — same prompts format as process_folder.py:

  python infer_sahi.py \\
      --input_folder   path/to/images/ \\
      --prompts        path/to/prompts.json \\
      --output_folder  path/to/output/ \\
      --pretrain_model_path checkpoints/countgd_plusplus.pth \\
      --patch_size 1024 \\
      --overlap    200  \\
      --vis_output

Prompts JSON format (identical to process_folder.py):

  {
    "positive": {
      "text": "object name",
      "exemplars": {
        "image": "path/to/exemplar_image.jpg",
        "boxes": [[x1, y1, x2, y2], ...]
      }
    },
    "negative": []
  }
"""

import argparse
import glob
import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms as tv_nms

from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms_app as T


# ---------------------------------------------------------------------------
# Model setup  (shared with process_folder.py)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run CountGD++ on a folder of images using SAHI "
                    "(Slicing Aided Hyper Inference) for large-image support."
    )
    p.add_argument("--input_folder",  type=str, required=True,
                   help="Folder containing images to process.")
    p.add_argument("--prompts",       type=str, required=True,
                   help="Path to JSON file with positive/negative prompts "
                        "(same format as process_folder.py).")
    p.add_argument("--output_folder", type=str, required=True,
                   help="Folder where results and visualisations are saved.")
    p.add_argument("--pretrain_model_path", type=str,
                   default="checkpoints/countgd_plusplus.pth",
                   help="Path to CountGD++ checkpoint.")
    p.add_argument("--conf_thresh", type=float, default=0.23,
                   help="Confidence threshold (default: 0.23). "
                        "Increase to reduce false positives.")
    p.add_argument("--vis_output", action="store_true",
                   help="Save annotated images with predicted bounding boxes.")

    # SAHI parameters
    p.add_argument("--patch_size", type=int, default=1024,
                   help="Tile size in pixels (default: 1024). Each image is "
                        "divided into square tiles of this size. Smaller tiles "
                        "give higher effective resolution at the cost of more "
                        "inference passes.")
    p.add_argument("--overlap", type=int, default=200,
                   help="Overlap between adjacent tiles in pixels (default: 200). "
                        "Larger overlap reduces missed detections at tile edges.")
    p.add_argument("--nms_iou", type=float, default=0.5,
                   help="IoU threshold for NMS used to merge detections from "
                        "overlapping tiles (default: 0.5).")

    # Model / distributed boilerplate (mirrors process_folder.py so the same
    # checkpoint loading code works unchanged)
    p.add_argument("--device", default="cuda")
    p.add_argument("--options", nargs="+", action=DictAction,
                   help="Override config values via key=value pairs.")
    p.add_argument("--remove_difficult", action="store_true")
    p.add_argument("--fix_size",         action="store_true")
    p.add_argument("--note",             default="")
    p.add_argument("--resume",           default="")
    p.add_argument("--finetune_ignore",  type=str, nargs="+")
    p.add_argument("--start_epoch",      default=0, type=int)
    p.add_argument("--eval",             action="store_false")
    p.add_argument("--num_workers",      default=8, type=int)
    p.add_argument("--test",             action="store_true")
    p.add_argument("--debug",            action="store_true")
    p.add_argument("--find_unused_params", action="store_true")
    p.add_argument("--save_results",     action="store_true")
    p.add_argument("--save_log",         action="store_true")
    p.add_argument("--world_size",       default=1, type=int)
    p.add_argument("--dist_url",         default="env://")
    p.add_argument("--rank",             default=0, type=int)
    p.add_argument("--local_rank",       type=int)
    p.add_argument("--local-rank",       type=int)
    p.add_argument("--amp",              action="store_true")
    return p


def build_model_and_transforms(args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data_transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])

    cfg = SLConfig.fromfile("cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    for k, v in cfg._cfg_dict.to_dict().items():
        if k not in vars(args):
            setattr(args, k, v)
        else:
            raise ValueError(f"Key {k} already used by args.")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    from models.GroundingDINO import groundingdino_app
    model, _, _ = groundingdino_app.build_groundingdino(args)
    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, data_transform


# ---------------------------------------------------------------------------
# Prompt loading  (mirrors process_folder.py)
# ---------------------------------------------------------------------------

def _box_points(boxes: List[List[float]]) -> List[List[float]]:
    """Convert [[x1,y1,x2,y2],...] to gradio image-prompter box format."""
    return [[x1, y1, 2.0, x2, y2, 3.0] for x1, y1, x2, y2 in boxes]


def load_prompts(prompts_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Return (positive_prompts, negative_prompts_list).

    positive_prompts keys: text, image (PIL), points (gradio format)
    negative_prompts_list: list of dicts with same keys
    """
    with open(prompts_path) as f:
        raw = json.load(f)

    def _build(entry: Dict) -> Dict:
        img = Image.open(entry["exemplars"]["image"]).convert("RGB")
        return {
            "text":   entry["text"],
            "image":  img,
            "points": _box_points(entry["exemplars"]["boxes"]),
        }

    pos = _build(raw["positive"])
    negs = [_build(n) for n in raw.get("negative", [])]
    return pos, negs


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _get_box_inputs(points: List[List[float]]) -> List[List[float]]:
    return [[p[0], p[1], p[3], p[4]] for p in points if p[2] == 2.0 and p[5] == 3.0]


def _preprocess(transform, image: Image.Image, prompts: Dict):
    input_image, _ = transform(image, None)
    exemplar_boxes = _get_box_inputs(prompts["points"])
    input_exemplar, exemplar = transform(
        prompts["image"],
        {"exemplars": torch.tensor(exemplar_boxes, dtype=torch.float)},
    )
    return input_image, input_exemplar, exemplar["exemplars"]


def _run_model(model, transform, image: Image.Image,
               pos_prompts: Dict, neg_prompts_list: List[Dict],
               device: torch.device, conf_thresh: float, caption: str):
    """Run one forward pass; return (norm_boxes [N,4], scores [N]) as numpy."""
    input_image, input_exemplar, pos_exemplar = _preprocess(transform, image, pos_prompts)

    neg_images, neg_exemplars = [], []
    for neg in neg_prompts_list:
        _, neg_ex_img, neg_ex = _preprocess(transform, image, neg)
        neg_images.append(
            nested_tensor_from_tensor_list(neg_ex_img.unsqueeze(0).to(device))
        )
        neg_exemplars.append([neg_ex.to(device)])

    with torch.no_grad():
        out = model(
            nested_tensor_from_tensor_list(input_image.unsqueeze(0).to(device)),
            nested_tensor_from_tensor_list(input_exemplar.unsqueeze(0).to(device)),
            [pos_exemplar.to(device)],
            neg_images,
            neg_exemplars,
            captions=[caption],
        )

    input_ids = out["input_ids"][0]
    logits = out["pred_logits"].sigmoid()[0]
    boxes  = out["pred_boxes"][0]

    # Find first "." separator token (id 1012)
    sep_idx = next(
        (i for i, t in enumerate(input_ids) if int(t) == 1012),
        logits.shape[1],
    )
    pos_logits = logits[:, : sep_idx + 1]
    neg_logits = logits[:, sep_idx + 1 :]

    # Stage-1: confidence threshold
    pos_scores = pos_logits.max(dim=-1).values
    mask = pos_scores > conf_thresh

    # Stage-2: positive score must beat negative score when negatives present
    if neg_logits.shape[1] > 0:
        neg_scores = neg_logits.max(dim=-1).values
        mask = mask & (pos_scores > neg_scores)

    sel_boxes  = boxes[mask].cpu().numpy()   # normalized (cx, cy, w, h)
    sel_scores = pos_scores[mask].cpu().numpy()
    return sel_boxes, sel_scores


# ---------------------------------------------------------------------------
# SAHI core
# ---------------------------------------------------------------------------

def slice_image(
    image: Image.Image, patch_size: int, overlap: int
) -> List[Tuple[Image.Image, int, int]]:
    """
    Slice *image* into overlapping square tiles.

    Returns a list of (tile, x_offset, y_offset) tuples.  Edge tiles are
    shifted inward so every tile is exactly patch_size × patch_size — no
    padding is added, which keeps the model's statistics consistent.
    """
    W, H = image.size
    stride = patch_size - overlap
    slices = []
    y = 0
    while y < H:
        y_end   = min(y + patch_size, H)
        y_start = y_end - patch_size if (y_end == H and y_end - y < patch_size) else y
        x = 0
        while x < W:
            x_end   = min(x + patch_size, W)
            x_start = x_end - patch_size if (x_end == W and x_end - x < patch_size) else x
            slices.append((image.crop((x_start, y_start, x_end, y_end)), x_start, y_start))
            if x_end == W:
                break
            x += stride
        if y_end == H:
            break
        y += stride
    return slices


def predict_sahi(
    model, transform, image: Image.Image,
    pos_prompts: Dict, neg_prompts_list: List[Dict],
    device: torch.device, conf_thresh: float, caption: str,
    patch_size: int, overlap: int, nms_iou: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SAHI inference: slice → predict per tile → map back → NMS.

    Returns (boxes, scores) where boxes are absolute (x1, y1, x2, y2) in the
    original image coordinate system.
    """
    W, H = image.size
    tiles = slice_image(image, patch_size, overlap)
    logging.info("  SAHI: %d tiles (%dx%d, overlap %d)", len(tiles), patch_size, patch_size, overlap)

    all_boxes:  List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for tile, x_off, y_off in tiles:
        pw, ph = tile.size
        norm_boxes, scores = _run_model(
            model, transform, tile, pos_prompts, neg_prompts_list,
            device, conf_thresh, caption,
        )
        if len(norm_boxes) == 0:
            continue

        # Convert normalized (cx, cy, w, h) → absolute (x1, y1, x2, y2)
        cx = norm_boxes[:, 0] * pw + x_off
        cy = norm_boxes[:, 1] * ph + y_off
        bw = norm_boxes[:, 2] * pw
        bh = norm_boxes[:, 3] * ph
        x1 = np.clip(cx - bw / 2, 0, W)
        y1 = np.clip(cy - bh / 2, 0, H)
        x2 = np.clip(cx + bw / 2, 0, W)
        y2 = np.clip(cy + bh / 2, 0, H)

        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0)

    boxes_cat  = np.concatenate(all_boxes,  axis=0)
    scores_cat = np.concatenate(all_scores, axis=0)

    # Cross-tile NMS
    keep = tv_nms(
        torch.from_numpy(boxes_cat).float(),
        torch.from_numpy(scores_cat).float(),
        nms_iou,
    ).numpy()

    return boxes_cat[keep], scores_cat[keep]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_boxes_on_image(
    image: Image.Image,
    boxes: np.ndarray,   # (N, 4) absolute x1,y1,x2,y2
) -> Image.Image:
    """Draw green bounding boxes; returns a copy."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    W, _ = vis.size
    lw = max(2, W // 600)
    for x1, y1, x2, y2 in boxes.astype(int):
        draw.rectangle([x1, y1, x2, y2], outline=(50, 220, 50), width=lw)
    return vis


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp",
            "*.tif", "*.tiff", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF")
    imgs: List[str] = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(folder, e)))
    return sorted(imgs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("Patch size: %d  Overlap: %d  NMS IoU: %.2f",
                 args.patch_size, args.overlap, args.nms_iou)

    model, transform = build_model_and_transforms(args)
    model = model.to(device)

    pos_prompts, neg_prompts_list = load_prompts(args.prompts)

    # Build caption: "positive text . negative text . negative text . "
    caption = pos_prompts["text"] + " . "
    for neg in neg_prompts_list:
        caption += neg["text"] + " . "
    logging.info("Caption: %s", caption)

    images = list_images(args.input_folder)
    logging.info("Found %d images in %s", len(images), args.input_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    results: Dict[str, Any] = {}

    for idx, file_path in enumerate(images):
        file_name = os.path.basename(file_path)
        logging.info("Processing image %d/%d: %s", idx + 1, len(images), file_name)

        image = Image.open(file_path).convert("RGB")
        W, H  = image.size
        logging.info("  Image size: %dx%d", W, H)

        boxes, scores = predict_sahi(
            model, transform, image,
            pos_prompts, neg_prompts_list,
            device, args.conf_thresh, caption,
            args.patch_size, args.overlap, args.nms_iou,
        )

        count = len(boxes)
        logging.info("  Count: %d", count)

        results[file_name] = {
            "count":  count,
            "boxes":  boxes.tolist(),
            "scores": scores.tolist(),
        }

        if args.vis_output and count > 0:
            vis = draw_boxes_on_image(image, boxes)
            vis.save(os.path.join(args.output_folder, file_name))

    out_json = os.path.join(args.output_folder, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved results to %s", out_json)

    print("\n==== Summary ====")
    total = sum(r["count"] for r in results.values())
    print(f"Images processed : {len(images)}")
    print(f"Total detections : {total}")
    print(f"Results saved to : {out_json}")


if __name__ == "__main__":
    main()
