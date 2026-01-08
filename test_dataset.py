#!/usr/bin/env python3
import torch
import argparse, glob, json, logging, os
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont  # pip install pillow
import numpy as np
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms_app as T
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# https://github.com/PhyscalX/gradio-image-prompter/tree/main/backend/gradio_image_prompter/templates/component
import io
import matplotlib.patches as patches
import random
CONF_THRESH = 0.23

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Script to test different datasets")
    p.add_argument("--dataset_folder", type=str, help="includes the COCO annotation file and images for the dataset")
    p.add_argument("--pretrain_model_path", type=str, required=True)
    p.add_argument("--pos_text", action="store_true")
    p.add_argument("--num_pos_exemp", type=int, default=0, help="max number of positive exemplars to use per image, if fewer boxes available than --num_pos_exemp, just uses all the available boxes")
    p.add_argument("--use_ext_pos_exemp", action="store_true", help="apply the positive exemplar for an object in an image as an 'external exemplar' to the entire dataset (defaults to using each image's own 'internal' exemplars)")
    p.add_argument("--neg_text", action="store_true")
    p.add_argument("--num_neg_exemp", type=int, default=0, help="max number of negative exemplars to use per category in the image, if fewer boxes available than --num_neg_exemp, just uses all the available boxes")
    p.add_argument("--use_ext_neg_exemp", action="store_true", help="apply the negative exemplar for an object in an image as an 'external exemplar' to the entire dataset (defaults to using each image's own 'internal' exemplars)")
    p.add_argument("--out_dir", type=str, default="./model-output")
    p.add_argument("--vis_exemp", action="store_true", help="visualize the positive exemplars, note that these can cover the predicted labels")

    # Model parameters
    p.add_argument(
        "--device", default="cuda", help="device to use for inference"
    )
    p.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    p.add_argument("--remove_difficult", action="store_true")
    p.add_argument("--fix_size", action="store_true")

    # training parameters
    p.add_argument("--note", default="", help="add some notes to the experiment")
    p.add_argument("--resume", default="", help="resume from checkpoint")
    p.add_argument("--finetune_ignore", type=str, nargs="+")
    p.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    p.add_argument("--eval", action="store_false")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--test", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--find_unused_params", action="store_true")
    p.add_argument("--save_results", action="store_true")
    p.add_argument("--save_log", action="store_true")

    # distributed training parameters
    p.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    p.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    p.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    p.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    p.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    p.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return p

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])

    return box_inputs
    
def preprocess(transform, image, input_prompts = None):
    if input_prompts == None:
        prompts = { "image": image, "points": []}
    else:
        prompts = input_prompts

    input_image, _ = transform(image, None)
    exemplar = get_box_inputs(prompts["points"])
    # Wrapping exemplar in a dictionary to apply only relevant transforms
    input_image_exemplar, exemplar = transform(prompts['image'], {"exemplars": torch.tensor(exemplar, dtype=torch.float)})
    exemplar = exemplar["exemplars"]

    return input_image, input_image_exemplar, exemplar
    
# Get counting model.
def build_model_and_transforms(args):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile("cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from models.GroundingDINO import groundingdino_app

    build_func = groundingdino_app.build_groundingdino
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform

def get_box_coords_from_boxes(image, boxes):
    """
    Get box coordinates in the format (x, y, box_w, box_h) such that (x, y) is the top left of the box with width [box_w] and height [box_h] with all coordinates in the image coordinate system
    """
    (w, h) = image.size
    center_x = w * boxes[:, 0]
    center_y = h * boxes[:, 1]
    box_w = w * boxes[:, 2]
    box_h = h * boxes[:, 3]
    (x, y) = center_x - box_w/2, center_y - box_h/2
    return (x, y, box_w, box_h)

def get_xy_from_boxes(image, boxes):
    """
    Get box centers and return in image coordinates
    """
    (w, h) = image.size
    x = w * boxes[:, 0]
    y = h * boxes[:, 1]

    return x, y

def generate_heatmap(image, boxes):
    # Plot results.
    (w, h) = image.size
    det_map = np.zeros((h, w))
    x, y = get_xy_from_boxes(image, boxes)

    # Box centers are floating point, convert to int and clip them at edge of box
    x = np.clip(np.around(x).astype(int), 0, w - 1)
    y = np.clip(np.around(y).astype(int), 0, h - 1)

    det_map[y, x] = 1
    det_map = ndimage.gaussian_filter(
        det_map, sigma=(w // 200, w // 200), order=0
    )
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)
    ax.imshow(
        det_map, cmap="jet", alpha=0.2,
        extent=[0, w, h, 0],  # x0,x1,y0,y1
        origin="upper", interpolation="none"
    )
    plt.axis('off')

    # Plot bounding boxes
    (x0, y0, box_w, box_h) = get_box_coords_from_boxes(image, boxes)

    for box_ind in range(boxes.shape[0]):
        (x_i, y_i, box_w_i, box_h_i) = (x0[box_ind], y0[box_ind], box_w[box_ind], box_h[box_ind])
        # Create a Rectangle patch
        rect = patches.Rectangle((x_i, y_i), box_w_i, box_h_i, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()

    output_img = Image.open(img_buf)
    return output_img

def get_boxes_from_prediction(model_output, text, keywords = ""):
    input_ids = model_output["input_ids"][0]
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    boxes = model_output["pred_boxes"][0]

    # [pos_neg_split_idx] is the index of the first occurence of the "." separating token.
    for idx in range(len(input_ids)):
        token = input_ids[idx]
        if token == 1012:
            pos_neg_split_idx = idx
            break
    
    pos_logits = logits[:, :(pos_neg_split_idx + 1)]
    neg_logits = logits[:, (pos_neg_split_idx + 1):]

    # Stage 1 filtering:
    box_mask = pos_logits.max(dim=-1).values > CONF_THRESH
    boxes = boxes[box_mask, :]
    logits = logits[box_mask, :]

    # Stage 2 filtering:
    pos_logits = pos_logits[box_mask, :]
    neg_logits = neg_logits[box_mask, :]
    box_mask = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
    boxes = boxes[box_mask, :].cpu().numpy()
    logits = logits[box_mask, :].cpu().numpy().max(axis=-1)

    return boxes, logits

def predict(model, transform, image, positive_text, positive_prompts, negative_texts, negative_prompts, device):
    keywords = "" # do not handle this for now
    input_image, input_image_pos_exemplar, pos_exemplar = preprocess(transform, image, positive_prompts)
    negative_images = []
    neg_exemplars = []
    for ind in range(len(negative_prompts)):
        _, input_image_neg_exemplar, neg_exemplar = preprocess(transform, image, negative_prompts[ind])
        negative_images.append(nested_tensor_from_tensor_list(input_image_neg_exemplar.unsqueeze(0).to(device)))
        neg_exemplars.append([neg_exemplar.to(device)])

    input_images = input_image.unsqueeze(0).to(device)
    input_image_pos_exemplars = input_image_pos_exemplar.unsqueeze(0).to(device)
    pos_exemplars = [pos_exemplar.to(device)]

    caption = positive_text + " . "
    for negative_text in negative_texts:
        caption = caption + negative_text + " . "
    with torch.no_grad():
        model_output = model(
                nested_tensor_from_tensor_list(input_images),
                nested_tensor_from_tensor_list(input_image_pos_exemplars),
                pos_exemplars,
                negative_images,
                neg_exemplars,
                captions=[caption],
            )
        
    return get_boxes_from_prediction(model_output, positive_text, keywords)

def _predict(image, positive_text, positive_prompts, negative_text, negative_prompts):
        return predict(model, transform, image, positive_text, positive_prompts, negative_text, negative_prompts, device)

def count(image, positive_text, positive_prompts, negative_texts, negative_prompts):
        if positive_prompts is None:
            positive_prompts = {"image": image, "points": []}
        boxes, scores = _predict(image, positive_text, positive_prompts, negative_texts, negative_prompts)
        predicted_count = len(boxes)
        output_img = generate_heatmap(image, boxes)
        (x0, y0, box_w, box_h) = get_box_coords_from_boxes(image, boxes)
        boxes = np.stack((x0, y0, box_w, box_h), axis=-1)

        return boxes, scores
    
def get_category_folders(root: str, args) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p) and os.path.basename(p) == args.category])

def list_images_in_split(split_dir: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    imgs: List[str] = []
    for e in exts: imgs.extend(glob.glob(os.path.join(split_dir, e)))
    return sorted(imgs)

def load_coco_annotations(split_dir: str) -> Dict[str, Any]:
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        logging.warning("No COCO annotations found at %s", ann_path)
        return {}
    with open(ann_path, "r", encoding="utf-8") as f: return json.load(f)

def _category_palette(categories: List[str]) -> Dict[str, Tuple[int,int,int]]:
    pal = {}
    for name in categories:
        h = abs(hash(name))
        pal[name] = (64 + (h & 0x7F), 64 + ((h>>7)&0x7F), 64 + ((h>>14)&0x7F))
    return pal

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int,int]:
    """Robust text measurement across Pillow versions."""
    try:
        # Preferred in newer Pillow
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        try:
            # Fallback: font-based sizing
            return font.getsize(text)  # or font.getbbox(text)
        except Exception:
            # Last resort: rough guess
            return (max(6 * len(text), 1), 12)

def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], text: str,
                color: Tuple[int,int,int], font) -> None:
    x, y = xy
    tw, th = _measure_text(draw, text, font)
    pad = 2
    rect = [x, max(0, y - th - pad*2), x + tw + pad*2, y]
    try:
        draw.rectangle(rect, fill=color)
    except Exception as e:
        print("not drawing label for " + text)
    draw.text((x + pad, rect[1] + pad), text, fill=(255,255,255), font=font)

def visualize_from_count_anno(img_path: str,
                              per_cat_boxes: Dict[str, List[List[float]]],
                              palette: Dict[str, Tuple[int,int,int]],
                              save_path: str) -> None:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logging.warning("Failed to open %s: %s", img_path, e); return
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for cat_name, boxes in per_cat_boxes.items():
        color = palette.get(cat_name, (255,255,255))
        for x, y, w, h in boxes:
            x2, y2 = x + w, y + h
            draw.rectangle([x, y, x2, y2], outline=color, width=3)
            _draw_label(draw, (int(x), int(y)), cat_name, color, font)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        img.save(save_path); logging.info("Saved visualization: %s", save_path)
    except Exception as e:
        logging.error("Failed to save %s: %s", save_path, e)

def visualize_from_count_pred(img_path: str,
                              per_cat_boxes: Dict[str, List[List[float]]],
                              per_cat_exemplars: Dict[str, List[List[float]]],
                              palette: Dict[str, Tuple[int,int,int]],
                              save_path: str) -> None:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logging.warning("Failed to open %s: %s", img_path, e); return
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for cat_name, boxes in per_cat_boxes.items():
        color = palette.get(cat_name, (255,255,255))
        for x, y, w, h in boxes:
            x2, y2 = x + w, y + h
            draw.rectangle([x, y, x2, y2], outline=color, width=3)
            _draw_label(draw, (int(x), int(y)), cat_name, color, font)
    for cat_name, boxes in per_cat_exemplars.items():
        color = (0, 0, 0)
        for x, y, w, h in boxes:
            x2, y2 = x + w, y + h
            draw.rectangle([x, y, x2, y2], outline=color, width=3)
            _draw_label(draw, (int(x), int(y)), cat_name + " exemplar", color, font)
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        img.save(save_path); logging.info("Saved visualization: %s", save_path)
    except Exception as e:
        logging.error("Failed to save %s: %s", save_path, e)
                                  
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
args = build_arg_parser().parse_args()
device = get_device()
model, transform = build_model_and_transforms(args)
model = model.to(device)

logging.info("Dataset folder: %s", args.dataset_folder)
logging.info("Output dir: %s", args.out_dir)

results: Dict[str, Dict[str, Any]] = {}
images = list_images_in_split(args.dataset_folder)
annotations = load_coco_annotations(args.dataset_folder)

# Build count_anno  {image_name: {category_name: [bbox,...]}}
cat_id_to_text = {c["id"]: c["name"] for c in annotations.get("categories", [])}
cat_text_to_id = {cat_id_to_text[cat_id]: cat_id for cat_id in cat_id_to_text}
count_anno: Dict[str, Dict[str, List[List[float]]]] = {}
image_name_to_id = {}
for image in annotations.get("images", []):
    img_name, img_id = image["file_name"], image["id"]
    image_name_to_id[img_name] = img_id
    per_cat: Dict[str, List[List[float]]] = {}
    for ann in annotations.get("annotations", []):
        if ann["image_id"] != img_id: continue
        cat_name = cat_id_to_text.get(ann["category_id"], "unknown")
        per_cat.setdefault(cat_name, []).append(ann["bbox"])
    count_anno[img_name] = per_cat

palette = _category_palette(sorted(set(cat_id_to_text.values())))
coco_predictions = []
img_counter = 0
num_images = len(count_anno)
for img_file, per_cat_boxes_gt in count_anno.items():
    img_path = os.path.join(args.dataset_folder, img_file)
    if not os.path.isfile(img_path):
        logging.warning("Missing image referenced in JSON: %s", img_path); continue
    
    save_path = os.path.join(args.out_dir, img_file)
    # Run inference.
    per_cat_boxes_pred = {}
    per_cat_exemplars_pred = {}
    if args.use_ext_pos_exemp or args.use_ext_neg_exemp:
        with open(os.path.join(args.dataset_folder, "external_exemplars.json")) as f:
            external_exemplars = json.load(f)
    for cat in per_cat_boxes_gt:
        image = Image.open(img_path).convert("RGB")
        if args.use_ext_pos_exemp:
            ext_img_file = external_exemplars[cat]["image"]
            external_image = Image.open(os.path.join(args.dataset_folder, ext_img_file)).convert("RGB")
            pos_exemps = external_exemplars[cat]["boxes"][:args.num_pos_exemp]
            pos_prompts = {"image": external_image, "points":[]}
            for x1, y1, w, h in pos_exemps:
                x2, y2 = x1 + w, y1 + h
                pos_prompts["points"].append([x1, y1, 2.0, x2, y2, 3.0]) # gradio image prompter format for box
        else:
            boxes_gt = per_cat_boxes_gt[cat]
            pos_exemps = boxes_gt[:args.num_pos_exemp]
            pos_prompts = {"image": image, "points": []}
            for x1, y1, w, h in pos_exemps:
                x2, y2 = x1 + w, y1 + h
                pos_prompts["points"].append([x1, y1, 2.0, x2, y2, 3.0]) # gradio image prompter format for box
        if args.pos_text:
            text = cat
        else:
            text = ""
        # Gather negative prompts.
        neg_texts = [pot_neg_cat for pot_neg_cat in per_cat_boxes_gt if pot_neg_cat != cat]
        neg_prompts = []
        for neg_text in neg_texts:
            if args.use_ext_neg_exemp:
                ext_img_file = external_exemplars[neg_text]["image"]
                external_image = Image.open(os.path.join(args.dataset_folder, ext_img_file)).convert("RGB")
                neg_exemps = external_exemplars[neg_text]["boxes"][:args.num_neg_exemp]
                neg_prompt = {"image": external_image, "points":[]}
                for x1, y1, w, h in neg_exemps:
                    x2, y2 = x1 + w, y1 + h
                    neg_prompt["points"].append([x1, y1, 2.0, x2, y2, 3.0]) # gradio image prompter format for box
            else:
                boxes_gt = per_cat_boxes_gt[neg_text]
                neg_exemps = boxes_gt[:args.num_neg_exemp]
                neg_prompt = {"image": image, "points":[]}
                for x1, y1, w, h in neg_exemps:
                    x2, y2 = x1 + w, y1 + h
                    neg_prompt["points"].append([x1, y1, 2.0, x2, y2, 3.0]) # gradio image prompter format for box
            neg_prompts.append(neg_prompt)
                
        if not args.neg_text: # If there are no negative texts, then no negative prompts for now
            neg_texts = []
            neg_prompts = []
        boxes_pred, scores_pred = count(image, text, pos_prompts, neg_texts, neg_prompts)
        per_cat_boxes_pred[cat] = boxes_pred
        per_cat_exemplars_pred[cat] = pos_exemps
        # Add results to COCO predictions json.
        for ind in range(boxes_pred.shape[0]):
            bbox = boxes_pred[ind,:].tolist() 
            score = float(scores_pred[ind])
            image_id = int(image_name_to_id[img_file])
            category_id = int(cat_text_to_id[cat])
            coco_predictions.append({"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score})

    img_counter +=1
    print("Processed Images: " + str(img_counter) + "/" + str(num_images))
            

# Write COCO predictions to JSON.
os.makedirs(args.out_dir, exist_ok=True)
out_json = os.path.join(args.out_dir, "coco_predictions.json")
with open(out_json, "w") as f:
    json.dump(coco_predictions, f)
print(f"Saved {len(coco_predictions)} detections to {out_json}")

print("\n==== Summary ====")
print(f"{args.dataset_folder:30s} | images: {num_images:5d}")
