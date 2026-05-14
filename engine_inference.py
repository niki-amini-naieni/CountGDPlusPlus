# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import json
import numpy as np
import glob
import math
import os
import sys
import io
import contextlib
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from util.utils import to_device
from util.visualizer import renorm
import torch
import torchvision.transforms.functional as F

import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator
from datasets.transforms import RandomResize
from scipy.stats import bernoulli
import scipy.ndimage as ndimage

from PIL import Image
import datasets.transforms as T


from segment_anything import sam_model_registry, SamPredictor

def get_pseudo_exemplars(outputs, image_sizes, labels_uncropped, cat_list, captions, box_threshold, num_exemplars=3):
    logits = outputs['pred_logits'].sigmoid()
    boxes = outputs['pred_boxes']
    input_ids = outputs['token']['input_ids']
    tokenized = outputs['token']

    bs = len(captions)
    pseudo_exemplars = []

    for batch_ind in range(bs):
        # Get the scores for the boxes corresponding to the specified objects.
        logits_sample = logits[batch_ind, :, :]
        scores_sample = logits_sample.max(dim=-1).values
        boxes_sample = boxes[batch_ind, :, :]
        tokenized_sample = tokenized[batch_ind]
        caption_sample = captions[batch_ind]
        cat_list_sample = cat_list[batch_ind]
        label_sample = labels_uncropped[batch_ind][0] # This works because all boxes in an image correspond to the same label in FSC-147.

        # Only use boxes above the box threshold.
        box_mask = scores_sample > box_threshold
        if torch.sum(box_mask).item() > 0:
            boxes_sample = boxes_sample[box_mask, :]
            scores_sample = scores_sample[box_mask]

            # Out of all the boxes, select at most [num_exemplars] of the highest scoring boxes.
            scores_sample, indices = torch.sort(scores_sample, dim=0, descending=True)
            boxes_sample = boxes_sample[indices, :]
            num_exemplars_sample = min(num_exemplars, boxes_sample.shape[0])
            pseudo_exemplars_sample = boxes_sample[:num_exemplars_sample, :]

            # Convert the normalized boxes to the exemplars format.
            image_size_sample = image_sizes[batch_ind]
            (img_h, img_w) = (image_size_sample[0], image_size_sample[1])
            cx = img_w * pseudo_exemplars_sample[:, 0]
            cy = img_h * pseudo_exemplars_sample[:, 1]
            w = img_w * pseudo_exemplars_sample[:, 2]
            h = img_h * pseudo_exemplars_sample[:, 3]
            x0 = torch.clamp(cx - w/2, min=0, max=img_w)
            x1 = torch.clamp(cx + w/2, min=0, max=img_w)
            y0 = torch.clamp(cy - h/2, min=0, max=img_h)
            y1 = torch.clamp(cy + h/2, min=0, max=img_h)
            pseudo_exemplars.append(torch.stack([x0, y0, x1, y1], dim=-1))
        else:
            pseudo_exemplars.append(torch.empty((0,4)).cuda())

    # Make sure all samples in a batch have the same number of exemplars
    min_exemplars_in_batch = min([exemp.shape[0] for exemp in pseudo_exemplars])
    pseudo_exemplars = [exemp[:min_exemplars_in_batch] for exemp in pseudo_exemplars]
    # Return the pseudo exemplars.
    return pseudo_exemplars

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]
        labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        min_exemplars_in_batch = min([exemp.shape[0] for exemp in exemplars])
        shot_num = min(3, min_exemplars_in_batch)
        print("Shot Num: " + str(shot_num))
        if args.train_with_exemplar_only:
            if shot_num == 0:
                continue
            model.drop_text = True
            # Remove text from samples.
            new_captions = []
            for sample_ind in range(len(labels_uncropped)):
                caption = captions[sample_ind]
                label = labels_uncropped[sample_ind][0]
                sample_cap_list = cap_list[sample_ind]
                new_caption = caption.replace(sample_cap_list[label] + " ", "")
                new_captions.append(new_caption)
            captions = new_captions
        else:
            model.drop_text = False

        # Use modality dropout with 50% chance if have more than one exemplar.
        if args.modality_dropout and shot_num > 0:
            p = 0.5
            r = bernoulli.rvs(p, size=1)
            if r == 1:
                print("Applying Modality Dropout")
                # Apply modality dropout.
                r = bernoulli.rvs(p, size=1)
                if r == 0:
                    # Use text only.
                    shot_num = 0
                    print("Using text only due to modality dropout")
                else:
                    # Use exemplars only.
                    model.drop_text = True
                    print("Using exemplars only due to modality dropout")
                    # Remove text from samples.
                    new_captions = []
                    for sample_ind in range(len(labels_uncropped)):
                        caption = captions[sample_ind]
                        label = labels_uncropped[sample_ind][0]
                        sample_cap_list = cap_list[sample_ind]
                        new_caption = caption.replace(sample_cap_list[label] + " ", "")
                        new_captions.append(new_caption)
                    captions = new_captions

        # Adjust number of exemplars based on [shot_num].
        exemplars = [exemp[:shot_num] for exemp in exemplars]

        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, exemplars, labels_uncropped, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_sam(sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def tt_norm_sam(predictor, pred_cnt, image, exemplars, size, points):
    e_cnt = 0
    avg_cnt = 0
    (h, w) = (size[0], size[1])
    xv, yv = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    image_cv = numpy_2_cv2(image)
    predictor.set_image(image_cv)

    for exemp in exemplars:
        in_exemp = (points[:, 0] * w > exemp[0]) * (points[:, 0] * w < exemp[2])
        in_exemp = (
            (in_exemp) * (points[:, 1] * h > exemp[1]) * (points[:, 1] * h < exemp[3])
        )
        # There are at least 2 points inside the exemplar [exemp].
        if np.sum(in_exemp) >= 2:
            print("refining tt norm with SAM")
            sam_mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=exemp[None, :],
                multimask_output=False,
            )
            x_mask = xv[sam_mask.squeeze()]
            y_mask = yv[sam_mask.squeeze()]
            mask_points = np.stack((x_mask, y_mask), axis=1)
            num_points_in_mask = 0
            for point in points:
                in_mask = False
                # Check if [point] lies inside the mask.
                for mask_point in mask_points:
                    if mask_point[0] == round(w * point[0]) and mask_point[1] == round(
                        h * point[1]
                    ):
                        in_mask = True
                        break
                if in_mask:
                    num_points_in_mask += 1

            # There is an exemplar mask with more than one detected instance.
            if num_points_in_mask >= 2:
                e_cnt += 1
            # Add to the average exemplar count using the SAM mask instead of the exemplar.
            avg_cnt += num_points_in_mask
        else:
            # Add to the average exemplar count using the exemplar.
            avg_cnt += np.sum(in_exemp)

    # If there are at least 2 exemplar masks with more than one detected instance, apply the TT-Norm.
    if e_cnt >= 2:
        avg_cnt = avg_cnt / exemplars.shape[0]
        print("Using TT-Norm")
        print("orig count: " + str(pred_cnt))
        pred_cnt = pred_cnt / avg_cnt
        print("new count: " + str(pred_cnt))

    return pred_cnt

def numpy_2_cv2(np_img):
    if np.min(np_img) < 0:
        raise Exception("image min is less than 0. Img min: " + str(np.min(np_img)))
    if np.max(np_img) > 1:
        raise Exception("image max is greater than 1. Img max: " + str(np.max(np_img)))

    np_img = (np_img * 255).astype(np.uint8)
    cv2_image = np.asarray(np_img)
    return cv2_image


def next_divisor(x: float, y: float) -> float:
    """
    Returns x if y is divisible by x.
    Otherwise, returns the next smaller float (below x) that evenly divides y.
    """
    if y % x == 0:
        return x
    
    # Start slightly below x and move downward until finding a divisor
    step = 0.0001  # controls precision of the search
    candidate = x - step

    while candidate > 0:
        if y % candidate <= 0.01:
            return round(candidate, 6)
        candidate -= step

    return None  # no valid divisor found

def crop_divisible(sample, crop_width, crop_height, overlap_width, overlap_height):
    (h, w) = sample.shape[1], sample.shape[2]

    crop_height = int(next_divisor(crop_height, h))
    crop_width = int(next_divisor(crop_width, w))

    samples_cropped = []
    start_y = 0
    end_y = crop_height
    start_x = 0
    end_x = crop_width
    boundaries_x = []
    boundaries_y = []

    while end_y < h:
        end_y = start_y + crop_height
        if end_y > h:
            break
        boundaries_row_x = []
        boundaries_row_y = []
        while end_x < w:
            end_x = start_x + crop_width
            if end_x > w:
                start_x = 0
                end_x = crop_width
                start_y = start_y + crop_height - overlap_height
                break
            samples_cropped.append(
                RandomResize([800], max_size=1333)(
                    sample[:, start_y:end_y, start_x:end_x].unsqueeze(0)
                )[0].squeeze()
            )
            boundaries_row_x.append((start_x, end_x))
            boundaries_row_y.append((start_y, end_y))
            start_x = start_x + crop_width - overlap_width
        boundaries_x.append(boundaries_row_x)
        boundaries_y.append(boundaries_row_y)

    return samples_cropped, boundaries_x, boundaries_y, crop_width, crop_height

def crop(sample, crop_width, crop_height, overlap_width, overlap_height):
    (h, w) = sample.shape[1], sample.shape[2]

    samples_cropped = []
    start_y = 0
    end_y = crop_height
    start_x = 0
    end_x = crop_width
    boundaries_x = []
    boundaries_y = []

    while end_y < h:
        end_y = start_y + crop_height
        if end_y > h:
            # Shift up to increase overlap when hit bottom end of image.
            shift_up = end_y - h
            start_y = start_y - shift_up
            end_y = h
        boundaries_row_x = []
        boundaries_row_y = []
        while end_x < w:
            end_x = start_x + crop_width
            if end_x > w:
                # Shift left to increase overlap when hit right end of image.
                shift_left = end_x - w
                start_x = start_x - shift_left
                end_x = w
            samples_cropped.append(
                RandomResize([800], max_size=1333)(
                    sample[:, start_y:end_y, start_x:end_x].unsqueeze(0)
                )[0].squeeze()
            )
            boundaries_row_x.append((start_x, end_x))
            boundaries_row_y.append((start_y, end_y))
            start_x = start_x + crop_width - overlap_width
        boundaries_x.append(boundaries_row_x)
        boundaries_y.append(boundaries_row_y)

        start_x = 0
        end_x = crop_width
        start_y = start_y + crop_height - overlap_height

    return samples_cropped, boundaries_x, boundaries_y

# Save CountGD-Box detections in COCO format.
countgd_detections = []
anno_id = 1
num_cropped = 0
def get_count_errs(
    model,
    args,
    samples,
    exemplars,
    outputs,
    box_threshold,
    text_threshold,
    targets,
    tokenized_captions,
    input_captions,
    cap_list,
    predictor=None,
):
    global countgd_detections
    global anno_id
    global num_cropped

    with open(args.datasets, 'r') as f:
        datasets_file = json.load(f)
        datasets_path = datasets_file["val"][0]["anno"]
        images_path = datasets_file["val"][0]["root"]
    with open(datasets_path, 'r') as f:
        datasets_file = json.load(f)
    
    
    img_id_to_name = {image["id"]: image["file_name"] for image in datasets_file["images"]}
    
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    samples = samples.to_img_list()
    sizes = [target["size"] for target in targets]
    orig_sizes = [target['orig_size'] for target in targets]
    img_ids = [target['image_id'].item() for target in targets]

    abs_errs = []
    for sample_ind in range(len(targets)):
        sample_logits = logits[sample_ind]
        sample_boxes = boxes[sample_ind]
        sample = samples[sample_ind]
        size = sizes[sample_ind]
        sample_exemplars = exemplars[sample_ind]
        img_name = img_id_to_name[img_ids[sample_ind]]
        img_path = os.path.join(images_path, img_name)

        sample_caption = input_captions[sample_ind]

        # Get the index ([end_idx]) where the main tokens end (the '.' separator indicates this for GroundingDINO).
        for token_ind in range(len(tokenized_captions["input_ids"][sample_ind])):
            idx = tokenized_captions["input_ids"][sample_ind][token_ind]
            if idx == 1012:
                end_idx = token_ind
                break

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        sample_logits = sample_logits[box_mask, :]
        sample_boxes = sample_boxes[box_mask, :]
        orig_size = orig_sizes[sample_ind]
        coco_boxes = []
        # Get median object widths and heights to avoid outliers
        obj_widths = []
        obj_heights = []
        coco_dets = []
        for coco_idx in range(sample_logits.shape[0]):
            score = float(sample_logits[coco_idx].max(dim=-1).values)
            coco_det = {"image_id": targets[sample_ind]['image_id'].item(), "category_id": 1, "score": score}
            bbox = sample_boxes[coco_idx, :]
            obj_widths.append((bbox[2] * size[1]).item())
            obj_heights.append((bbox[3] * size[0]).item())
            (h, w) = orig_size[0], orig_size[1]
            box_w = w * bbox[2]
            box_h = h * bbox[3]
            x0 = w * bbox[0] - box_w / 2
            y0 = h * bbox[1] - box_h / 2
            x1 = x0 + box_w
            y1 = y0 + box_h
            coco_det["bbox"] = [int(x0), int(y0), int(box_w), int(box_h)]
            coco_boxes.append(coco_det["bbox"])
            coco_det["point"] = [float(w*bbox[0]), float(h*bbox[1])]
            coco_det['area'] = float((box_w * box_h).item())
            coco_dets.append(coco_det)

        text_mask = (sample_logits[:, 1:end_idx] > text_threshold).sum(dim=-1) == (
            end_idx - 1
        )
        sample_logits = sample_logits[text_mask, :]
        sample_boxes = sample_boxes[text_mask, :]

        gt_count = targets[sample_ind]['labels'].shape[0]
        pred_cnt = sample_logits.shape[0]

        # Predicted max # of objects with some slack (see https://github.com/niki-amini-naieni/CountGD/issues/60#issuecomment-3717822451). Sometimes need to apply some slack/provide a 'soft' threshold for the adaptive cropping to be activated when the model nears the 900-object limit.
        slack = 100
        if args.crop and pred_cnt >= args.num_select - slack:
            coco_dets = []
            print("Detected high number of objects, cropping with super resolution")
            normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            crop_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    normalize,
                ]
            )
            # Get high-res crops
            if args.num_exemplars <= 0:
                crop_dir = os.path.join(args.superres_folder, "pseudo-crops")
            elif args.num_exemplars == 3 and args.no_text:
                crop_dir = os.path.join(args.superres_folder, "exemplars-only-crops")
            elif args.num_exemplars == 3:
                crop_dir = os.path.join(args.superres_folder, "exemplars-and-text-crops")
            else:
                raise Exception("High resolution crops only obtained for the cases of synthetic and pseudo-exemplars or 3 real exemplars")
            crop_paths = glob.glob(os.path.join(crop_dir, "crop-*" + img_name[:-4] + "*.png"))

            # Get the count for each crop and sum the counts
            pred_cnt = 0
            transformed_boxes = []
            crop_names = [os.path.basename(crop_path) for crop_path in crop_paths]
            crop_rows = [int(crop_name[5]) for crop_name in crop_names]
            crop_cols = [int(crop_name[7]) for crop_name in crop_names]
            num_rows = max(crop_rows) + 1
            num_cols = max(crop_cols) + 1

            for crop_path in crop_paths:
                crop_name = os.path.basename(crop_path)
                crop_row = int(crop_name[5])
                crop_col = int(crop_name[7])
                transformed_crop, crop_exemplars = crop_transform(Image.open(crop_path).convert('RGB'), {"exemplars": torch.tensor([])})
                transformed_crop = transformed_crop.cuda().unsqueeze(0)
                crop_exemplars = crop_exemplars["exemplars"].cuda()
                (super_h, super_w) = transformed_crop.shape[-2], transformed_crop.shape[-1]
                (total_super_h, total_super_w) = super_h * num_rows, super_w * num_cols
                print(transformed_crop.shape)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # Use 'label' of 0 at inference since only input a single text prompt instead of all COCO classes.
                    if args.no_text:
                        outputs = model(
                            transformed_crop,
                            nested_tensor_from_tensor_list([sample]),
                            [sample_exemplars],# Add batch dimension 
                            [],
                            [],
                            captions=input_captions,
                        )
                    else:
                        outputs = model(
                            transformed_crop,
                            nested_tensor_from_tensor_list([transformed_crop[0]]),
                            [crop_exemplars],# Add batch dimension 
                            [],
                            [],
                            captions=input_captions,
                        )

                crop_logits = outputs["pred_logits"].sigmoid()[0]
                crop_boxes = outputs["pred_boxes"][0]
                crop_box_mask = crop_logits.max(dim=-1).values > box_threshold
                crop_logits = crop_logits[crop_box_mask, :]
                crop_boxes = crop_boxes[crop_box_mask, :]
                pred_cnt += crop_boxes.shape[0]
                for coco_idx in range(crop_logits.shape[0]):
                    score = float(crop_logits[coco_idx].max(dim=-1).values)
                    coco_det = {"image_id": targets[sample_ind]['image_id'].item(), "category_id": 1, "score": score}
                    box = crop_boxes[coco_idx, :]
                    cx, cy, box_w, box_h = box[0]*super_w, box[1]*super_h, box[2]*super_w, box[3]*super_h
                    x0, y0 = cx - box_w/2, cy - box_h/2
                    offset_x = crop_col * super_w
                    offset_y = crop_row * super_h
                    x0 = x0 + offset_x
                    y0 = y0 + offset_y
                    # Transform the boxes back to the original image space.
                    (orig_h, orig_w) = orig_sizes[0][0], orig_sizes[0][1]
                    x0, y0, box_w, box_h = x0 * (orig_w/total_super_w), y0 * (orig_h/total_super_h), box_w * (orig_w/total_super_w), box_h * (orig_h/total_super_h)
                    transformed_boxes.append([x0.item(), y0.item(), box_w.item(), box_h.item()])
                    coco_det["bbox"] = [int(x0), int(y0), int(box_w), int(box_h)]
                    coco_boxes.append(coco_det["bbox"])
                    coco_det["point"] = [float(w*bbox[0]), float(h*bbox[1])]
                    coco_det['area'] = float((box_w * box_h).item())
                    coco_dets.append(coco_det)

            # Plot the bounding boxes onto the original image, location preserved by superresolution method https://www.bubbi.app/editing/upscaler

            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            plt_image = Image.open(img_path)
            ax.imshow(plt_image)
            
            # Plot bounding boxes
            for box_ind in range(len(transformed_boxes)):
                (x_i, y_i, box_w_i, box_h_i) = (transformed_boxes[box_ind][0], transformed_boxes[box_ind][1], transformed_boxes[box_ind][2], transformed_boxes[box_ind][3])
                # Create a Rectangle patch
                rect = patches.Rectangle((x_i, y_i), box_w_i, box_h_i, edgecolor='cyan', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

            #ax.set_title("Pred: " + str(pred_cnt) + ", GT: " + str(gt_count))
            #plt.savefig("debug-" + img_name, format='png', bbox_inches='tight')
            #plt.close()

        elif args.crop and pred_cnt >= args.num_select - slack and False:
            """
            The below code block shows how the original crops were obtained before applying the super-resolution
            """
            num_cropped+=1
            # If crop image do not apply TT-Norm as double counting may occur on crop boundaries for a few (but not all) samples. In other words, cropping may (incorrectly) cause higher counts around boundaries, which may lead to false detection of self-similarity if TT-Norm is applied. Solution for now is to just disable TT-Norm when apply cropping.
            print("Detected high number of objects, cropping...")

            # Crop image.
            (h, w) = size[0], size[1]
            (orig_h, orig_w) = orig_size[0], orig_size[1]

            # Get crop size.
            obj_width = np.min(obj_widths)
            obj_height = np.min(obj_heights)

            # Limit crop size to include approximately 100 objects assuming 16 << args.num_select.
            crop_width = 25 * obj_width
            crop_height = 25 * obj_height

            # Get overlap size. Ensures each object instance is fully seen by at least one crop window.
            overlap_width = 0
            overlap_height = 0

            print("crop_width: " + str(crop_width))
            print("crop_height: " + str(crop_height))

            samples_cropped, boundaries_x, boundaries_y, crop_width, crop_height = crop_divisible(
                sample, crop_width, crop_height, overlap_width, overlap_height
            )

            num_batches = int(np.ceil(len(samples_cropped) / 10))
            print("num_batches: " + str(num_batches))
            logits_cropped = []
            boxes_cropped = []

            pred_cnt = 0
            transformed_boxes = []
            for batch_ind in range(num_batches):
                print(batch_ind)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # Use 'label' of 0 at inference since only input a single text prompt instead of all COCO classes (as do during training).
                    sample_subset = samples_cropped[
                        batch_ind * 10 : min((batch_ind + 1) * 10, len(samples_cropped))
                    ]

                    outputs_high_objects = model(
                        nested_tensor_from_tensor_list(sample_subset),
                        nested_tensor_from_tensor_list([input_image_exemplars] * len(sample_subset)),
                        [synthetic_exemplars] * len(sample_subset),
                        [torch.tensor([0]).cuda() for _ in range(len(sample_subset))],
                        captions=[sample_caption] * len(sample_subset)
                    )
                logits_cropped.append(outputs_high_objects["pred_logits"].sigmoid())
                boxes_cropped.append(outputs_high_objects["pred_boxes"])

            logits_cropped = torch.cat(logits_cropped)
            boxes_cropped = torch.cat(boxes_cropped)

            for row_ind in range(len(boundaries_x)):
                for col_ind in range(len(boundaries_x[0])):
                    crop_ind = row_ind * len(boundaries_x[0]) + col_ind
                    sample_logits_cropped = logits_cropped[crop_ind]
                    sample_boxes_cropped = boxes_cropped[crop_ind]
                    start_x, end_x = (
                        boundaries_x[row_ind][col_ind][0],
                        boundaries_x[row_ind][col_ind][1],
                    )
                    start_y, end_y = (
                        boundaries_y[row_ind][col_ind][0],
                        boundaries_y[row_ind][col_ind][1],
                    )

                    # Get the index ([end_idx]) where the special tokens end (the '.' separator indicates this for GroundingDINO).
                    for token_ind in range(
                        len(tokenized_captions["input_ids"][sample_ind])
                    ):
                        idx = tokenized_captions["input_ids"][sample_ind][token_ind]
                        if idx == 1012:
                            end_idx = token_ind
                            break

                    box_mask = sample_logits_cropped.max(dim=-1).values > box_threshold
                    sample_logits_cropped = sample_logits_cropped[box_mask, :]
                    sample_boxes_cropped = sample_boxes_cropped[box_mask, :]

                    text_mask = (
                        sample_logits_cropped[:, 1:end_idx] > text_threshold
                    ).sum(dim=-1) == (end_idx - 1)
                    sample_logits_cropped = sample_logits_cropped[text_mask, :]
                    sample_boxes_cropped = sample_boxes_cropped[text_mask, :]

                    # Transform boxes back to the original image space.
                    for box in sample_boxes_cropped:
                        cx, cy, box_w, box_h = box[0]*crop_width + start_x, box[1]*crop_height + start_y, box[2]*crop_width, box[3]*crop_height
                        scale_w = orig_w/w
                        scale_h = orig_h/h
                        cx, cy, box_w, box_h = cx*scale_w, cy*scale_h, box_w*scale_w, box_h*scale_h
                        x0, y0 = cx - box_w/2, cy - box_h/2
                        transformed_boxes.append([x0.item(), y0.item(), box_w.item(), box_h.item()])

                    pred_crop_cnt = sample_boxes_cropped.shape[0]

                    pred_cnt += pred_crop_cnt

                        # Plot result

            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            plt_image = Image.open(img_path)
            ax.imshow(plt_image)
            
            # Plot bounding boxes
            for box_ind in range(len(transformed_boxes)):
                (x_i, y_i, box_w_i, box_h_i) = (transformed_boxes[box_ind][0], transformed_boxes[box_ind][1], transformed_boxes[box_ind][2], transformed_boxes[box_ind][3])
                # Create a Rectangle patch
                rect = patches.Rectangle((x_i, y_i), box_w_i, box_h_i, edgecolor='cyan', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
            # Plot pseudo-exemplars
            for box_ind in range(len(exemplars[0])):
                (x_i, y_i, box_w_i, box_h_i) = (exemplars[0][box_ind][0], exemplars[0][box_ind][1], exemplars[0][box_ind][2], exemplars[0][box_ind][3])
                # Create a Rectangle patch
                rect = patches.Rectangle((int(x_i), int(y_i)), int(box_w_i), int(box_h_i), edgecolor='red', linewidth=2, facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
            # Save the crops (this shows how the original crops were obtained)
            for row_ind in range(len(boundaries_x)):
                for col_ind in range(len(boundaries_x[0])):
                    scale_w = (orig_w/w).item()
                    scale_h = (orig_h/h).item()
                    start_x = boundaries_x[row_ind][col_ind][0] * scale_w
                    start_y = boundaries_y[row_ind][col_ind][0] * scale_h
                    crop = Image.open(img_path).crop((start_x, start_y, start_x + crop_width*scale_w, start_y + crop_height*scale_h))
                    crop.save("crop-" + str(row_ind) + "-" + str(col_ind) + "-" + img_name)

            ax.set_title("Pred: " + str(pred_cnt) + ", GT: " + str(gt_count))
            plt.savefig("debug-" + img_name, format='png', bbox_inches='tight')
            plt.close()

        elif args.sam_tt_norm and args.num_exemplars > 0:
            pred_cnt = tt_norm_sam(
                predictor,
                pred_cnt,
                renorm(sample.cpu()).permute(1, 2, 0).numpy(),
                sample_exemplars.cpu().numpy(),
                size.cpu().numpy(),
                sample_boxes[:, :2].cpu().numpy(),
            )

        print("Pred Count: " + str(pred_cnt) + ", GT Count: " + str(gt_count))

        abs_errs.append(np.abs(gt_count - pred_cnt))
        
        # add coco dets
        for coco_det in coco_dets:
            coco_det["id"] = anno_id
            anno_id +=1
        countgd_detections += coco_dets

    return abs_errs


@torch.no_grad()
def evaluate(
    model,
    model_without_ddp,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):
    model.eval()
    criterion.eval()

    if args.sam_tt_norm:
        predictor = get_sam(sam_checkpoint=args.sam_model_path)
    else:
        predictor = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO

        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item["name"] for item in category_dict]
    else:
        cat_list = args.val_label_list
    caption = " . ".join(cat_list) + " ."
    print("Input text prompt:", caption)

    # Get the transform for the synthetic exemplar images.
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    synthetic_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    with open(args.datasets, 'r') as f:
        datasets_path = json.load(f)["val"][0]["anno"]
    with open(datasets_path, 'r') as f:
        datasets_file = json.load(f)
    
    
    img_id_to_name = {image["id"]: image["file_name"] for image in datasets_file["images"]}

    if args.use_synth_exemplars:
        synthetic_exemplar_dir = args.synth_exemplar_folder
        with open(os.path.join(synthetic_exemplar_dir, "generated_exemplars.json"), 'r') as f:
            exemplars_file = json.load(f)

    abs_errs = []
    for samples, targets in metric_logger.log_every(
        data_loader, 10, header, logger=logger
    ):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        if not args.remove_bad_exemplar:
            exemplars = [t["exemplars"][: args.num_exemplars].to(device) for t in targets]
        else:
            exemplars = []
            for t in targets:
                if t['image_id'] != 152:
                    exemplars.append(t["exemplars"][: args.num_exemplars].to(device))
                else:
                    exemplars.append(torch.tensor([]).to(device))
        labels = [t["labels"].to(device) for t in targets]

        bs = samples.tensors.shape[0]
        image_sizes = samples.imgsize()
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        input_captions = [cat_list[target["labels"][0]] + " ." for target in targets]
        cap_list = [[cat_list[target['labels'][0]]] for target in targets]
        if args.no_text:
            input_captions = [" ." for target in targets]

        print("input_captions: " + str(input_captions))

        img_id = targets[0]["image_id"].item()
        img_name = img_id_to_name[img_id]
        print(img_name)
        if args.use_synth_exemplars:
          synthetic_exemplars = exemplars_file[img_name]
          input_image_exemplars, exemplars = synthetic_transform(
          Image.open(os.path.join(synthetic_exemplar_dir, img_name)).convert("RGB"), {"exemplars": torch.tensor(synthetic_exemplars).unsqueeze(0)}
              )# Unsqueeze since only one synthetic exemplar per image, so need to add a dimension to align with cases where there are multiple exemplars for an image
          exemplars = exemplars["exemplars"]# only synthetic exemplars used

        with torch.cuda.amp.autocast(enabled=args.amp):
            # Use 'label' of 0 at inference since only input a single text prompt instead of all COCO classes.

            if args.use_synth_exemplars:
                outputs = model(
                    samples,
                    nested_tensor_from_tensor_list([input_image_exemplars.to(device)]),
                    [exemplars.to(device)],# Add batch dimension to synthetic exemplars
                    [],
                    [],
                    captions=input_captions,
                )
            else:
                outputs = model(
                    samples,
                    samples,
                    exemplars,
                    [],
                    [],
                    captions=input_captions,
                )                

            # Second forward pass with self-generated pseudo-exemplars if real, manually annotated exemplars are not available:
            if args.use_synth_exemplars or args.num_exemplars <= 0:
              pseudo_exemplars = get_pseudo_exemplars(outputs, image_sizes, [torch.tensor([0]).to(device) for t in targets], cap_list, input_captions, args.box_threshold, num_exemplars=args.num_pseudo_exemplars)
              
              outputs = model(
                  samples,
                  samples,
                  pseudo_exemplars,
                  [],
                  [],
                  captions=input_captions,
              )

            tokenized_captions = outputs["token"]
            # Convert pseudo-exemplar to visualization format.
            (img_h, img_w) = (image_sizes[0][0], image_sizes[0][1])
            (orig_h, orig_w) = (orig_target_sizes[0][0], orig_target_sizes[0][1])

            abs_errs_batch = get_count_errs(
                model,
                args,
                samples,
                exemplars,
                outputs,
                args.box_threshold,
                args.text_threshold,
                targets,
                tokenized_captions,
                input_captions,
                cap_list,
                predictor=predictor,
            )
            abs_errs += abs_errs_batch

        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res["boxes"]
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1
                )

                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())

                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
            
    count_mae = sum(abs_errs) / len(abs_errs)
    count_rmse = (np.array(abs_errs) ** 2).mean() ** (1 / 2)
    print("Total # of Images Tested: " + str(len(abs_errs)))
    print("Total MAE: " + str(count_mae) + ", Total RMSE: " + str(count_rmse))

    with open(args.fscd_gt_file, 'r') as in_file:
        gt_coco_fscd = json.load(in_file)
    gt_coco_fscd['annotations'] = countgd_detections
    pred_coco_fscd = gt_coco_fscd
    with open(args.coco_output_file, 'w') as out_file:
        json.dump(pred_coco_fscd, out_file)
    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, "results-{}.pkl".format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    
    with contextlib.redirect_stdout(io.StringIO()):
        # Suppress print output from unused [GroundingDINO] functions.
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
    
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]

    return count_mae, stats, coco_evaluator
