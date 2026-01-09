import glob
import random
import torch
from PIL import Image
import numpy as np
import argparse
import json
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms_app as T
import glob
import os
from scipy.io import loadmat
from util.misc import nested_tensor_from_tensor_list

def get_args_parser():
    parser = argparse.ArgumentParser("Testing on ShanghaiTech with pseudo-exemplars", add_help=False)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="checkpoint to load",
        default="checkpoints/countgd_plusplus.pth",
    )
    parser.add_argument(
        "--config",
        help="config file",
        default="cfg_app.py",
    )
    parser.add_argument(
        "--image_folder",
        default="data/ShanghaiTech/part_B/test_data/images",
        help="folder path for the images",
    )
    parser.add_argument(
        "--gt_folder",
        default="data/ShanghaiTech/part_B/test_data/ground-truth",
        help="folder path for the ground truth"
    )
    parser.add_argument(
        "--text",
        default='human',
        help="input text description",
    )
    parser.add_argument(
        "--confidence_thresh", help="confidence threshold for model", default=0.23, type=float
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser

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
        boxes_sample = boxes[batch_ind, :, :]
        tokenized_sample = tokenized[batch_ind]
        caption_sample = captions[batch_ind]
        cat_list_sample = cat_list[batch_ind]
        label_sample = labels_uncropped[batch_ind][0] 
        start_ind = caption_sample.find(cat_list_sample[label_sample])
        end_ind = start_ind + len(cat_list_sample[label_sample]) - 1
        beg_pos = tokenized_sample.char_to_token(start_ind)
        end_pos = tokenized_sample.char_to_token(end_ind)
        logits_sample = logits_sample[:, beg_pos: (end_pos + 1)]
        scores_sample = logits_sample.max(dim=-1).values

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
    cfg = SLConfig.fromfile(args.config)
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    device = torch.device(args.device)
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

countgd_plusplus_shanghaitech_results = {}
shanghaitech_gt = {}
parser = argparse.ArgumentParser("Testing Counting Model", parents=[get_args_parser()])
args = parser.parse_args()
model, transform = build_model_and_transforms(args)
model = model.cuda()
# Get images and GT.
images = glob.glob(os.path.join(args.image_folder, '*.jpg'))
gt_counts = glob.glob(os.path.join(args.gt_folder, '*.mat'))
gt_counts = {(os.path.basename(gt_file)[3:-3] + "jpg"): int(loadmat(gt_file)['image_info'][0,0]['number'][0,0]) for gt_file in gt_counts}
print(gt_counts)
abs_errs = []
img_counter = 0
num_images = len(images)
for long_image_name in images:
  image_name = os.path.basename(long_image_name)
  gt_count = gt_counts[image_name]
  input_image, target = transform(Image.open(long_image_name).convert("RGB"), {"exemplars": torch.tensor([])})
  input_image = input_image.cuda()
  input_exemplar = target["exemplars"].cuda()
  input_text = args.text
  with torch.no_grad():
      outputs = model(
          nested_tensor_from_tensor_list(input_image.unsqueeze(0)),
          nested_tensor_from_tensor_list(input_image.unsqueeze(0)),
          [input_exemplar],
          [],
          [],
          captions=[input_text + " ."],
      )

      # Second forward pass with self-generated exemplars:
      pseudo_exemplars = get_pseudo_exemplars(outputs, [(input_image.size()[-2], input_image.size()[-1])], [torch.tensor([0]).cuda()], [[input_text]], [input_text + " ."], args.confidence_thresh)

      model_output = model(
          nested_tensor_from_tensor_list(input_image.unsqueeze(0)),
          nested_tensor_from_tensor_list(input_image.unsqueeze(0)),
          pseudo_exemplars,
          [],
          [],
          captions=[input_text + " ."],
      )

  logits = model_output["pred_logits"][0].sigmoid()
  boxes = model_output["pred_boxes"][0]

  # Only keep boxes with confidence above threshold.
  box_mask = logits.max(dim=-1).values > args.confidence_thresh
  logits = logits[box_mask, :]
  boxes = boxes[box_mask, :]
  pred_count = boxes.shape[0]

  countgd_plusplus_shanghaitech_results[image_name] = int(pred_count)
  shanghaitech_gt[image_name] = gt_count

  img_counter+=1
  print("Processed image " + str(img_counter) + "/" + str(num_images))
  print("Pred Count: " + str(pred_count) + ", GT Count: " + str(gt_count))
  abs_errs.append(abs(gt_count - pred_count))

abs_errs = np.array(abs_errs)
mae = np.mean(abs_errs)
rmse = np.sqrt(np.mean(abs_errs ** 2))

print("MAE: " + str(mae))
print("RMSE: " + str(rmse))
print("Num Images: " + str(len(abs_errs)))

