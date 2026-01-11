import spaces
import gradio as gr
import random
import torch
from PIL import Image
import json
import numpy as np
import argparse
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms_app as T
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# https://github.com/PhyscalX/gradio-image-prompter/tree/main/backend/gradio_image_prompter/templates/component
import io
from enum import Enum
import os
cwd = os.getcwd()
# Suppress warnings to avoid overflowing the log.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.patches as patches

from gradio_image_prompter import ImagePrompter



CONF_THRESH = 0.23
MAX_NEGS = 10

# MODEL:
def get_args_parser():
    parser = argparse.ArgumentParser("CountGD++ Demo", add_help=False)
    parser.add_argument(
        "--device", default="cuda", help="device to use for inference"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="checkpoints/countgd_plusplus.pth",
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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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

# APP:
def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])

    return box_inputs

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

def generate_heatmap(image, boxes, scores=None):
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

    plt.axis('off')

    # Plot bounding boxes
    (x0, y0, box_w, box_h) = get_box_coords_from_boxes(image, boxes)

    for box_ind in range(boxes.shape[0]):
        (x_i, y_i, box_w_i, box_h_i) = (x0[box_ind], y0[box_ind], box_w[box_ind], box_h[box_ind])
        
        # Draw black border (slightly thicker, drawn first)
        rect_border = patches.Rectangle(
            (x_i, y_i), box_w_i, box_h_i,
            linewidth=2.5, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect_border)
    
        if scores is not None:
            s = float(scores[box_ind])
            ax.text(
                x_i, y_i, f"{s:.2f}",              # text position and content
                fontsize=12, color="white",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
                ha="left", va="top",               # top-left corner of box
                zorder=5, clip_on=True
            )

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()

    output_img = Image.open(img_buf)
    return output_img
    
def preprocess(transform, image, input_prompts = None):
    if input_prompts == None:
        prompts = { "image": image, "points": []}
    else:
        prompts = input_prompts

    input_image, _ = transform(image, None)
    exemplar = get_box_inputs(prompts["points"])
    # Wrapping exemplar in a dictionary to apply only relevant transforms
    input_image_exemplar, exemplar = transform(prompts['image'], {"exemplars": torch.tensor(exemplar)})
    exemplar = exemplar["exemplars"]

    return input_image, input_image_exemplar, exemplar

def get_boxes_from_prediction(model_output):
    input_ids = model_output["input_ids"][0]
    print("input_ids")
    print(input_ids)
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    print("logits")
    print(logits[:, :len(input_ids)])
    boxes = model_output["pred_boxes"][0]

    # [pos_neg_split_idx] is the index of the first occurence of the "." separating token.
    for idx in range(len(input_ids)):
        token = input_ids[idx]
        if token == 1012:
            pos_neg_split_idx = idx
            break

    print("pos/neg split idx: " + str(pos_neg_split_idx))
    
    pos_logits = logits[:, :(pos_neg_split_idx + 1)]
    neg_logits = logits[:, (pos_neg_split_idx + 1):]

    # Stage 1 filtering:
    box_mask = pos_logits.max(dim=-1).values > CONF_THRESH
    boxes = boxes[box_mask, :]
    logits = logits[box_mask, :]
    scores = logits.max(dim=-1).values

    # Stage 2 filtering:
    pos_logits = pos_logits[box_mask, :]
    neg_logits = neg_logits[box_mask, :]
    box_mask = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
    boxes = boxes[box_mask, :].cpu().numpy()
    logits = logits[box_mask, :].cpu().numpy()
    scores = scores[box_mask].cpu().numpy()

    return boxes, scores

def predict(model, transform, image, positive_text, positive_prompts, negative_texts, negative_prompts, device):
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
        print("pos exemps: " + str(pos_exemplars))
        model_output = model(
                nested_tensor_from_tensor_list(input_images),
                nested_tensor_from_tensor_list(input_image_pos_exemplars),
                pos_exemplars,
                negative_images,
                neg_exemplars,
                captions=[caption],
            )
        
    return get_boxes_from_prediction(model_output)

examples = [
    ["img/336.jpg", "strawberry", {"image": "img/synthetic_strawberry.jpg"}, {"image": "img/synthetic_blueberry.jpg"}, "blueberry"],
    ["img/336.jpg", "blueberry", {"image": "img/synthetic_blueberry.jpg"}, {"image": "img/synthetic_strawberry.jpg"}, "strawberry"],
    ["img/blood_cells_teaser.jpg", "red blood cell", {"image": "img/blood_cells_teaser.jpg"}, {"image": "img/blood_cells_teaser.jpg"}, "white blood cell"],
    ["img/black_and_white_marbles.jpg", "black marble", {"image": "img/synthetic_black_marble.jpg"}, {"image": "img/synthetic_white_marble.jpg"}, "white marble"],
    ["img/black_and_white_marbles.jpg", "white marble", {"image": "img/synthetic_white_marble.jpg"}, {"image": "img/synthetic_black_marble.jpg"}, "black marble"],
    ["img/penguins.jpg", "penguin", {"image": "img/penguins.jpg"}, {"image": "img/penguins.jpg"}, ""],
    ["img/penguins-dark.jpg", "penguin", {"image": "img/penguins-dark.jpg"}, {"image": "img/penguins-dark.jpg"}, ""],
    ["img/utensils.jpg", "utensil", {"image": "img/utensils.jpg"}, {"image": "img/utensils.jpg"}, ""],
    ["img/utensils.jpg", "fork", {"image": "img/utensils.jpg"}, {"image": "img/utensils.jpg"}, "spoon"],
    ["img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg", "yellow soybean", {"image": "img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg"}, {"image": "img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg"}, "brown peppercorn"],
    ["img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg", "brown peppercorn", {"image": "img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg"}, {"image": "img/FOO_INTER_BEA2_PEP1_101_089_b3e939.jpg"}, "yellow soybean"],
    ["img/humans.jpg", "human", {"image": "img/humans.jpg"}, {"image": "img/humans.jpg"}, ""],
    ["img/cars.jpg", "car", {"image": "img/cars.jpg"}, {"image": "img/cars.jpg"}, ""],
    ["img/cars.jpg", "yellow car", {"image": "img/cars.jpg"}, {"image": "img/cars.jpg"}, "red car"]
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Counting Application", parents=[get_args_parser()])
    args = parser.parse_args()
    device = get_device()
    model, transform = build_model_and_transforms(args)
    model = model.to(device)

    def _predict(image, positive_text, positive_prompts, negative_text, negative_prompts):
        return predict(model, transform, image, positive_text, positive_prompts, negative_text, negative_prompts, device)


    @spaces.GPU
    def count_main(image, positive_text, positive_prompts, negative_idx, *negative_outputs):
        if positive_prompts is None:
            positive_prompts = {"image": image, "points": []}
        negative_texts = []
        negative_prompts = []
        # [negative_idx] indicates up to what index is visible.
        for ind in range(negative_idx):
            (negative_prompt, negative_text) = negative_outputs[2 * ind], negative_outputs[2 * ind + 1]
            if negative_prompt is not None:
                negative_prompts.append(negative_prompt)
                negative_texts.append(negative_text)
            elif len(negative_text) > 0:
                negative_prompts.append({"image": image, "points": []})
                negative_texts.append(negative_text)
        boxes, scores = _predict(image, positive_text, positive_prompts, negative_texts, negative_prompts)
        predicted_count = len(boxes)
        output_img = generate_heatmap(image, boxes, scores=None) # Set [scores] to the [scores] tensor computed above to visualize the confidence scores
        num_positive_exemplars = len(get_box_inputs(positive_prompts["points"]))

        return (gr.Image(output_img, visible=True), gr.Number(label="Predicted Count", visible=True, value=predicted_count))

    def save_prompts(positive_text, positive_prompts, negative_idx, *negative_outputs):
        # Create directory to save prompts
        os.makedirs("saved_prompts", exist_ok=True)

        prompt_json = {
            "positive": {
                "text": positive_text, 
                "exemplars":{
                    "image": os.path.join("saved_prompts", "pos_exemplar_image.png"), 
                    "boxes":[]
                }
            }, 
            "negative": []
        }

        # Save positive prompts
        positive_prompts["image"].convert("RGB").save(os.path.join("saved_prompts", "pos_exemplar_image.png"))
        prompt_json["positive"]["exemplars"]["boxes"] = get_box_inputs(positive_prompts["points"])

        # Save negative prompts
        # [negative_idx] indicates up to what index is visible.
        for ind in range(negative_idx):
            (negative_prompt, negative_text) = negative_outputs[2 * ind], negative_outputs[2 * ind + 1]
            prompt_json["negative"].append({
                "text": negative_text,
                "exemplars": {"image":os.path.join("saved_prompts", "neg_exemplar_image_" + str(ind) + ".png"), "boxes": get_box_inputs(negative_prompt["points"])}
            })
            negative_prompt["image"].convert("RGB").save(os.path.join("saved_prompts", "neg_exemplar_image_" + str(ind) + ".png"))

        with open(os.path.join("saved_prompts", "prompts.json"), 'w') as out_f:
            json.dump(prompt_json, out_f)

    def add_neg(idx):
        """Reveal the next hidden pair. Return one update per output + the new idx."""
        next_idx = min(idx + 1, MAX_NEGS)
        wrapper_updates = [gr.Group(visible=(i < next_idx)) for i in range(MAX_NEGS)]

        # First output is the updated state; the rest are component updates
        return [idx + 1] + wrapper_updates

    with gr.Blocks(title="CountGD++: Generalized Prompting for Open-World Counting", theme="soft", head="""<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=1">""") as demo:

        gr.Markdown(
            """
            # <center>CountGD++: Generalized Prompting for Open-World Counting
            <center><h3>Count objects with positive and/or negative text, positive and/or negative visual exemplars, or both together.</h3>
            <h3>Scroll down to try more examples</h3>
            <h3><a href='https://arxiv.org/abs/2512.23351' target='_blank' rel='noopener'>[paper]</a>
                <a href='https://github.com/niki-amini-naieni/CountGDPlusPlus' target='_blank' rel='noopener'>[code]</a></h3></center>
                <center>limitation: pseudo-exemplars and adaptive cropping are not implemented in the app.</center>
            """
            )

        with gr.Row():
            with gr.Column():
                input_image_main = gr.Image(type='pil', label='Input Image', show_label='True', value="img/336.jpg", interactive=True)
                positive_text_main = gr.Textbox(label="What would you like to count?", placeholder="", value="strawberry")
                positive_exemplar_image_main = ImagePrompter(type='pil', label='Visual Exemplar Image (+)', show_label=True, value={"image": "img/synthetic_strawberry.jpg", "points": []}, interactive=True)
                # Add negative examples
                # Precreate pairs: (image uploader, text)
                neg_slots = []
                shells = []
                with gr.Column():
                    for i in range(MAX_NEGS):
                        with gr.Group(visible=False) as shell:
                            if i > 0:
                                img = ImagePrompter(type='pil', label='Visual Exemplar Image (-)', show_label=True, interactive=True)
                                txt = gr.Textbox(label="What would you not like to count?", placeholder="", value="")
                                neg_slots.append((img, txt))
                                shells.append(shell)
                            else:
                                img = ImagePrompter(type='pil', label='Visual Exemplar Image (-)', show_label=True, value={"image": "img/synthetic_blueberry.jpg", "points": []}, interactive=True)
                                txt = gr.Textbox(label="What would you not like to count?", placeholder="blueberry", value="")
                                neg_slots.append((img, txt))
                                shells.append(shell)                               
            
                # Flatten outputs list once
                neg_outputs = [c for pair in neg_slots for c in pair]
                neg_idx = gr.State(0)
                add_neg_button = gr.Button("Add Negative", variant="primary")
                add_neg_button.click(add_neg, inputs=neg_idx, outputs=[neg_idx] + shells)
            with gr.Column():
                detected_instances_main = gr.Image(label="Detected Instances", show_label='True', interactive=False)
                pred_count_main = gr.Number(label="Predicted Count")
                submit_btn_main = gr.Button("Count", variant="primary")
                save_prompts_btn = gr.Button("Save Prompts", variant="primary")
                clear_btn_main = gr.ClearButton(variant="secondary")
        gr.Examples(label="Examples: click on a row to load the example. Add visual exemplars by drawing boxes on the loaded \"Visual Exemplar Image.\"", examples=examples, inputs=[input_image_main, positive_text_main, positive_exemplar_image_main] + neg_outputs)
        submit_btn_main.click(fn=count_main, inputs=[input_image_main, positive_text_main, positive_exemplar_image_main, neg_idx] + neg_outputs, outputs=[detected_instances_main, pred_count_main])
        save_prompts_btn.click(fn=save_prompts, inputs=[positive_text_main, positive_exemplar_image_main, neg_idx] + neg_outputs)
        clear_btn_main.add([input_image_main, positive_text_main, positive_exemplar_image_main, detected_instances_main, pred_count_main] + neg_outputs)


    demo.queue().launch(share=True)


