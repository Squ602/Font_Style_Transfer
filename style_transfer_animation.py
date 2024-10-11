import argparse
import os
import subprocess
import warnings
from functools import partial
from typing import List

import clip
import cv2
import numpy as np
import pydiffvg
import torch
import torchvision.transforms as transforms
from PIL import Image
from templates import image_templates, text_templates
from tqdm import tqdm as std_tqdm
from utils import (
    calc_dist,
    center_crop,
    clip_normalize,
    compose_text_with_templates,
    init_curves,
    init_point,
    make_dist,
    make_text_img,
    render_drawing,
    render_scaled,
    seed_everything,
)

tqdm = partial(std_tqdm, dynamic_ncols=True)
warnings.simplefilter("ignore")

device = torch.device("cuda")
print(f"device : {device}\n")

pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

# Load the model
clip_model, preprocess = clip.load("ViT-B/32", device, jit=False)


def encode_prompt(prompt: str):
    with torch.no_grad():
        # Encode prompt
        template_text = compose_text_with_templates(prompt, image_templates)
        tokens = clip.tokenize(template_text).to(pydiffvg.get_device())
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def font_style_transfer_animation(opt):
    # ##### Make text image #######
    if opt.path is not None:
        text_img = Image.open(opt.path).convert("RGB")
        text_img = text_img.resize((512, 512))
    else:
        text_img = make_text_img(opt.text, opt.font_path)

    opt.save_dir = os.path.join(
        opt.save_dir,
        f"{opt.prompt[0].replace(' ', '_')}_to_{opt.prompt[-1].replace(' ', '_')}_{opt.text}",
    )
    os.makedirs(opt.save_dir, exist_ok=True)

    save_image_dir = os.path.join(opt.save_dir, "images")
    os.makedirs(save_image_dir, exist_ok=True)

    init_points, index = init_point(text_img, opt.num_paths)
    dist = make_dist(text_img)

    gamma = 1.0
    content_image = np.array(text_img)
    content_image = torch.from_numpy(content_image).to(torch.float32) / 255.0
    content_image = content_image.pow(gamma)
    content_image = content_image.to(pydiffvg.get_device())
    content_image = content_image.unsqueeze(0)
    content_image = content_image.permute(0, 3, 1, 2)  # NHWC -> NCHW
    canvas_width, canvas_height = content_image.shape[3], content_image.shape[2]

    cropper = transforms.Compose(
        [
            transforms.RandomCrop(opt.crop_size),
        ]
    )
    augment = transforms.Compose(
        [
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224),
        ]
    )

    with torch.no_grad():
        # Encode text
        template_source = compose_text_with_templates(opt.source, text_templates)
        tokens_source = clip.tokenize(template_source).to(pydiffvg.get_device())
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        # Encode content image
        source_features = clip_model.encode_image(
            clip_normalize(content_image, pydiffvg.get_device())
        )
        source_features /= source_features.clone().norm(dim=-1, keepdim=True)

    # caluculate distance
    target_dist = calc_dist(content_image, dist)

    # Initialize Curves
    shapes, shape_groups = init_curves(
        opt, init_points, index, canvas_width, canvas_height
    )
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,
        *scene_args,
    )

    points_vars = []
    stroke_width_vars = []
    color_vars = []

    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not opt.blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if opt.blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

    # Optimizer
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.05)

    # ################ Optimization  ################
    for num, prompt in enumerate(opt.prompt):
        text_features = encode_prompt(prompt)
        t = tqdm(range(opt.num_iter))
        for i in t:
            t.set_description(f"Iteration {i+1}")

            points_optim.zero_grad()
            if len(stroke_width_vars) > 0:
                width_optim.zero_grad()
            color_optim.zero_grad()

            # ################ render the image ################
            img = render_drawing(
                shapes,
                shape_groups,
                canvas_width,
                canvas_height,
                opt.seed,
                debug=opt.debug,
            )

            # ################ shape loss ################
            img_dist = calc_dist(img, dist)

            shape_loss = (img_dist - target_dist).pow(2).mean()
            # ################ augment ################
            img_proc = []
            for _ in range(opt.num_augs):
                target_crop = cropper(img)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
            img_proc = torch.cat(img_proc, dim=0)
            im_batch = img_proc

            # ################ loss patch  ################
            # Encode augmented image
            batch_features = clip_model.encode_image(clip_normalize(im_batch, device))
            batch_features /= batch_features.clone().norm(dim=-1, keepdim=True)

            # calculate　ΔI
            img_direction = batch_features - source_features
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            # Calculate ΔT
            text_direction = (text_features - text_source).repeat(
                batch_features.size(0), 1
            )
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

            # Calculate cosine similarty & Threshold rejection
            loss_patch = 0
            loss_temp = 1 - torch.cosine_similarity(
                img_direction, text_direction, dim=1
            )
            loss_temp[loss_temp < opt.thresh] = 0
            loss_patch += loss_temp.mean()

            # ################ total loss ################
            total_loss = opt.lambda_patch * loss_patch + opt.lambda_shape * shape_loss

            # ################ Backpropagate the gradients ################
            total_loss.backward()

            # ################ Take a gradient descent step ################
            points_optim.step()
            if len(stroke_width_vars) > 0:
                width_optim.step()
            color_optim.step()

            if len(stroke_width_vars) > 0:
                for path in shapes:
                    path.stroke_width.data.clamp_(1.0, opt.max_width)
            if opt.blob:
                for group in shape_groups:
                    group.fill_color.data.clamp_(0.0, 1.0)
            else:
                for group in shape_groups:
                    group.stroke_color.data.clamp_(0.0, 1.0)

            img = img.detach().cpu().numpy()[0]
            img = img.transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            save_path = os.path.join(
                save_image_dir, f"{(opt.num_iter*num + i):04d}.png"
            )
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            t.set_postfix({"render_loss": total_loss.item()})

    return (
        render_scaled(
            shapes,
            shape_groups,
            canvas_width,
            canvas_height,
            opt.seed,
            scale_factor=opt.scale_factor,
        )
        .detach()
        .cpu()
        .numpy()[0]
    )


def gen_image(opt):
    print("text:", opt.text)
    img = font_style_transfer_animation(opt)
    img = img.transpose(1, 2, 0)
    img = img * 255
    img = img.astype(np.uint8)

    print(f"finished {opt.text}\n")
    return img


def main(opt):
    assert len(opt.text) == 1, "The number of characters must be 1"
    _ = gen_image(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="C", help="text")
    parser.add_argument(
        "--prompt",
        type=List[str],
        default=["Starry Night by Vincent van Gogh", "Sunflowers by Vincent van Gogh"],
        help="prompt",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="./font-file/NotoSansJP-Regular.otf",
        help="font path",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./result/animation/", help="save directory"
    )

    parser.add_argument("--thresh", type=float, default=0.7)
    parser.add_argument("--lambda_patch", type=float, default=9000)
    parser.add_argument("--lambda_shape", type=float, default=1500)

    parser.add_argument(
        "--color",
        nargs="*",
        default=None,
        type=float,
        help="Initial color of beizer curves [R, G, B](value range 0~1). If not specified, None.",
    )
    parser.add_argument("--source", type=str, default="a photo")

    parser.add_argument(
        "--num_iter", type=int, default=200, help="Number of iterations"
    )
    parser.add_argument(
        "--num_paths", type=int, default=512, help="Number of bezeir curves"
    )
    parser.add_argument(
        "--max_width", type=float, default=2.0, help="Max width of curves"
    )
    parser.add_argument("--crop_size", type=int, default=160, help="Cropped image size")
    parser.add_argument("--num_augs", type=int, default=64, help="Number of patches")

    parser.add_argument(
        "--blob", type=bool, default=True, help="Use closed bezier curves"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save process image with file of same name",
    )

    parser.add_argument(
        "--scale_factor",
        type=int,
        default=1,
        help="Output image size is 512*scale_factor",
    )
    parser.add_argument("--path", type=str, default=None, help="Input image path")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    opt = parser.parse_args()
    seed_everything(opt.seed)
    main(opt)
    subprocess.run(
        f"ffmpeg -r 20 -i {opt.save_dir}/images/%04d.png -vcodec libx264 -pix_fmt yuv420p {opt.save_dir}/output.mp4",
        shell=True,
    )
