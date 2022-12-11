import argparse
import os
import random
import warnings

import clip
import cv2
import numpy as np
import pydiffvg
import torch
import torchvision.transforms as transforms
import utils
from PIL import Image
from templates import imagenet_templates, text_templates
from tqdm import tqdm

warnings.simplefilter("ignore")

os.environ["FFMPEG_BINARY"] = "ffmpeg"

random.seed(1234)
torch.manual_seed(1234)

device = torch.device("cuda")
print(f"device : {device}\n")

pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

FONT_PATH = "./font-file/NotoSansJP-Regular.otf"

# Load the model
clip_model, preprocess = clip.load("ViT-B/32", device, jit=False)


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def font_style_transfer(opt):

    # ##### Make text image #######
    if opt.path is not None:
        text_img = Image.open(opt.path).convert("RGB")
        text_img = text_img.resize((512, 512))
    else:
        text_img = utils.make_text_img(opt.text, FONT_PATH)

    if opt.debug:
        name_dir = f"{opt.save_dir}process/{opt.text}_{opt.prompt}/"
        os.makedirs(name_dir, exist_ok=True)

    init_points, index = utils.init_point(text_img, opt.num_paths)
    dist = utils.make_dist(text_img)

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
        # Encode prompt
        template_text = compose_text_with_templates(opt.prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(pydiffvg.get_device())
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Encode text
        template_source = compose_text_with_templates(opt.source, text_templates)
        tokens_source = clip.tokenize(template_source).to(pydiffvg.get_device())
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        # Encode content image
        source_features = clip_model.encode_image(utils.clip_normalize(content_image, pydiffvg.get_device()))
        source_features /= source_features.clone().norm(dim=-1, keepdim=True)

    # caluculate distance
    target_dist = utils.calc_dist(content_image, dist)

    # Initialize Curves
    shapes, shape_groups = utils.init_curves(opt, init_points, index, canvas_width, canvas_height)
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
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
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # ################ Optimization  ################
    t = tqdm(range(opt.num_iter))
    for i in t:
        t.set_description(f"Iteration {i+1}")

        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()

        # ################ render the image ################
        img = utils.render_drawing(shapes, shape_groups, canvas_width, canvas_height, i, debug=opt.debug)

        # ################ shape loss ################
        img_dist = utils.calc_dist(img, dist)

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
        batch_features = clip_model.encode_image(utils.clip_normalize(im_batch, device))
        batch_features /= batch_features.clone().norm(dim=-1, keepdim=True)

        # calculate　ΔI
        img_direction = batch_features - source_features
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        # Calculate ΔT
        text_direction = (text_features - text_source).repeat(batch_features.size(0), 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        # Calculate cosine similarty & Threshold rejection
        loss_patch = 0
        loss_temp = 1 - torch.cosine_similarity(img_direction, text_direction, dim=1)
        loss_temp[loss_temp < opt.thresh] = 0
        loss_patch += loss_temp.mean()

        # ################ loss glob #####################
        # Encode redered image
        glob_features = clip_model.encode_image(utils.clip_normalize(img, device))
        glob_features /= glob_features.clone().norm(dim=-1, keepdim=True)

        # alculate　ΔI
        glob_direction = glob_features - source_features
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        # ################ reg_tv ################
        reg_tv = opt.lambda_tv * utils.get_image_prior_losses(img)

        # ################ total loss ################
        total_loss = opt.lambda_patch * loss_patch + opt.lambda_dir * loss_glob + reg_tv + opt.lambda_shape * shape_loss

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

        if opt.debug:
            img = img.detach().cpu().numpy()[0]
            img = img.transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            cv2.imwrite(
                os.path.join(name_dir, f"{i}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

        t.set_postfix({"render_loss": total_loss.item()})

    return (
        utils.render_scaled(
            shapes,
            shape_groups,
            canvas_width,
            canvas_height,
            t=opt.num_iter,
            scale_factor=opt.scale_factor,
        )
        .detach()
        .cpu()
        .numpy()[0]
    )


def gen_image(opt):
    print("prompt:", opt.prompt)
    print("text:", opt.text)
    img = font_style_transfer(opt)
    img = img.transpose(1, 2, 0)
    img = img * 255
    img = img.astype(np.uint8)

    if opt.path is not None and opt.text is None:
        basename = os.path.basename(opt.input_img_path)
        names = os.path.splitext(basename)
        name = names[0]
    else:
        name = opt.text

    os.makedirs(opt.save_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(opt.save_dir, f"{name}_{opt.prompt}_.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )
    print(f"finished {opt.prompt}\n")
    return img


def main(opt):
    if len(opt.text) > 1:
        orig_text = opt.text
        texts = [c for c in opt.text]
        images = []
        for text in texts:
            opt.text = text
            img = gen_image(opt)
            img = utils.center_crop(img, 420, 420)
            images.append(img)

        img_h = cv2.hconcat(images)
        cv2.imwrite(os.path.join(opt.save_dir, f"{orig_text}.png"), cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR))

    else:
        _ = gen_image(opt)


parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default="A", help="text")
parser.add_argument("--prompt", type=str, default="Starry Night by Vinvent van gogh", help="prompt")
parser.add_argument("--save_dir", type=str, default="./result/", help="save directory")

parser.add_argument("--thresh", type=float, default=0.7)
parser.add_argument("--lambda_tv", type=float, default=2e-3)
parser.add_argument("--lambda_patch", type=float, default=9000)
parser.add_argument("--lambda_dir", type=float, default=500)
parser.add_argument("--lambda_shape", type=float, default=1500)

parser.add_argument(
    "--color",
    nargs="*",
    default=None,
    type=float,
    help="initial color of beizer curves [R, G, B](value range 0~1). If not specified, None.",
)
parser.add_argument("--source", type=str, default="a photo")

parser.add_argument("--num_iter", type=int, default=200, help="Number of iterations")
parser.add_argument("--num_paths", type=int, default=512, help="Number of bezeir curves")
parser.add_argument("--max_width", type=float, default=2.0, help="max width of curves")
parser.add_argument("--crop_size", type=int, default=160, help="cropped image size")
parser.add_argument("--num_augs", type=int, default=64, help="number of patches")

parser.add_argument("--blob", type=bool, default=True, help="use closed bezier curves")
parser.add_argument("--debug", type=bool, default=False, help="save process images")

parser.add_argument("--scale_factor", type=int, default=1, help="output image size is 512*scale_factor")
parser.add_argument("--path", type=str, default=None, help="input image path")

opt = parser.parse_args()


class Config:
    # parameters
    thresh = 0.7  # default 0.7
    lambda_tv = 2e-3  # default 2e-3
    lambda_patch = 9000.0  # default 9000
    lambda_dir = 500  # default 500
    lambda_shape = 1500  # lambda_shape of shape loss. default 1500

    color = None  # initial color of beizer curves [R, G, B]. If not specified, None.
    source = "a photo"  # default "a photo"

    num_iter = 200  # default 200
    num_paths = 512  # number of bezier curves. default 512
    max_width = 2.0  # max width of bezier curves. default 2.0
    crop_size = 160  # default 160
    num_augs = 64  # number of augment. default 64

    blob = True  # use closed bezier curves
    debug = False  # save process image

    # Change as you like
    prompt = "Starry Night by Vincent van gogh"
    text = "ST"  # "如"

    save_dir = "./result/"
    scale_factor = 1  # output image size is 512*scale_factor
    path = None  # input image path


if __name__ == "__main__":
    # opt = Config()
    main(opt)
