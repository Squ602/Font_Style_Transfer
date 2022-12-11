import copy
import random

import cv2
import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# from https://github.com/pschaldenbrand/StyleCLIPDraw/blob/master/Style_ClipDraw.ipynb
def render_drawing(shapes, shape_groups, canvas_width, canvas_height, n_iter, debug=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (
        1 - img[:, :, 3:4]
    )
    if debug:
        pydiffvg.imwrite(img.cpu(), "./katei.png", gamma=1.0)
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return img


# from https://github.com/pschaldenbrand/StyleCLIPDraw/blob/master/Style_ClipDraw.ipynb
def render_scaled(shapes, shape_groups, original_height, original_width, scale_factor=4, t=1):
    with torch.no_grad():
        shapes_resized = copy.deepcopy(shapes)
        for i in range(len(shapes)):
            shapes_resized[i].stroke_width = shapes[i].stroke_width * scale_factor
            for j in range(len(shapes[i].points)):
                shapes_resized[i].points[j] = shapes[i].points[j] * scale_factor
        img = render_drawing(
            shapes_resized,
            shape_groups,
            int(original_width * scale_factor),
            int(original_height * scale_factor),
            t,
        )
        return img


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode="bicubic")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def center_crop(img, w, h):
    """Center crop

    Args:
        img (numpy.array): input image
        w (int): width
        h (int): height

    Returns:
        crop_img: croped image
    """
    center_x = int(img.shape[1] / 2)
    center_y = int(img.shape[0] / 2)
    w2 = int(w / 2)
    h2 = int(h / 2)
    crop_img = img[center_y - h2 : center_y + h2, center_x - w2 : center_x + w2]
    return crop_img


def make_text_img(text, font_path):
    """return text image

    Args:
        text (string): text to convert

    Returns:
        PIL.Image : Image of text
    """
    ttfontname = font_path
    fontsize = 430
    canvasSize = (512, 512)
    backgroundRGB = (255, 255, 255)
    textRGB = (0, 0, 0)
    img = Image.new("RGB", canvasSize, backgroundRGB)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(ttfontname, fontsize)
    textWidth, textHeight = draw.textsize(text, font=font)
    textTopLeft = (
        canvasSize[0] // 2 - textWidth // 2,
        canvasSize[1] // 2 - textHeight // 2 - 70,
    )
    draw.text(textTopLeft, text, fill=textRGB, font=font)
    return img


def init_point(img, num_stroke):
    """Return init points

    Args:
        img (PIL.Image): Input Image
        num_stroke (np.array): Init points

    Returns:

    """
    img = np.array(img)
    points = np.where(img == 0.0)
    index = np.random.choice(list(range(len(points[0]))), num_stroke, replace=True)
    return points, index


def make_dist(img):
    """Make distans transform map

    Args:
        img (PIL.Image): Input image

    Returns:
        torch.tensor : distans transform map
    """
    np_img = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    dist = cv2.distanceTransform(gray, cv2.DIST_L2, maskSize=0)
    cv2.normalize(dist, dist, 0, 100.0, cv2.NORM_MINMAX)
    dist = torch.from_numpy(dist).to(torch.float32)
    dist = dist.pow(1.0)
    dist = dist.to(pydiffvg.get_device())
    return dist


def calc_dist(img, dist):
    target_dist = img.clone()
    target_dist = 255 - target_dist
    for i in range(3):
        target_dist[:, i, :, :] = target_dist[:, i, :, :] * dist

    return target_dist


def init_curves(opt, init_points, index, canvas_width, canvas_height):
    """_summary_

    Args:
        opt (_type_): _description_
        init_points (_type_): _description_
        index (_type_): _description_
        canvas_width (_type_): _description_
        canvas_height (_type_): _description_

    Returns:
        _type_: _description_
    """
    shapes = []
    shape_groups = []

    for i in range(opt.num_paths):
        num_segments = random.randint(3, 5) if opt.blob else random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (
            float(init_points[1][index[i]] / canvas_width),
            float(init_points[0][index[i]] / canvas_height),
        )
        points.append(p0)
        for j in range(num_segments):
            radius = 0.05
            p1 = (
                p0[0] + radius * (random.random() - 0.5),
                p0[1] + radius * (random.random() - 0.5),
            )
            p2 = (
                p1[0] + radius * (random.random() - 0.5),
                p1[1] + radius * (random.random() - 0.5),
            )
            p3 = (
                p2[0] + radius * (random.random() - 0.5),
                p2[1] + radius * (random.random() - 0.5),
            )
            points.append(p1)
            points.append(p2)
            if j < num_segments - 1:
                points.append(p3)
                p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(1.0),
            is_closed=opt.blob,
        )
        shapes.append(path)

        if opt.blob:
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=torch.tensor([opt.color[0], opt.color[1], opt.color[2], random.random()])
                if opt.color
                else torch.tensor([random.random(), random.random(), random.random(), random.random()]),
            )
        else:
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([random.random(), random.random(), random.random(), random.random()]),
            )
        shape_groups.append(path_group)

    return shapes, shape_groups
