from typing import Any
import sys
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

sys.path.append("/kaggle/input/imc25-codes/mast3r")

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def revert_rotate(kpt, direction, width, height):
    if direction == 0:
        rotated_kpt = kpt
    elif direction == 90:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = kpt[:, 1]
        rotated_kpt[:, 1] = width - 1 - kpt[:, 0]
    elif direction == 180:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = width - 1 - kpt[:, 0]
        rotated_kpt[:, 1] = height - 1 - kpt[:, 1]
    elif direction == 270:
        rotated_kpt = np.zeros_like(kpt)
        rotated_kpt[:, 0] = height - 1 - kpt[:, 1]
        rotated_kpt[:, 1] = kpt[:, 0]
    return rotated_kpt


def keypoint_matching_MAST3R(
    image_pairs: list[tuple[str, str]],
    dir_pairs: list[tuple[int, int]],
    images_dir: str,
    # sample_thresh: float = 0.5,
    image_size: list = [1024, 1024],
    max_matching_num: int = 10000,
    min_matches: int = 15,
    rects: dict | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[Any, Any]:
    """
    # Matching
    images_dir = params["data_dict"][0].parent
    image_pairs = list(
        zip(image_pair_df["key1"].values.tolist(), image_pair_df["key2"].values.tolist())
    )
    dir_pairs = list(
        zip(image_pair_df["dir1"].values.tolist(), image_pair_df["dir2"].values.tolist())
    )
    keypoints, matches = keypoint_matcing_DKM(
        image_pairs,
        dir_pairs,
        images_dir,
        **params["keypoint_matching_args"],
        rects=rects,
        device=params["device"],
    )
    """
    size = image_size[0]
    square_ok = False

    model_name = "/kaggle/input/imc-models/pytorch/bundle/1/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_name, img_size=image_size)
    model = model.to(device)

    keypoints = {}
    matches = {}
    for pair, dir_pair in tqdm(zip(image_pairs, dir_pairs), total=len(image_pairs)):
        key1, key2 = pair
        dir1, dir2 = dir_pair

        img1 = cv2.imread(str(images_dir / key1), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(images_dir / key2), cv2.IMREAD_COLOR)

        # Crop
        if rects is not None and key1 in rects:
            # sfm mkpc
            if type(rects[key1]) == list:
                rect1 = rects[key1]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect1 = rects[key1][key2]
            img1 = img1[rect1[1] : rect1[3], rect1[0] : rect1[2], :]
        else:
            rect1 = [0, 0, 0, 0]

        if rects is not None and key2 in rects:
            # sfm mkpc
            if type(rects[key2]) == list:
                rect2 = rects[key2]
            # pair mkpc
            elif type(rects[key1]) == dict:
                rect2 = rects[key2][key1]
            img2 = img2[rect2[1] : rect2[3], rect2[0] : rect2[2], :]
        else:
            rect2 = [0, 0, 0, 0]

        # Rotate
        if dir1 == 90:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif dir1 == 180:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif dir1 == 270:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if dir2 == 90:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
        elif dir2 == 180:
            img2 = cv2.rotate(img2, cv2.ROTATE_180)
        elif dir2 == 270:
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        imgs = []
        img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        for img in [img1PIL, img2PIL]:
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)

            W, H = img.size
            cx, cy = W // 2, H // 2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not (square_ok) and W == H:
                    halfh = 3 * halfw / 4
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

            W2, H2 = img.size
            imgs.append(
                dict(
                    img=ImgNorm(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                )
            )

        output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]
        desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2**13
        )

        # ignore small border around the edge
        H0, W0 = view1["true_shape"][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3)
            & (matches_im0[:, 0] < int(W0) - 3)
            & (matches_im0[:, 1] >= 3)
            & (matches_im0[:, 1] < int(H0) - 3)
        )

        H1, W1 = view2["true_shape"][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3)
            & (matches_im1[:, 0] < int(W1) - 3)
            & (matches_im1[:, 1] >= 3)
            & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        W_A, H_A = img1PIL.size
        W_B, H_B = img2PIL.size
        kpts1 = matches_im0.astype(np.float32)
        kpts2 = matches_im1.astype(np.float32)
        if kpts1.shape[0] < min_matches or kpts2.shape[0] < min_matches:
            continue

        # Rescale keypoints
        kpts1[:, 0] = kpts1[:, 0] * W_A / int(W0)
        kpts1[:, 1] = kpts1[:, 1] * H_A / int(H0)
        kpts2[:, 0] = kpts2[:, 0] * W_B / int(W1)
        kpts2[:, 1] = kpts2[:, 1] * H_B / int(H1)

        # Revert Rotate
        kpts1 = revert_rotate(kpts1, dir1, W_A, H_A)
        kpts2 = revert_rotate(kpts2, dir2, W_B, H_B)

        # Revert Crop
        kpts1[:, 0] += rect1[0]
        kpts1[:, 1] += rect1[1]
        kpts2[:, 0] += rect2[0]
        kpts2[:, 1] += rect2[1]

        if key1 not in keypoints:
            keypoints[key1] = kpts1
            matches1 = np.array(list(range(0, kpts1.shape[0])))
        else:
            n = keypoints[key1].shape[0]
            keypoints[key1] = np.concatenate([keypoints[key1], kpts1])
            matches1 = np.array(list(range(0, kpts1.shape[0]))) + n

        if key2 not in keypoints:
            keypoints[key2] = kpts2
            matches2 = np.array(list(range(0, kpts2.shape[0])))
        else:
            n = keypoints[key2].shape[0]
            keypoints[key2] = np.concatenate([keypoints[key2], kpts2])
            matches2 = np.array(list(range(0, kpts2.shape[0]))) + n

        _matches = np.zeros((matches1.shape[0], 2)).astype(np.int64)
        _matches[:, 0] = matches1
        _matches[:, 1] = matches2
        matches.setdefault(key1, {})
        matches[key1][key2] = _matches

    return keypoints, matches
