from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import h5py


# circle keypoints on the image (PIL Image)
# 3x3 circle, light green
def draw_keypoints(im, kpts, color=(0, 255, 0)):
    for kpt in kpts:
        x, y = int(kpt[0]), int(kpt[1])
        # draw circle
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                im.putpixel((x + i, y + j), color)
    return im


# 1x1 dot, light green
def draw_keypoints_dot(im, kpts):
    for kpt in kpts:
        x, y = int(kpt[0]), int(kpt[1])
        im.putpixel((x, y), (0, 255, 0))
    return im


# 2x2 square, light green
def draw_keypoints_square(im, kpts, color=(0, 255, 0)):
    for kpt in kpts:
        x, y = int(kpt[0]), int(kpt[1])
        # draw square
        for i in range(-1, 1):
            for j in range(-1, 1):
                im.putpixel((x + i, y + j), color)
    return im


def inversed_idxes(kpts1, idxs):
    inversed_idxs = []
    for i in range(len(kpts1)):
        if i not in idxs:
            inversed_idxs.append(i)
    return np.array(inversed_idxs)


# draw yellow lines between matched keypoints
def draw_lines(im, kpts1, kpts2, matches, w1):
    matched_kpts1 = kpts1[matches[:, 0]]
    matched_kpts2 = kpts2[matches[:, 1]] + np.array([w1, 0])
    draw = ImageDraw.Draw(im)
    for i in range(len(matched_kpts1)):
        x1, y1 = int(matched_kpts1[i][0]), int(matched_kpts1[i][1])
        x2, y2 = int(matched_kpts2[i][0]), int(matched_kpts2[i][1])
        draw.line((x1, y1, x2, y2), fill=(150, 255, 0), width=1)
    return im


class VisualizeMatches:
    def __init__(self, image_path1: str, image_path2: str, keypoints_file: str, matches_file: str):
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.keypoints_file = keypoints_file
        self.matches_file = matches_file

    def _preproc(self):
        im1 = Image.open(self.image_path1)
        im2 = Image.open(self.image_path2)
        im1_np = np.array(im1)
        im2_np = np.array(im2)

        h1, w1 = im1_np.shape[:2]
        h2, w2 = im2_np.shape[:2]
        h_max = max(h1, h2)
        w = w1 + w2

        filename1, filename2 = self.image_path1.split("/")[-1], self.image_path2.split("/")[-1]
        with h5py.File(self.keypoints_file, mode="r") as f_keypoints:
            kpts1 = f_keypoints[filename1][:]  # type: ignore
            kpts2 = f_keypoints[filename2][:]  # type: ignore
        with h5py.File(self.matches_file, mode="r") as f_keypoints:
            matches = f_keypoints[filename1][filename2][:]  # type: ignore

        im = Image.new("RGB", (w, h_max))
        im.paste(im1, (0, 0))
        im.paste(im2, (w1, 0))

        # draw keypoints on the image
        not_matched = inversed_idxes(kpts1, matches[:, 0])
        if not_matched.shape[0] > 0:
            im = draw_keypoints(im, kpts1[not_matched], color=(255, 0, 0))
        im = draw_keypoints(im, kpts1[matches[:, 0]], color=(0, 255, 0))
        not_matched = inversed_idxes(kpts2, matches[:, 1])
        if not_matched.shape[0] > 0:
            im = draw_keypoints(im, kpts2[not_matched] + np.array([w1, 0]), color=(255, 0, 0))
        im = draw_keypoints(im, kpts2[matches[:, 1]] + np.array([w1, 0]), color=(0, 255, 0))
        return im, kpts1, kpts2, matches, h1, h2, w1

    def _draw_texts(self, im, filename1, filename2, kpts1, kpts2, matches, h1, h2) -> Image.Image:
        font = ImageFont.truetype("WorkSans-VariableFont_wght.ttf", 12)
        draw = ImageDraw.Draw(im)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text(
                (20 + dx, h1 - 40 + dy),
                "{} - {} kpts".format(filename1, kpts1.shape[0]),
                "white",
                font=font,
            )
            draw.text(
                (20 + dx, h1 - 30 + dy),
                "{} - {} kpts".format(filename2, kpts2.shape[0]),
                "white",
                font=font,
            )
            draw.text(
                (20 + dx, h1 - 20 + dy), "{} matches".format(matches.shape[0]), "white", font=font
            )

        draw.text(
            (20, h1 - 40),
            "{} - {} kpts".format(filename1, kpts1.shape[0]),
            "black",
            font=font,
        )
        draw.text(
            (20, h1 - 30),
            "{} - {} kpts".format(filename2, kpts2.shape[0]),
            "black",
            font=font,
        )
        draw.text((20, h1 - 20), "{} matches".format(matches.shape[0]), "black", font=font)
        return im

    def visualize_keypoints(self) -> Image.Image:
        im, kpts1, kpts2, matches, h1, h2, _ = self._preproc()
        filename1, filename2 = self.image_path1.split("/")[-1], self.image_path2.split("/")[-1]
        im = self._draw_texts(im, filename1, filename2, kpts1, kpts2, matches, h1, h2)
        return im

    def visualize_matches(self) -> Image.Image:
        im, kpts1, kpts2, matches, h1, h2, w1 = self._preproc()
        filename1, filename2 = self.image_path1.split("/")[-1], self.image_path2.split("/")[-1]
        im = draw_lines(im, kpts1, kpts2, matches, w1)
        im = self._draw_texts(im, filename1, filename2, kpts1, kpts2, matches, h1, h2)
        return im
