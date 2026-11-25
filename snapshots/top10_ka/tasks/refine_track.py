# This is only a code segment
# You can easily use it in the codebase such as https://www.kaggle.com/code/nartaa/imc2024-starter/notebook.
import torch
import copy
from tqdm import tqdm

import numpy as np


def refine_track_imc(
    sorted_point3Ds,
    best_rec,
    image_stats,
    track_predictor,
    device,
    pradius=15,
    max_refine_num=4096,
    cfg=None,
):
    # Only do for one scene (B=1) and one track (N=1) once
    B = 1
    N = 1
    # only refine max_refine_num to avoid taking mnuch time
    sorted_point3Ds = sorted_point3Ds[:max_refine_num]
    for point3D in tqdm(sorted_point3Ds, desc=f"refining tracks by vggsfm"):
        point_track = point3D.track
        if point_track.length() >= 3:
            # ignore two -view tracks
            patches = []
            xy_fracs = []
            xy_floors = []
            imgids = []
            pointids = []
            reproj_errors = []
            for ele in point_track.elements:
                image_id = ele.image_id
                point2d_id = ele.point2D_idx
                img = best_rec.images[image_id]
                point2d_xy = img.points2D[point2d_id]
                image_name = img.name
                cam = best_rec.cameras[img.camera_id]

                # pick the rgb image content
                # change this to your command like imread
                rgb255 = image_stats[image_name]["rgb255"]

                # get the reprojected keypoints
                reproj_pts = cam.img_from_cam(img.cam_from_world * point3D.xyz)
                xy_floor = np.floor(point2d_xy.xy)
                xy_frac = point2d_xy.xy - xy_floor
                try:
                    # one element of a track cooresponds to one keypoint in one image
                    # we extract the pradiusxpradius image patch there
                    cur_patch = extract_patch_pad(rgb255, xy_floor.astype(int), pradius=pradius)
                except:
                    print("extract_patch_pad fails")
                    continue

                reproj_errors.append(np.linalg.norm(reproj_pts - point2d_xy.xy))
                xy_floors.append(xy_floor)
                xy_fracs.append(xy_frac)
                patches.append(numpy_image_to_torch(cur_patch)[None])
                imgids.append(image_id)
                pointids.append(point2d_id)

            if len(reproj_errors) > 3:
                if not cfg.random_q:
                    sortorder = sorted(range(len(reproj_errors)), key=lambda i: reproj_errors[i])
                patches_comb = torch.cat(patches).to(device)
                patches_comb = patches_comb[sortorder]
                xy_fracs = [xy_fracs[i] for i in sortorder]
                xy_floors = [xy_floors[i] for i in sortorder]
                imgids = [imgids[i] for i in sortorder]
                pointids = [pointids[i] for i in sortorder]
                reproj_errors = [reproj_errors[i] for i in sortorder]
                patch_feat = track_predictor.fine_fnet(patches_comb)
                S, C_out, psize, _ = patch_feat.shape
                patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
                patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")
                patch_query_points = (
                    torch.from_numpy(xy_fracs[0]).float()[None, None].to(device) + pradius
                )

                # feed patches into vggsfm fine track predictor
                fine_pred_track_lists, _, _, query_point_feat = track_predictor.fine_predictor(
                    query_points=patch_query_points, fmaps=patch_feat, iters=6, return_feat=True
                )
                fine_pred_track = fine_pred_track_lists[-1].cpu().numpy()
                fine_pred_track = fine_pred_track.squeeze()
                for pidx in range(len(imgids)):
                    saveimgid = imgids[pidx]
                    savepointid = pointids[pidx]
                    # please be careful that the center of a patch is not (0,0)
                    # the left top corner is (0,0)
                    pred_xy = xy_floors[pidx] - pradius + fine_pred_track[pidx]

                    best_rec.images[saveimgid].points2D[savepointid].xy = pred_xy
    return best_rec


def extract_patch_pad(image, xy, pradius):
    x, y = xy
    H, W, _ = image.shape

    # Calculate the coordinates of the top-left corner of the patch
    x1 = max(x - pradius, 0)
    y1 = max(y - pradius, 0)

    # Calculate the coordinates of the bottom-right corner of the patch
    x2 = min(x + pradius, W - 1)
    y2 = min(y + pradius, H - 1)

    # Extract the patch using numpy slicing
    patch = image[y1 : y2 + 1, x1 : x2 + 1, :]

    # Expected size of the patch
    expected_size = 2 * pradius + 1

    # Create an empty array of the expected size
    padded_patch = np.zeros((expected_size, expected_size, 3), dtype=image.dtype)

    # Calculate offsets for the centering the patch
    offset_y = (expected_size - (y2 - y1 + 1)) // 2
    offset_x = (expected_size - (x2 - x1 + 1)) // 2

    # Place the patch in the center of the padded_patch
    padded_patch[offset_y : offset_y + y2 - y1 + 1, offset_x : offset_x + x2 - x1 + 1] = patch

    return padded_patch


def extract_patch(image, xy, pradius):
    x, y = xy
    H, W, _ = image.shape

    # Calculate the coordinates of the top-left corner of the patch
    x1 = max(x - pradius, 0)
    y1 = max(y - pradius, 0)

    # Calculate the coordinates of the bottom-right corner of the patch
    x2 = min(x + pradius, W - 1)
    y2 = min(y + pradius, H - 1)

    # Extract the patch using numpy slicing
    patch = image[y1 : y2 + 1, x1 : x2 + 1, :]
    return patch


"""
register_ratio = register された画像の割合
"""
if len(maps) > 0:
    best_rec = maps[best_idx]
    register_extri = {}
    register_image_num = len(best_rec.images)
    register_ratio = register_image_num / (input_image_num)


if cfg.vsfm_refine:
    with torch.no_grad():
        track_predictor = vggsfm.track_predictor

        if cfg.ref_by_trackL:
            # sort the tracks by the length of tracks
            point3D_list = [
                (point3D.track.length(), point3D_id)
                for point3D_id, point3D in best_rec.points3D.items()
            ]
        else:
            # sort the tracks by mean reprojection error
            point3D_list = [
                (point3D.error, point3D_id) for point3D_id, point3D in best_rec.points3D.items()
            ]

        if len(point3D_list) > 0:
            tmp_rec = copy.deepcopy(best_rec)
            point3D_list.sort(reverse=True, key=lambda x: x[0])
            # tracks/3D points, sorted by track length or reprojection error
            sorted_point3Ds = [best_rec.points3D[id] for _, id in point3D_list]

            if cfg.vsfm_half:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    print("Half FOR VGGSFM")
                    best_rec = refine_track_imc(
                        sorted_point3Ds,
                        best_rec,
                        image_stats,
                        track_predictor,
                        device,
                        max_refine_num=cfg.vsfm_refine_maxn,
                        cfg=cfg,
                    )
            else:
                best_rec = refine_track_imc(
                    sorted_point3Ds,
                    best_rec,
                    image_stats,
                    track_predictor,
                    device,
                    max_refine_num=cfg.vsfm_refine_maxn,
                    cfg=cfg,
                )

            # set options for bundle adjustment
            options = pycolmap.IncrementalPipelineOptions()
            ba_options = pycolmap.BundleAdjustmentOptions()

            best_rec.filter_observations_with_negative_depth()
            pycolmap.bundle_adjustment(best_rec, ba_options)
            best_rec.filter_observations_with_negative_depth()
            # max reproject error: 4
            # min triangulation angle: 1.5
            best_rec.filter_all_points3D(
                4,
                1.5,
            )

            # we drop this refinement if it leads to some images unregisted
            filter_image_ids = best_rec.filter_images(
                options.min_focal_length_ratio,
                options.max_focal_length_ratio,
                options.max_extra_param,
            )
            if len(filter_image_ids) > 0:
                best_rec = tmp_rec
            else:
                best_rec = best_rec
            tmp_rec = None

        if cfg.ba_till_filter:
            # go further
            options = pycolmap.IncrementalPipelineOptions()
            ba_options = pycolmap.BundleAdjustmentOptions()
            for cur_max_rep_error in [4, 3.5, 3, 2.5, 2]:
                print(
                    "------------------------------------------filter by cur_max_rep_error ",
                    cur_max_rep_error,
                )
                cur_rec = copy.deepcopy(best_rec)
                cur_rec.filter_observations_with_negative_depth()
                # cur_rec.filter_all_points3D(cur_max_rep_error, 1.5, )
                pycolmap.bundle_adjustment(cur_rec, ba_options)
                cur_rec.filter_observations_with_negative_depth()
                cur_rec.filter_all_points3D(
                    cur_max_rep_error,
                    1.5,
                )
                filter_image_ids = cur_rec.filter_images(
                    options.min_focal_length_ratio,
                    options.max_focal_length_ratio,
                    options.max_extra_param,
                )
                if len(filter_image_ids) > 0:
                    break
                else:
                    best_rec = cur_rec
            cur_rec = None
