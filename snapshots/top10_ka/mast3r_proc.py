"""
TODO:
* device
*

"""

import sys

from cv2 import norm

sys.path.append("/kaggle/input/smly-mast3r-pkgs/mast3r")
sys.path.append("/kaggle/input/smly-mast3r-pkgs/mast3r/dust3r")

import os
from copy import deepcopy
import pickle
import argparse
import tempfile
import numpy as np
import pandas as pd

from dust3r.utils.image import load_images  # type: ignore
from mast3r.image_pairs import make_pairs  # type: ignore
from mast3r.retrieval.processor import Retriever  # type: ignore
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # type: ignore
from mast3r.model import AsymmetricMASt3R  # type: ignore

import torch

import math


_EPS = np.finfo(float).eps * 4.0


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        # print("special case")
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None


def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, -1) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean traffansformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified)."""

    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        # print (vector_norm(q), np.linalg.norm(q))
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    """Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.

    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.
    """

    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))
    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.Inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.Inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n - 2):
        for jj in range(ii + 1, n - 1):
            for kk in range(jj + 1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(
                    ev_coord[:, i], gt_coord[:, i], usesvd=False
                )
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(
                        ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]]
                    )
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze(
                        (no_inl_ref > max_no_inl)
                        | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)),
                        axis=0,
                    )
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (
                            best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update]
                        )
                        best_transf_matrix[to_update] = transf_matrix

    for i in range(r):
        print(
            f"Registered cameras {int(max_no_inl[0, i])}/{n} for threshold {ransac_threshold[0, i]}"
        )

    best_model = {
        "valid_cams": idx_cams,
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix,
    }
    return best_model


def pose_c2w_to_R_t(pose_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a camera-to-world pose matrix (4x4) to a
    world-to-camera rotation matrix (3x3) and translation vector (3x1).

    Args:
        pose_c2w: A 4x4 NumPy array representing the camera-to-world
                  transformation matrix [R_c2w | t_c2w; 0 | 1].
                  R_c2w transforms points from camera to world coordinates.
                  t_c2w is the camera center position in world coordinates.

    Returns:
        A tuple containing:
        - R_w2c: A 3x3 NumPy array representing the world-to-camera
                 rotation matrix. Transforms points from world to camera coordinates.
        - t_w2c: A 3x1 NumPy array representing the world origin's position
                 in camera coordinates. This is the translation vector often
                 used in the projection matrix P = K[R|t].
    """
    # --- Input validation ---
    if not isinstance(pose_c2w, np.ndarray):
        raise TypeError("pose_c2w must be a NumPy array.")
    if pose_c2w.shape != (4, 4):
        raise ValueError("pose_c2w must be a 4x4 matrix.")
    # Optional: Check if the bottom row is [0, 0, 0, 1]
    # if not np.allclose(pose_c2w[3, :], [0, 0, 0, 1]):
    #     print("Warning: Bottom row of pose_c2w is not [0, 0, 0, 1].")
    # Optional: Check if the rotation part is a valid rotation matrix
    # R_c2w_check = pose_c2w[:3, :3]
    # if not np.allclose(np.linalg.det(R_c2w_check), 1.0):
    #      print(f"Warning: Determinant of the rotation part is not close to 1 ({np.linalg.det(R_c2w_check):.4f}).")
    # if not np.allclose(R_c2w_check.T @ R_c2w_check, np.identity(3)):
    #      print("Warning: Rotation part is not orthogonal (R^T * R != I).")

    # --- Extract R_c2w and t_c2w ---
    # R_c2w: Rotation matrix from camera coordinates to world coordinates (top-left 3x3)
    R_c2w = pose_c2w[:3, :3]
    # t_c2w: Camera center position in world coordinates (top-right 3x1)
    # Ensure t_c2w is a column vector (3x1)
    t_c2w = pose_c2w[:3, 3].reshape(3, 1)

    # --- Calculate R_w2c and t_w2c ---
    # R_w2c: Rotation matrix from world coordinates to camera coordinates
    # R_w2c = R_c2w^T (inverse of rotation matrix is its transpose)
    R_w2c = R_c2w.T

    # t_w2c: World origin position in camera coordinates
    # t_w2c = -R_w2c * t_c2w
    t_w2c = -R_w2c @ t_c2w  # Use '@' for matrix multiplication

    # Ensure t_w2c is also a 3x1 column vector
    t_w2c = t_w2c.reshape(3, 1)

    return R_w2c, t_w2c


def run_sfm(filelist: list[str], device_id: int) -> dict[str, dict[str, str]]:
    optim_level = "refine+depth"
    retrieval_model = "/kaggle/input/imc-models/pytorch/bundle/1/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
    model_weight_path = "/kaggle/input/imc-models/pytorch/bundle/1/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    device = "cuda:{device_id:d}".format(device_id=device_id)
    image_size = 512
    # image_size = 1024
    # image_size = 768
    # image_size = 960
    lr1 = 0.07
    niter1 = 300
    lr2 = 0.01
    niter2 = 300
    shared_intrinsics = True
    matching_conf_thr = 0.0

    # scenegraph_type = "complete"
    scenegraph_type = "retrieval"

    image_size_ = (image_size, image_size)
    # model = AsymmetricMASt3R.from_pretrained(model_weight_path, img_size=image_size_).to(device)
    model = AsymmetricMASt3R.from_pretrained(model_weight_path).to(device)

    # Ignore FutureWarning
    imgs = load_images(filelist, size=image_size, verbose=False)
    assert len(imgs) > 1, "Not enough images loaded"
    n_images = len(imgs)

    if scenegraph_type == "retrieval":
        winsize = min(20, n_images)  # Na
        refid = min(10, n_images - 1)  # k
        scene_graph_params = [scenegraph_type, str(winsize), str(refid)]
    elif scenegraph_type == "complete":
        scene_graph_params = [scenegraph_type]
    else:
        raise ValueError("Invalid scenegraph_type")

    scene_graph = "-".join(scene_graph_params)

    sim_matrix = None
    if "retrieval" in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(
        imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix
    )

    with tempfile.TemporaryDirectory() as outdir:
        # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=outdir)
        os.makedirs(cache_dir, exist_ok=True)
        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            model,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            device=device,
            opt_depth="depth" in optim_level,
            shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr,
        )

    im_poses = scene.get_im_poses().cpu().numpy().astype(np.float64)

    # Parse the reconstruction object to get the rotation matrix and translation vector
    # obtained for each image in the reconstruction
    results = {}
    # camid_im_map = {}
    for idx in range(len(filelist)):
        key = str(filelist[idx])
        pose_c2w = im_poses[idx]
        rotation_matrix, translation_vector = pose_c2w_to_R_t(pose_c2w)
        print(">>DEBUG(R,t)", rotation_matrix, translation_vector)

        results[key] = {}
        results[key]["R"] = deepcopy(rotation_matrix)
        results[key]["t"] = deepcopy(translation_vector)

    # u_cameras = []
    # g_cameras = []
    # for idx in range(len(filelist)):
    #     key = str(filelist[idx])
    #     if key in results:
    #         g_R = deepcopy(results[key]["R"])
    #         g_t = deepcopy(results[key]["t"])
    #         g_C = -g_R.T @ g_t

    #         rotation_matrix = im_poses[idx][:3, :3]
    #         translation_vector = im_poses[idx][:3, 3]
    #         u_R = deepcopy(rotation_matrix)
    #         u_t = deepcopy(translation_vector)
    #         u_C = -u_R.T @ u_t
    #         g_cameras.append(g_C.reshape(3, 1))
    #         u_cameras.append(u_C.reshape(3, 1))

    # g_cameras = np.array(g_cameras).reshape(3, -1)
    # u_cameras = np.array(u_cameras).reshape(3, -1)
    # inl_cf = 0
    # strict_cf = -1
    # thresholds = np.array([0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
    # model = register_by_Horn(u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf)
    # T = np.squeeze(model["transf_matrix"][-1])

    # for idx in range(len(filelist)):
    #     key = str(filelist[idx])
    #     if key not in results:
    #         rotation_matrix = im_poses[idx][:3, :3]
    #         translation_vector = im_poses[idx][:3, 3]

    #         Tcw2 = np.eye(4)
    #         Tcw2[:3, :3] = deepcopy(rotation_matrix)
    #         Tcw2[:3, 3] = deepcopy(translation_vector)
    #         Tw2c = np.linalg.inv(Tcw2)
    #         Tw1c = np.matmul(T, Tw2c)
    #         Tcw1 = np.linalg.inv(Tw1c)
    #         results[key]["R"] = deepcopy(Tcw1[:3, :3])
    #         results[key]["t"] = deepcopy(Tcw1[:3, 3])

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASt3R-SfM")
    parser.add_argument("--dataset", type=str, default="imc", help="dataset name (default: imc)")
    parser.add_argument(
        "--scene_name", type=str, default="scene", help="scene name (default: scene)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/kaggle/input/image-matching-challenge-2025/test",
        help="base path for the dataset",
    )
    parser.add_argument("--submission_path", type=str, help="path to submission file")
    parser.add_argument(
        "--cluster_data_dict_path",
        type=str,
        help="path to cluster data dict file",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="device id (default: 0)",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset
    scene_name = args.scene_name
    submission_path = args.submission_path
    cluster_data_dict_path = args.cluster_data_dict_path
    base_path = args.base_path

    # Load the data dict from the specified path
    with open(cluster_data_dict_path, "rb") as f:
        cluster_data_dict = pickle.load(f)

    filelist = [str(p) for p in cluster_data_dict[dataset][scene_name]]
    results = run_sfm(filelist, args.device_id)

    # Create Submission
    submission = {
        "image_path": [],
        "dataset": [],
        "scene": [],
        "rotation_matrix": [],
        "translation_vector": [],
    }
    for image in cluster_data_dict[dataset][scene_name]:
        image_path = str(image.relative_to(base_path))
        submission["image_path"].append(image_path)
        submission["dataset"].append(dataset)
        submission["scene"].append(scene_name)

        if str(image) in results:
            R = results[str(image)]["R"].reshape(-1)  # type: ignore
            T = results[str(image)]["t"].reshape(-1)  # type: ignore
            # submission["scene"].append(scene)
        else:
            R = np.eye(3).reshape(-1)
            T = np.zeros((3))
            # submission["scene"].append("outliers")

        submission["rotation_matrix"].append(arr_to_str(R))
        submission["translation_vector"].append(arr_to_str(T))

    submission_df = pd.DataFrame.from_dict(submission)
    print(f"Save to {submission_path}")
    submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main(parse_args())
