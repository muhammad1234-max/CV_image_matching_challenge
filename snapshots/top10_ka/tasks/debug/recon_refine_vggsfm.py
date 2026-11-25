import torch

# import sys
# sys.path.append("/kaggle/working/vggsfm")

import pickle
import os
from pathlib import Path
from PIL import Image
import copy
from copy import deepcopy
import pycolmap
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from omegaconf import OmegaConf
from kornia.utils.grid import create_meshgrid
from kornia.geometry.subpix import dsnt
from lightglue.utils import numpy_image_to_torch

from base_track_predictor import BaseTrackerPredictor
from blocks import BasicEncoder, ShallowEncoder

import math

_EPS = np.finfo(float).eps * 4.0


def compute_score_fn(
    query_point_feat,
    patch_feat,
    fine_pred_track,
    sradius,
    psize,
    B,
    N,
    S,
    C_out,
):
    """
    Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps,
    given the query point features and reference frame feature maps
    """

    # query_point_feat initial shape: B x N x C_out,
    # query_point_feat indicates the feat at the coorponsing query points
    # Therefore we don't have S dimension here
    query_point_feat = query_point_feat.reshape(B, N, C_out)
    # reshape and expand to B x (S-1) x N x C_out
    query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
    # and reshape to (B*(S-1)*N) x C_out
    query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

    # Radius and size for computing the score
    ssize = sradius * 2 + 1

    # Reshape, you know it, so many reshaping operations
    patch_feat = rearrange(patch_feat, "(b n) s c p q -> b s n c p q", b=B, n=N)

    # Again, we unfold the patches to smaller patches
    # so that we can then focus on smaller patches
    # patch_feat_unfold shape:
    # B x S x N x C_out x (psize - 2*sradius) x (psize - 2*sradius) x ssize x ssize
    # well a bit scary, but actually not
    patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1)

    # Do the same stuffs above, i.e., the same as extracting patches
    fine_prediction_floor = fine_pred_track.floor().int()
    fine_level_floor_topleft = fine_prediction_floor - sradius

    # Clamp to ensure the smaller patch is valid
    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize)
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2)

    # Prepare the batch indices and xy locations

    batch_indices_score = torch.arange(B)[:, None, None].expand(-1, S, N)  # BxSxN
    batch_indices_score = batch_indices_score.reshape(-1).to(patch_feat_unfold.device)  # B*S*N
    y_indices = fine_level_floor_topleft[..., 0].flatten()  # Flatten H indices
    x_indices = fine_level_floor_topleft[..., 1].flatten()  # Flatten W indices

    reference_frame_feat = patch_feat_unfold.reshape(
        B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
    )

    # Note again, according to pytorch convention
    # x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
    reference_frame_feat = reference_frame_feat[batch_indices_score, :, x_indices, y_indices]
    reference_frame_feat = reference_frame_feat.reshape(B, S, N, C_out, ssize, ssize)
    # pick the frames other than the first one, so we have S-1 frames here
    reference_frame_feat = reference_frame_feat[:, 1:].reshape(
        B * (S - 1) * N, C_out, ssize * ssize
    )

    # Compute similarity
    sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)
    softmax_temp = 1.0 / C_out**0.5
    heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
    # 2D heatmaps
    heatmap = heatmap.reshape(B * (S - 1) * N, ssize, ssize)  # * x ssize x ssize

    coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
    grid_normalized = create_meshgrid(
        ssize, ssize, normalized_coordinates=True, device=heatmap.device
    ).reshape(1, -1, 2)

    var = (
        torch.sum(grid_normalized**2 * heatmap.view(-1, ssize * ssize, 1), dim=1)
        - coords_normalized**2
    )
    std = torch.sum(
        torch.sqrt(torch.clamp(var, min=1e-10)), -1
    )  # clamp needed for numerical stability

    score = std.reshape(B, S - 1, N)
    # set score as 1 for the query frame
    score = torch.cat([torch.ones_like(score[:, 0:1]), score], dim=1)
    return score


def refine_track(
    images,
    fine_fnet,
    fine_tracker,
    coarse_pred,
    compute_score=False,
    pradius=15,
    sradius=2,
    fine_iters=6,
    cfg=None,
):
    """
    Refines the tracking of images using a fine track predictor and a fine feature network.
    Check https://arxiv.org/abs/2312.04563 for more details.

    Args:
        images (torch.Tensor): The images to be tracked.
        fine_fnet (nn.Module): The fine feature network.
        fine_tracker (nn.Module): The fine track predictor.
        coarse_pred (torch.Tensor): The coarse predictions of tracks.
        compute_score (bool, optional): Whether to compute the score. Defaults to False.
        pradius (int, optional): The radius of a patch. Defaults to 15.
        sradius (int, optional): The search radius. Defaults to 2.
        cfg (dict, optional): The configuration dictionary. Defaults to None.

    Returns:
        torch.Tensor: The refined tracks.
        torch.Tensor, optional: The score.
    """
    # coarse_pred shape: BxSxNx2,
    # where B is the batch, S is the video/images length, and N is the number of tracks
    # now we are going to extract patches with the center at coarse_pred
    # Please note that the last dimension indicates x and y, and hence has a dim number of 2
    B, S, N, _ = coarse_pred.shape
    _, _, _, H, W = images.shape

    # Given the raidus of a patch, compute the patch size
    psize = pradius * 2 + 1

    # Note that we assume the first frame is the query frame
    # so the 2D locations of the first frame are the query points
    query_points = coarse_pred[:, 0]

    # Given 2D positions, we can use grid_sample to extract patches
    # but it takes too much memory.
    # Instead, we use the floored track xy to sample patches.

    # For example, if the query point xy is (128.16, 252.78),
    # and the patch size is (31, 31),
    # our goal is to extract the content of a rectangle
    # with left top: (113.16, 237.78)
    # and right bottom: (143.16, 267.78).
    # However, we record the floored left top: (113, 237)
    # and the offset (0.16, 0.78)
    # Then what we need is just unfolding the images like in CNN,
    # picking the content at [(113, 237), (143, 267)].
    # Such operations are highly optimized at pytorch
    # (well if you really want to use interpolation, check the function extract_glimpse() below)

    with torch.no_grad():
        content_to_extract = images.reshape(B * S, 3, H, W)
        C_in = content_to_extract.shape[1]

        # Please refer to https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # for the detailed explanation of unfold()
        # Here it runs sliding windows (psize x psize) to build patches
        # The shape changes from
        # (B*S)x C_in x H x W to (B*S)x C_in x H_new x W_new x Psize x Psize
        # where Psize is the size of patch
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    # Floor the coarse predictions to get integers and save the fractional/decimal
    track_int = coarse_pred.floor().int()
    track_frac = coarse_pred - track_int

    # Note the points represent the center of patches
    # now we get the location of the top left corner of patches
    # because the ouput of pytorch unfold are indexed by top left corner
    topleft = track_int - pradius
    topleft_BSN = topleft.clone()

    # clamp the values so that we will not go out of indexes
    # NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).
    # You need to seperately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize)

    # Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape(B * S, N, 2)

    # Prepare batches for indexing, shape: (B*S)xN
    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    # Extract image patches based on top left corners
    # extracted_patches: (B*S) x N x C_in x Psize x Psize
    extracted_patches = content_to_extract[batch_indices, :, topleft[..., 1], topleft[..., 0]]

    # Feed patches to fine fent for features
    patch_feat = fine_fnet(extracted_patches.reshape(B * S * N, C_in, psize, psize))

    C_out = patch_feat.shape[1]

    # Refine the coarse tracks by fine_tracker

    # reshape back to B x S x N x C_out x Psize x Psize
    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")

    # Prepare for the query points for fine tracker
    # They are relative to the patch left top corner,
    # instead of the image top left corner now
    # patch_query_points: N x 1 x 2
    # only 1 here because for each patch we only have 1 query point
    patch_query_points = track_frac[:, 0] + pradius
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1)

    # Feed the PATCH query points and tracks into fine tracker
    fine_pred_track_lists, _, _, query_point_feat = fine_tracker(
        query_points=patch_query_points,
        fmaps=patch_feat,
        iters=fine_iters,
        return_feat=True,
    )

    # relative the patch top left
    fine_pred_track = fine_pred_track_lists[-1].clone()

    # From (relative to the patch top left) to (relative to the image top left)
    for idx in range(len(fine_pred_track_lists)):
        fine_level = rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N)
        fine_level = fine_level.squeeze(-2)
        fine_level = fine_level + topleft_BSN
        fine_pred_track_lists[idx] = fine_level

    # relative to the image top left
    refined_tracks = fine_pred_track_lists[-1].clone()
    refined_tracks[:, 0] = query_points

    score = None

    if compute_score:
        score = compute_score_fn(
            query_point_feat,
            patch_feat,
            fine_pred_track,
            sradius,
            psize,
            B,
            N,
            S,
            C_out,
        )

    return refined_tracks, score


class TrackerPredictor(nn.Module):
    def __init__(
        self,
        cfg=None,
        **extra_args,
    ):
        super(TrackerPredictor, self).__init__()
        """
        COARSE and FINE are the dicts to construct the modules

        Both coarse_predictor and fine_predictor are constructed as a BaseTrackerPredictor,
        check track_modules/base_track_predictor.py
        
        Both coarse_fnet and fine_fnet are constructed as a 2D CNN network
        check track_modules/blocks.py for BasicEncoder and ShallowEncoder 
        """

        self.cfg = cfg

        # coarse predictor
        self.coarse_down_ratio = 2
        self.coarse_fnet = BasicEncoder(stride=4, cfg=cfg)
        # self.coarse_predictor = instantiate(COARSE.PREDICTOR, _recursive_=False, stride=COARSE.stride, cfg=cfg)
        self.coarse_predictor = BaseTrackerPredictor(stride=4, cfg=cfg)

        # fine predictor, forced to use stride = 1
        # self.fine_fnet = instantiate(FINE.FEATURENET, _recursive_=False, stride=1, cfg=cfg)
        self.fine_fnet = ShallowEncoder(stride=1, cfg=cfg)
        # self.fine_predictor = instantiate(FINE.PREDICTOR, _recursive_=False, stride=1, cfg=cfg)
        self.fine_predictor = BaseTrackerPredictor(
            depth=4,
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,
            hidden_size=256,
            fine=True,
            use_spaceatt=False,
            stride=1,
            cfg=cfg,
        )

    def forward(
        self,
        images,
        query_points,
        fmaps=None,
        coarse_iters=6,
        inference=True,
        fine_tracking=True,
    ):
        """
        Args:
            images (torch.Tensor): Images as RGB, in the range of [0, 1], with a shape of B x S x 3 x H x W.
            query_points (torch.Tensor): 2D xy of query points, relative to top left, with a shape of B x N x 2.
            fmaps (torch.Tensor, optional): Precomputed feature maps. Defaults to None.
            coarse_iters (int, optional): Number of iterations for coarse prediction. Defaults to 6.
            inference (bool, optional): Whether to perform inference. Defaults to True.
            fine_tracking (bool, optional): Whether to perform fine tracking. Defaults to True.

        Returns:
            tuple: A tuple containing fine_pred_track, coarse_pred_track, pred_vis, and pred_score.
        """

        if fmaps is None:
            fmaps = self.process_images_to_fmaps(images)

        if inference:
            torch.cuda.empty_cache()

        # Coarse prediction
        coarse_pred_track_lists, pred_vis = self.coarse_predictor(
            query_points=query_points,
            fmaps=fmaps,
            iters=coarse_iters,
            down_ratio=self.coarse_down_ratio,
        )
        coarse_pred_track = coarse_pred_track_lists[-1]

        if inference:
            torch.cuda.empty_cache()

        if fine_tracking:
            # Refine the coarse prediction
            fine_pred_track, pred_score = refine_track(
                images,
                self.fine_fnet,
                self.fine_predictor,
                coarse_pred_track,
                compute_score=True,
                cfg=self.cfg,
            )

            if inference:
                torch.cuda.empty_cache()
        else:
            fine_pred_track = coarse_pred_track
            pred_score = torch.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score

    def process_images_to_fmaps(self, images):
        """
        This function processes images for inference.

        Args:
            images (np.array): The images to be processed.

        Returns:
            np.array: The processed images.
        """
        batch_num, frame_num, image_dim, height, width = images.shape
        assert batch_num == 1, "now we only support processing one scene during inference"
        reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
        if self.coarse_down_ratio > 1:
            # whether or not scale down the input images to save memory
            fmaps = self.coarse_fnet(
                F.interpolate(
                    reshaped_image,
                    scale_factor=1 / self.coarse_down_ratio,
                    mode="bilinear",
                    align_corners=True,
                )
            )
        else:
            fmaps = self.coarse_fnet(reshaped_image)
        fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

        return fmaps


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


class Config:
    vsfm_refine: bool = True
    ba_till_filter: bool = True
    ref_by_trackL: bool = True
    vsfm_half: bool = False

    vsfm_refine_maxn: int = 4096 * 2
    random_q: bool = False


import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMC 2025 Scene Refinement")
    parser.add_argument("--dataset", type=str, default="imc2024_lizard_pond", help="Dataset name")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
    )
    parser.add_argument("--model_name", type=str, default="v35", help="Model name")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/kaggle/input/image-matching-challenge-2025",
        help="Base path for data",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/kaggle/input/image-matching-challenge-2025/train/imc2024_lizard_pond",
        help="Directory containing images",
    )
    return parser.parse_args()


def main_dataset():
    args = parse_args()
    dataset = args.dataset
    model_name = args.model_name

    for p in Path(f"/kaggle/working/output/{model_name}/{dataset}/").glob(
        "cluster*/cluster_data_dict.pkl"
    ):
        print(">>", str(p))
        cluster_name = p.parent.name
        print(">>", cluster_name)
        main_scene(cluster_name)


def main_scene(scene) -> None:
    args = parse_args()
    dataset = args.dataset
    model_name = args.model_name
    base_path = args.base_path
    images_dir = args.images_dir

    device = torch.device(f"cuda:{args.device}")
    cluster_work_dir = Path(f"/kaggle/working/output/{model_name}/{dataset}/{scene}/")
    recon_work_dir = Path(
        f"/kaggle/working/output/{model_name}/{dataset}/{scene}/colmap_rec_aliked/"
    )
    final_colmap_out = Path(f"/kaggle/working/output/{model_name}/{dataset}/{scene}/colmap_final/")
    final_colmap_out.mkdir(parents=True, exist_ok=True)

    with open((cluster_work_dir / "cluster_data_dict.pkl"), "rb") as f:
        cluster_data_dict = pickle.load(f)

    if False:
        # data_dict, dataset, scene, images_dir, base_path
        generate_submission_csv_per_cluster(
            cluster_work_dir,
            final_colmap_out,
            # recon_work_dir,
            cluster_data_dict,
            dataset,
            scene,
            images_dir,
            base_path,
        )
        return

    checkpoint = torch.load("/kaggle/working/vggsfm_v2_0_0.bin", map_location=torch.device("cpu"))
    tracker_weight = {
        k[len("track_predictor.") :]: v
        for k, v in checkpoint.items()
        if k.startswith("track_predictor.")
    }
    tracker_cfg = OmegaConf.load("/kaggle/working/vggsfm_conf.yml")
    track_predictor = TrackerPredictor(cfg=tracker_cfg)
    track_predictor.load_state_dict(tracker_weight, strict=True)
    track_predictor.eval()
    track_predictor = track_predictor.to(device)

    cfg = Config()

    image_stats = {}
    for p in Path(images_dir).glob("*.png"):
        image_stats[p.name] = {}
        image_stats[p.name]["rgb255"] = np.array(Image.open(p))

    # maps = {}
    # best_img_num = 0
    # best_idx = 0
    for recon_path in recon_work_dir.glob("*"):
        map_idx = int(recon_path.name)
        recon = pycolmap.Reconstruction(str(recon_path))
        print(recon.summary())
        best_rec = recon
        # maps[map_idx] = recon
        # if best_img_num < len(maps[map_idx].images):
        #     best_img_num = len(maps[map_idx].images)
        #     best_idx = map_idx

        # if len(maps) > 0:
        #     best_rec = maps[best_idx]
        # else:
        #     raise ValueError

        if cfg.vsfm_refine:
            with torch.no_grad():

                if cfg.ref_by_trackL:
                    # sort the tracks by the length of tracks
                    point3D_list = [
                        (point3D.track.length(), point3D_id)
                        for point3D_id, point3D in best_rec.points3D.items()
                    ]
                else:
                    # sort the tracks by mean reprojection error
                    point3D_list = [
                        (point3D.error, point3D_id)
                        for point3D_id, point3D in best_rec.points3D.items()
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

                    # NOTE: best_rec.filter_... の attribution がないので ObservationManager を経由
                    # best_rec.filter_observations_with_negative_depth()
                    _observation_manager = pycolmap.ObservationManager(best_rec)
                    _observation_manager.filter_observations_with_negative_depth()

                    pycolmap.bundle_adjustment(best_rec, ba_options)

                    # NOTE: best_rec.filter_... の attribution がないので ObservationManager を経由
                    # best_rec.filter_observations_with_negative_depth()
                    _observation_manager = pycolmap.ObservationManager(best_rec)
                    _observation_manager.filter_observations_with_negative_depth()

                    # max reproject error: 4
                    # min triangulation angle: 1.5

                    # NOTE: best_rec.filter_... の attribute がないので ObservationManager を経由
                    # best_rec.filter_all_points3D(
                    #     4,
                    #     1.5,
                    # )
                    _observation_manager = pycolmap.ObservationManager(best_rec)
                    _observation_manager.filter_all_points3D(4, 1.5)
                    filter_image_ids = _observation_manager.filter_images(
                        options.min_focal_length_ratio,
                        options.max_focal_length_ratio,
                        options.max_extra_param,
                    )

                    # we drop this refinement if it leads to some images unregisted
                    # filter_image_ids = best_rec.filter_images(
                    #     options.min_focal_length_ratio,
                    #     options.max_focal_length_ratio,
                    #     options.max_extra_param,
                    # )

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

                        # NOTE: cur_rec.filter... は attribute がないので ObservationManager 経由
                        # cur_rec.filter_observations_with_negative_depth()
                        _observation_manager = pycolmap.ObservationManager(cur_rec)
                        _observation_manager.filter_observations_with_negative_depth()

                        # NOTE: 以下は最初からコメントアウトされていた
                        # cur_rec.filter_all_points3D(cur_max_rep_error, 1.5, )
                        pycolmap.bundle_adjustment(cur_rec, ba_options)

                        # NOTE: patch
                        _observation_manager = pycolmap.ObservationManager(cur_rec)
                        # cur_rec.filter_observations_with_negative_depth()
                        # cur_rec.filter_all_points3D(
                        #     cur_max_rep_error,
                        #     1.5,
                        # )
                        # filter_image_ids = cur_rec.filter_images(
                        #     options.min_focal_length_ratio,
                        #     options.max_focal_length_ratio,
                        #     options.max_extra_param,
                        # )
                        _observation_manager.filter_observations_with_negative_depth()
                        _observation_manager.filter_all_points3D(cur_max_rep_error, 1.5)
                        filter_image_ids = _observation_manager.filter_images(
                            options.min_focal_length_ratio,
                            options.max_focal_length_ratio,
                            options.max_extra_param,
                        )

                        if len(filter_image_ids) > 0:
                            break
                        else:
                            best_rec = cur_rec
                    cur_rec = None

        recon_dir = final_colmap_out / str(map_idx)
        recon_dir.mkdir(parents=True, exist_ok=True)
        best_rec.write(recon_dir)

    # data_dict, dataset, scene, images_dir, base_path
    generate_submission_csv_per_cluster(
        cluster_work_dir,
        final_colmap_out,
        cluster_data_dict,
        dataset,
        scene,
        images_dir,
        base_path,
    )


def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])


def generate_submission_csv_per_cluster(
    cluster_work_dir,
    final_colmap_out,
    data_dict,
    dataset,
    scene,
    images_dir,
    base_path,
) -> None:
    submission_path = cluster_work_dir / "submission.csv"

    maps = {}
    for recon_path in final_colmap_out.glob("*"):
        map_idx = int(recon_path.name)
        recon = pycolmap.Reconstruction(str(recon_path))
        maps[map_idx] = recon

    # 2. Look for the best reconstruction: The incremental mapping offered by
    # pycolmap attempts to reconstruct multiple models, we must pick the best one
    images_registered = 0
    best_idx = None

    print("Looking for the best reconstruction")

    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            try:
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx1
            except Exception:
                continue

    # Parse the reconstruction object to get the rotation matrix and translation vector
    # obtained for each image in the reconstruction
    results = {}
    camid_im_map = {}
    if best_idx is not None:
        for k, im in maps[best_idx].images.items():
            key = os.path.join(images_dir, im.name)
            results[key] = {}
            results[key]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            results[key]["t"] = deepcopy(np.array(im.cam_from_world.translation))

            camid_im_map[im.camera_id] = im.name

    print(results)

    # TODO: 以下がどのような役割を果たしているかを理解する
    # best map からの R, t は上のコードで完了しているはず。以下はなぜ必要？：
    # -> 別の component を連結するために用いている
    try:
        if best_idx is not None:
            for idx1, rec in maps.items():
                u_cameras = []
                g_cameras = []
                if idx1 != best_idx:
                    # best ではないものから取り出し

                    for k, im in rec.images.items():
                        key = os.path.join(images_dir, im.name)
                        if key in results:
                            g_R = deepcopy(results[key]["R"])
                            g_t = deepcopy(results[key]["t"])
                            g_C = -g_R.T @ g_t

                            u_R = deepcopy(im.cam_from_world.rotation.matrix())
                            u_t = deepcopy(np.array(im.cam_from_world.translation))
                            u_C = -u_R.T @ u_t
                            g_cameras.append(g_C.reshape(3, 1))
                            u_cameras.append(u_C.reshape(3, 1))
                    if len(g_cameras) < 3:
                        continue

                    # NOTE: best にはない位置を取得して追加してるのかも
                    g_cameras = np.array(g_cameras).reshape(3, -1)
                    u_cameras = np.array(u_cameras).reshape(3, -1)
                    inl_cf = 0
                    strict_cf = -1
                    thresholds = np.array(
                        [0.002, 0.005, 0.008, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
                    )
                    # thresholds = np.array(
                    #     [0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
                    # )
                    model = register_by_Horn(
                        u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf
                    )
                    T = np.squeeze(model["transf_matrix"][-1])
                    # print(T)
                    # print(T[:3].shape)
                    for k, im in rec.images.items():
                        key = os.path.join(images_dir, im.name)
                        if key not in results:
                            Tcw2 = np.eye(4)
                            Tcw2[:3, :3] = deepcopy(im.cam_from_world.rotation.matrix())
                            Tcw2[:3, 3] = deepcopy(np.array(im.cam_from_world.translation))
                            Tw2c = np.linalg.inv(Tcw2)
                            Tw1c = np.matmul(T, Tw2c)
                            Tcw1 = np.linalg.inv(Tw1c)
                            results[key]["R"] = deepcopy(Tcw1[:3, :3])
                            results[key]["t"] = deepcopy(Tcw1[:3, 3])
    except:
        pass
    # with open(os.path.join(work_dir, "camid_im_map.json"), "w") as f:
    #     json.dump(camid_im_map, f, indent=2)

    print(f"Registered: {dataset} / {scene} -> {len(results)} images")
    print(f"Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    # Create Submission
    submission = {
        "image_path": [],
        "dataset": [],
        "scene": [],
        "rotation_matrix": [],
        "translation_vector": [],
    }
    for image in data_dict[dataset][scene]:
        image_path = str(image.relative_to(base_path))
        submission["image_path"].append(image_path)
        submission["dataset"].append(dataset)

        # NOTE: outliers or scene
        submission["scene"].append(scene)

        if str(image) in results:
            R = results[str(image)]["R"].reshape(-1)
            T = results[str(image)]["t"].reshape(-1)
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


def refine_track_imc(
    sorted_point3Ds,
    best_rec,
    image_stats,
    track_predictor,
    device,
    pradius=15,
    max_refine_num=4096,
    cfg: Config | None = None,
):
    assert cfg is not None

    # Only do for one scene (B=1) and one track (N=1) once
    B = 1
    N = 1
    # only refine max_refine_num to avoid taking mnuch time
    # print(">DEBUG", len(sorted_point3Ds), max_refine_num)
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


if __name__ == "__main__":
    main_dataset()
