# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple, Union

from blocks import EfficientUpdateFormer, CorrBlock, EfficientCorrBlock, bilinear_sampler


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C


class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=4,
        corr_levels=5,
        corr_radius=4,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        fine=False,
        cfg=None,
    ):
        super(BaseTrackerPredictor, self).__init__()
        """
        The base template to create a track predictor
        
        Modified from https://github.com/facebookresearch/co-tracker/
        """

        self.cfg = cfg

        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_size = hidden_size
        self.fine = fine

        self.flows_emb_dim = latent_dim // 2
        self.transformer_dim = (
            self.corr_levels * (self.corr_radius * 2 + 1) ** 2 + self.latent_dim * 2
        )

        self.efficient_corr = cfg.MODEL.TRACK.efficient_corr

        if self.fine:
            # TODO this is the old dummy code, will remove this when we train next model
            self.transformer_dim += 4 if self.transformer_dim % 2 == 0 else 5
        else:
            self.transformer_dim += (4 - self.transformer_dim % 4) % 4

        space_depth = depth if use_spaceatt else 0
        time_depth = depth

        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim,
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
        )

        self.norm = nn.GroupNorm(1, self.latent_dim)

        # A linear layer to update track feats at each iteration
        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

        if not self.fine:
            self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward(self, query_points, fmaps=None, iters=4, return_feat=False, down_ratio=1):
        """
        query_points: B x N x 2, the number of batches, tracks, and xy
        fmaps: B x S x C x HH x WW, the number of batches, frames, and feature dimension.
                note HH and WW is the size of feature maps instead of original images
        """
        B, N, D = query_points.shape
        B, S, C, HH, WW = fmaps.shape

        assert D == 2

        # Scale the input query_points because we may downsample the images
        # by down_ratio or self.stride
        # e.g., if a 3x1024x1024 image is processed to a 128x256x256 feature map
        # its query_points should be query_points/4
        if down_ratio > 1:
            query_points = query_points / float(down_ratio)
        query_points = query_points / float(self.stride)

        # Init with coords as the query points
        # It means the search will start from the position of query points at the reference frames
        coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)

        # Sample/extract the features of the query points in the query frame
        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])

        # init track feats by query feats
        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)  # B, S, N, C
        # back up the init coords
        coords_backup = coords.clone()

        # Construct the correlation block
        if self.efficient_corr:
            fcorr_fn = EfficientCorrBlock(
                fmaps, num_levels=self.corr_levels, radius=self.corr_radius
            )
        else:
            fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        coord_preds = []

        # Iterative Refinement
        for itr in range(iters):
            # Detach the gradients from the last iteration
            # (in my experience, not very important for performance)
            coords = coords.detach()

            # Compute the correlation (check the implementation of CorrBlock)
            if self.efficient_corr:
                fcorrs = fcorr_fn.sample(coords, track_feats)
            else:
                fcorr_fn.corr(track_feats)
                fcorrs = fcorr_fn.sample(coords)  # B, S, N, corrdim

            corrdim = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corrdim)

            # Movement of current coords relative to query points
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_emb = get_2d_embedding(flows, self.flows_emb_dim, cat_coords=False)

            # (In my trials, it is also okay to just add the flows_emb instead of concat)
            flows_emb = torch.cat([flows_emb, flows], dim=-1)

            track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            # Concatenate them as the input for the transformers
            transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)

            if transformer_input.shape[2] < self.transformer_dim:
                # pad the features to match the dimension
                pad_dim = self.transformer_dim - transformer_input.shape[2]
                pad = torch.zeros_like(flows_emb[..., 0:pad_dim])
                transformer_input = torch.cat([transformer_input, pad], dim=2)

            # 2D positional embed
            # TODO: this can be much simplified
            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(HH, WW)).to(
                query_points.device
            )
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])
            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)

            x = transformer_input + sampled_pos_emb

            # B, N, S, C
            x = rearrange(x, "(b n) s d -> b n s d", b=B)

            # Compute the delta coordinates and delta track features
            delta = self.updateformer(x)
            # BN, S, C
            delta = rearrange(delta, " b n s d -> (b n) s d", b=B)
            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)

            # Update the track features
            track_feats_ = self.ffeat_updater(self.norm(delta_feats_)) + track_feats_
            track_feats = track_feats_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # BxSxNxC

            # B x S x N x 2
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            # Force coord0 as query
            # because we assume the query points should not be changed
            coords[:, 0] = coords_backup[:, 0]

            # The predicted tracks are in the original image scale
            if down_ratio > 1:
                coord_preds.append(coords * self.stride * down_ratio)
            else:
                coord_preds.append(coords * self.stride)

        # B, S, N
        if not self.fine:
            vis_e = self.vis_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(
                B, S, N
            )
            vis_e = torch.sigmoid(vis_e)
        else:
            vis_e = None

        if return_feat:
            return coord_preds, vis_e, track_feats, query_track_feat
        else:
            return coord_preds, vis_e
