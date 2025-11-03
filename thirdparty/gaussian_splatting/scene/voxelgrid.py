#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_scaling_rotation, strip_symmetric
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p


class FixedGridGaussianModel:
    """
    Fixed-grid Gaussian model where positions lie on a regular voxel grid.
    Each voxel always hosts one Gaussian whose properties (color/opacity/shape)
    are optimised over time.
    """

    def __init__(self, sh_degree: int, rgbfuntion="rgbv1", grid_resolution: int = 64):
        self.grid_resolution = int(grid_resolution)
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._trbf_center = torch.empty(0)
        self._trbf_scale = torch.empty(0)
        self._motion = torch.empty(0)
        self._omega = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0.0

        self.grid_bounds_min = None
        self.grid_bounds_max = None
        self.voxel_size = None

        self.setup_functions()
        self.rgbdecoder = None  # same interface as other models
        self.trbfoutput = None
        self.ts = None
        self.preprocesspoints = 0

    # ------------------------------------------------------------------
    # Interface compatibility helpers
    # ------------------------------------------------------------------

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_trbfcenter(self):
        return self._trbf_center

    @property
    def get_trbfscale(self):
        return self._trbf_scale

    # ------------------------------------------------------------------
    # Grid initialisation
    # ------------------------------------------------------------------

    def create_fixed_grid(self, grid_min, grid_max):
        res = self.grid_resolution
        grid_min = torch.tensor(grid_min, dtype=torch.float32, device="cuda")
        grid_max = torch.tensor(grid_max, dtype=torch.float32, device="cuda")

        lin_x = torch.linspace(grid_min[0], grid_max[0], res, device="cuda")
        lin_y = torch.linspace(grid_min[1], grid_max[1], res, device="cuda")
        lin_z = torch.linspace(grid_min[2], grid_max[2], res, device="cuda")
        grid_x, grid_y, grid_z = torch.meshgrid(lin_x, lin_y, lin_z, indexing="ij")
        centers = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

        self.grid_bounds_min = grid_min
        self.grid_bounds_max = grid_max
        self.voxel_size = (grid_max - grid_min) / (res - 1)

        return centers

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda")
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device="cuda")
        times = torch.tensor(np.asarray(pcd.times), dtype=torch.float32, device="cuda")

        scene_min = torch.min(points, dim=0).values
        scene_max = torch.max(points, dim=0).values
        margin = 0.05 * (scene_max - scene_min)
        grid_min = (scene_min - margin).cpu().numpy()
        grid_max = (scene_max + margin).cpu().numpy()

        grid_centers = self.create_fixed_grid(grid_min, grid_max)
        num_voxels = grid_centers.shape[0]

        # Positions stay fixed -> keep as buffer (no grad, no optimizer)
        self._xyz = nn.Parameter(grid_centers, requires_grad=False)

        # For attribute initialisation pick nearest point cloud element
        grid_batch = min(8192, num_voxels)
        point_batch = min(32768, points.shape[0])
        nearest_idx = []
        num_points = points.shape[0]

        for start in range(0, num_voxels, grid_batch):
            end = min(start + grid_batch, num_voxels)
            grid_chunk = grid_centers[start:end]
            best_dist = torch.full((grid_chunk.shape[0],), float("inf"), device="cuda")
            best_idx = torch.zeros((grid_chunk.shape[0],), dtype=torch.long, device="cuda")

            for p_start in range(0, num_points, point_batch):
                p_end = min(p_start + point_batch, num_points)
                points_chunk = points[p_start:p_end]
                dist = torch.cdist(grid_chunk, points_chunk, p=2)
                chunk_min, chunk_idx = torch.min(dist, dim=1)
                better = chunk_min < best_dist
                if torch.any(better):
                    best_dist[better] = chunk_min[better]
                    best_idx[better] = chunk_idx[better] + p_start
                del dist

            nearest_idx.append(best_idx)
            del grid_chunk, best_dist, best_idx
            torch.cuda.empty_cache()

        nearest_idx = torch.cat(nearest_idx, dim=0)

        nearest_colors = colors[nearest_idx]
        initial_times = times[nearest_idx].unsqueeze(-1)

        self._features_dc = nn.Parameter(nearest_colors.contiguous().requires_grad_(True))

        # Scaling ~ voxel size
        voxel_size_vec = self.voxel_size.to(device="cuda")
        base_scale = voxel_size_vec.expand(num_voxels, 3).clone()
        base_scale = torch.clamp(base_scale * 0.5, min=5e-3)
        self._scaling = nn.Parameter(self.scaling_inverse_activation(base_scale).contiguous().requires_grad_(True))

        rots = torch.zeros((num_voxels, 4), device="cuda")
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # Start semi-transparent, allow learning to bistable
        opacities = torch.full((num_voxels, 1), 0.05, device="cuda")
        self._opacity = nn.Parameter(self.inverse_opacity_activation(opacities).contiguous().requires_grad_(True))

        self._trbf_center = nn.Parameter(initial_times.contiguous().requires_grad_(True))
        trbf_scale = torch.zeros((num_voxels, 1), device="cuda")
        self._trbf_scale = nn.Parameter(trbf_scale.requires_grad_(True))

        # Buffers for stats and gradient accumulation
        self.max_radii2D = torch.zeros((num_voxels,), device="cuda")
        self.xyz_gradient_accum = torch.zeros((num_voxels, 1), device="cuda")
        self.denom = torch.zeros((num_voxels, 1), device="cuda")

        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)

    # ------------------------------------------------------------------
    # Optimisation plumbing
    # ------------------------------------------------------------------

    def training_setup(self, training_args):
        params = [
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._trbf_center], "lr": training_args.trbfc_lr, "name": "trbf_center"},
            {"params": [self._trbf_scale], "lr": training_args.trbfs_lr, "name": "trbf_scale"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        # Only used for xyz schedule in original code; here we just keep API.
        return self.xyz_scheduler_args(iteration)

    def zero_gradient_cache(self):
        self._features_dc_grd.zero_()
        self._scaling_grd.zero_()
        self._rotation_grd.zero_()
        self._opacity_grd.zero_()
        self._trbf_center_grd.zero_()
        self._trbf_scale_grd.zero_()

    def cache_gradient(self):
        self._features_dc_grd += self._features_dc.grad.detach()
        self._scaling_grd += self._scaling.grad.detach()
        self._rotation_grd += self._rotation.grad.detach()
        self._opacity_grd += self._opacity.grad.detach()
        self._trbf_center_grd += self._trbf_center.grad.detach()
        self._trbf_scale_grd += self._trbf_scale.grad.detach()

    def set_batch_gradient(self, cnt):
        ratio = 1.0 / cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd * ratio

    # ------------------------------------------------------------------
    # Densification API compatibility (no-ops)
    # ------------------------------------------------------------------

    def add_densification_stats(self, *args, **kwargs):
        return

    def compute_covariance(self):
        return self.covariance_activation(self.get_scaling, 1.0, self._rotation)

    def _prune_optimizer(self, mask):
        if self.optimizer is None:
            return {}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            stored_state = self.optimizer.state.get(param, None)
            new_param_data = param[mask]
            new_param = nn.Parameter(new_param_data.requires_grad_(True))

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            optimizable_tensors[group["name"]] = new_param
        return optimizable_tensors

    def prune_points(self, mask):
        # Optional pruning stage at the end of training.
        keep_mask = ~mask
        optimizable = self._prune_optimizer(keep_mask)

        self._xyz = nn.Parameter(self._xyz[keep_mask], requires_grad=False)
        if optimizable:
            self._features_dc = optimizable["f_dc"]
            self._opacity = optimizable["opacity"]
            self._scaling = optimizable["scaling"]
            self._rotation = optimizable["rotation"]
            self._trbf_center = optimizable["trbf_center"]
            self._trbf_scale = optimizable["trbf_scale"]
        else:
            self._features_dc = nn.Parameter(self._features_dc[keep_mask].detach().requires_grad_(True))
            self._opacity = nn.Parameter(self._opacity[keep_mask].detach().requires_grad_(True))
            self._scaling = nn.Parameter(self._scaling[keep_mask].detach().requires_grad_(True))
            self._rotation = nn.Parameter(self._rotation[keep_mask].detach().requires_grad_(True))
            self._trbf_center = nn.Parameter(self._trbf_center[keep_mask].detach().requires_grad_(True))
            self._trbf_scale = nn.Parameter(self._trbf_scale[keep_mask].detach().requires_grad_(True))

        self.max_radii2D = self.max_radii2D[keep_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]

        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
            self._trbf_center,
            self._trbf_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            opt_dict,
            self.spatial_lr_scale,
            self._trbf_center,
            self._trbf_scale,
        ) = model_args
        self.training_setup(training_args)
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)

        device = self._xyz.device
        self._features_dc_grd = torch.zeros_like(self._features_dc, device=device, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, device=device, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, device=device, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, device=device, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, device=device, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, device=device, requires_grad=False)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z"]
        for i in range(self._features_dc.shape[1]):
            attrs.append(f"f_dc_{i}")
        for i in range(self._scaling.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attrs.append(f"rot_{i}")
        attrs.append("opacity")
        attrs.append("trbf_center")
        attrs.append("trbf_scale")
        return attrs

    def save_ply(self, path):
        mkdir_p(path.rsplit("/", 1)[0])
        xyz = self._xyz.detach().cpu().numpy()
        features = self._features_dc.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        trbf_center = self._trbf_center.detach().cpu().numpy().reshape(-1)
        trbf_scale = self._trbf_scale.detach().cpu().numpy().reshape(-1)

        dtype_full = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ]
        dtype_full += [(f"f_dc_{i}", "f4") for i in range(features.shape[1])]
        dtype_full += [(f"scale_{i}", "f4") for i in range(scales.shape[1])]
        dtype_full += [(f"rot_{i}", "f4") for i in range(rots.shape[1])]
        dtype_full += [("opacity", "f4"), ("trbf_center", "f4"), ("trbf_scale", "f4")]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements["x"] = xyz[:, 0]
        elements["y"] = xyz[:, 1]
        elements["z"] = xyz[:, 2]

        for i in range(features.shape[1]):
            elements[f"f_dc_{i}"] = features[:, i]
        for i in range(scales.shape[1]):
            elements[f"scale_{i}"] = scales[:, i]
        for i in range(rots.shape[1]):
            elements[f"rot_{i}"] = rots[:, i]
        elements["opacity"] = opacities[:, 0]
        elements["trbf_center"] = trbf_center
        elements["trbf_scale"] = trbf_scale

        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack([plydata["vertex"][axis] for axis in ("x", "y", "z")], axis=1)
        feature_keys = [name for name in plydata["vertex"].data.dtype.names if name.startswith("f_dc_")]
        feature_keys.sort(key=lambda n: int(n.split("_")[-1]))
        features = np.stack([plydata["vertex"][k] for k in feature_keys], axis=1)

        scale_keys = [f"scale_{i}" for i in range(3)]
        rot_keys = [f"rot_{i}" for i in range(4)]

        scales = np.stack([plydata["vertex"][k] for k in scale_keys], axis=1)
        rots = np.stack([plydata["vertex"][k] for k in rot_keys], axis=1)
        opacity = np.array(plydata["vertex"]["opacity"])[:, None]
        trbf_center = np.array(plydata["vertex"]["trbf_center"])[:, None]
        trbf_scale = np.array(plydata["vertex"]["trbf_scale"])[:, None]

        device = "cuda"
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device), requires_grad=False)
        self._features_dc = nn.Parameter(torch.tensor(features, dtype=torch.float32, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float32, device=device).requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float32, device=device).requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float32, device=device).requires_grad_(True))

        num = self._xyz.shape[0]
        self.max_radii2D = torch.zeros((num,), device=device)
        self.xyz_gradient_accum = torch.zeros((num, 1), device=device)
        self.denom = torch.zeros((num, 1), device=device)

        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
