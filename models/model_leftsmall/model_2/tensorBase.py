# pylint: disable=E1111,E1102
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from nerf import NeRF
from utils import TVLoss, positional_encoding, raw2alpha, st


class AlphaGridMask(torch.nn.Module):
    """
    class for AlphaGridMask
    """

    def __init__(self, device, aabb, alpha_volume):
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device
        )

    def sample_alpha(self, xyz_sampled, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::sample_alpha")

        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::sample_alpha")

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    """
    class for MLP Render Feature
    """

    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, bias_enable=False, encode_app=False):
        super().__init__()
        self.encode_app = encode_app
        if self.encode_app:
            self.in_mlpC = 2 * max(viewpe, 0) * 3 + 2 * feape * inChanel + 3 * (viewpe > -1) + inChanel + 48
        else:
            self.in_mlpC = 2 * max(viewpe, 0) * 3 + 2 * feape * inChanel + 3 * (viewpe > -1) + inChanel

        self.viewpe = viewpe
        self.feape = feape

        layer1 = torch.nn.Linear(self.in_mlpC, featureC, bias=bias_enable)
        layer2 = torch.nn.Linear(featureC, featureC, bias=bias_enable)
        layer3 = torch.nn.Linear(featureC, 3, bias=bias_enable)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        # torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features, latent=None, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::PE+concate+mlp")
        if self.encode_app:
            indata = [features, latent]
        else:
            indata = [features]

        if self.viewpe > -1:
            indata += [viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::PE+concate+mlp")
        return rgb


class TensorBase(torch.nn.Module):
    """
    class for TensorBase
    """

    def __init__(  # pylint: disable=W0102
        self,
        aabb,
        gridSize,
        device,
        density_n_comp=8,
        appearance_n_comp=24,
        app_dim=27,
        shadingMode="MLP_PE",
        alphaMask=None,
        near_far=[2.0, 6.0],
        density_shift=-10,
        alphaMask_thres=0.001,
        distance_scale=25,
        rayMarch_weight_thres=0.001,
        pos_pe=6,
        view_pe=6,
        fea_pe=6,
        featureC=128,
        step_ratio=2.0,
        fea2denseAct="softplus",
        args=None,
    ):
        super().__init__()

        # new features
        if args.distributed:
            self.rank = args.rank
            self.world_size = args.world_size
            self.group = args.group

        self.run_nerf = args.run_nerf
        self.nonlinear_density = args.nonlinear_density
        self.ndims = args.ndims
        self.TV_weight_density = args.TV_weight_density
        self.TV_weight_app = args.TV_weight_app
        self.tvreg = TVLoss()

        self.density_n_comp = density_n_comp[: self.ndims]
        self.app_n_comp = appearance_n_comp[: self.ndims]
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]][: self.ndims]
        self.vecMode = [2, 1, 0][: self.ndims]
        self.comp_w = [1, 1, 1][: self.ndims]

        self.resMode = args.resMode if args.resMode is not None else [1]

        self.args = args
        self.n_level = args.n_level

        self.init_svd_volume(device)

        # feature renderer
        self.shadingMode = shadingMode
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.fea_pe = fea_pe
        self.featureC = featureC

        self.renderModule = MLPRender_Fea(
            self.app_dim, view_pe, fea_pe, featureC, args.bias_enable, args.encode_app
        ).to(device)

        self.run_nerf = args.run_nerf
        if self.run_nerf:
            self.init_nerf(args)

        self.n_importance = args.n_importance

    def init_nerf(self, args):
        self.nerf = NeRF(
            args,
            sum(self.density_n_comp) * len(self.resMode),
            sum(self.app_n_comp) * len(self.resMode),
        ).to(self.device)
        self.nerf_n_importance = args.nerf_n_importance
        self.run_nerf = True
        print("init run_nerf", self.nerf)

    def update_stepSize(self, gridSize):
        print("", flush=True)
        print(st.GREEN + "grid size" + st.RESET, gridSize, flush=True)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print(
            st.BLUE + f"sample step size: {self.stepSize:.3f} unit" + st.RESET,
            flush=True,
        )
        print(st.BLUE + f"default number: {self.nSamples}" + st.RESET, flush=True)

    def get_kwargs(self):
        return {
            "aabb": self.aabb,
            "gridSize": self.gridSize.tolist(),
            "density_n_comp": self.density_n_comp,
            "appearance_n_comp": self.app_n_comp,
            "app_dim": self.app_dim,
            "density_shift": self.density_shift,
            "alphaMask_thres": self.alphaMask_thres,
            "distance_scale": self.distance_scale,
            "rayMarch_weight_thres": self.rayMarch_weight_thres,
            "fea2denseAct": self.fea2denseAct,
            "near_far": self.near_far,
            "step_ratio": self.step_ratio,
            "shadingMode": self.shadingMode,
            "pos_pe": self.pos_pe,
            "view_pe": self.view_pe,
            "fea_pe": self.fea_pe,
            "featureC": self.featureC,
        }

    def init_svd_volume(self, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled, use_xyzb=False, return_feat=False, profiler=None):
        pass

    def compute_appfeature(self, xyz_sampled, use_xyzb=False, return_feat=False, profiler=None):
        pass

    def normalize_coord(self, xyz_sampled, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::normalize_coord")
        ret = (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1
        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::normalize_coord")
        return ret

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(ckpt["alphaMask.shape"])
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
            )
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::sample_ray")

        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        aabb = self.aabb.clone()
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::sample_ray")

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_within_hull(self, rays_o, rays_d, is_train=True, N_samples=-1, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::sample_ray_within_hull")

        near, far = self.near_far
        o_z = rays_o[:, -1:] - self.aabb[0, 2].item()
        d_z = rays_d[:, -1:]
        far = -(o_z / d_z)
        far[rays_d[:, 2] >= 0] = self.near_far[-1]
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(rays_o)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        if is_train:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(rays_o)
            z_vals = lower + (upper - lower) * t_rand

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        aabb = self.aabb
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)

        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::sample_ray_within_hull")

        return rays_pts, z_vals, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])

        for i in range(gridSize[0]):
            alpha_pred = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize)
            alpha[i] = alpha_pred.view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, _, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)

                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::feature2density")

        if self.fea2denseAct == "softplus":
            ret = F.softplus(density_features + self.density_shift)
            if profiler is not None:
                profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::feature2density")
            return ret
        elif self.fea2denseAct == "relu":
            ret = F.relu(density_features)
            if profiler is not None:
                profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::feature2density")
            return ret

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])

            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    def forward(
        self,
        rays_chunk,
        white_bg=True,
        is_train=False,
        N_samples=-1,
        profiler=None,
    ):
        rays_o = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6]

        if self.n_level > 2:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_within_hull(
                rays_o,
                viewdirs,
                is_train=is_train,
                N_samples=self.n_importance,
                profiler=profiler,
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid], profiler=profiler)
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~alpha_mask
                ray_valid = ~ray_invalid

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            if ray_valid.any():
                xyz_sampled = self.normalize_coord(xyz_sampled, profiler=profiler)
                sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], profiler=profiler)
                validsigma = self.feature2density(sigma_feature, profiler=profiler)
                sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale, profiler=profiler)

            app_mask = weight > self.rayMarch_weight_thres
            if app_mask.any():
                app_features = self.compute_appfeature(xyz_sampled[app_mask], profiler=profiler)
                fake_xyz_sampled_idxs = torch.zeros(xyz_sampled.shape[:-1])
                if self.args.encode_app:
                    app_latent = self.embedding_app(fake_xyz_sampled_idxs[app_mask])
                else:
                    app_latent = None
                valid_rgbs = self.renderModule(
                    viewdirs[app_mask],
                    app_features,
                    app_latent,
                    profiler=profiler,
                )
                rgb[app_mask] = valid_rgbs
            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb, -2)

            if profiler is not None:
                profiler.samples_collect(
                    xyz_sampled,
                    sigma[ray_valid].shape[0],
                    rgb[app_mask].shape[0],
                    ray_valid,
                )

            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1.0 - acc_map[..., None])
            rgb_map = rgb_map.clamp(0, 1)

            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)

            outputs = {"rgb_map": rgb_map, "depth_map": depth_map}

        else:
            # dense sample
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_o,
                viewdirs,
                is_train=is_train,
                N_samples=N_samples,
                profiler=profiler,
            )

            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid], profiler=profiler)
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~alpha_mask
                ray_valid = ~ray_invalid

            sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
            rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

            if ray_valid.any():
                xyz_sampled = self.normalize_coord(xyz_sampled, profiler=profiler)
                sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], profiler=profiler)

                validsigma = self.feature2density(sigma_feature, profiler=profiler)
                sigma[ray_valid] = validsigma

            _, weight, _ = raw2alpha(sigma, dists * self.distance_scale, profiler=profiler)

            app_mask = weight > self.rayMarch_weight_thres

            if app_mask.any():
                app_features = self.compute_appfeature(xyz_sampled[app_mask], profiler=profiler)

                valid_rgbs = self.renderModule(
                    viewdirs[app_mask],
                    app_features,
                    profiler=profiler,
                )
                rgb[app_mask] = valid_rgbs

            if profiler is not None:
                profiler.samples_collect(
                    xyz_sampled,
                    sigma[ray_valid].shape[0],
                    rgb[app_mask].shape[0],
                    ray_valid,
                )

            acc_map = torch.sum(weight, -1)
            rgb_map = torch.sum(weight[..., None] * rgb, -2)

            if white_bg or (is_train and torch.rand((1,)) < 0.5):
                rgb_map = rgb_map + (1.0 - acc_map[..., None])

            rgb_map = rgb_map.clamp(0, 1)
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)

            outputs = {"rgb_map": rgb_map, "depth_map": depth_map}

        return outputs
