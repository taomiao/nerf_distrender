import torch
import torch.nn.functional as F
from tensorBase import TensorBase


class TensorVMSplit_multi(TensorBase):
    """
    TensoRF VM multi-resolution plane
    """

    def __init__(self, aabb, gridSize, device, **kargs):
        super().__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp) * len(self.resMode), self.app_dim, bias=False).to(device)
        if self.nonlinear_density:
            self.basis_den = torch.nn.Linear(sum(self.density_n_comp) * len(self.resMode), 1, bias=False).to(device)
        if self.args.encode_app:
            self.embedding_app = torch.nn.Embedding(11500, 48).to(device)
        print("density_plane:", self.density_plane)
        print("density_line:", self.density_line)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        if self.args.ckpt is None:  # set gridsize de novo
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                for j in self.resMode:
                    plane_coef.append(
                        torch.nn.Parameter(
                            scale
                            * torch.randn(
                                (
                                    1,
                                    n_component[i],
                                    gridSize[mat_id_1] * j,
                                    gridSize[mat_id_0] * j,
                                )
                            )
                        )
                    )  #
                    line_coef.append(
                        torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id] * j, 1)))
                    )
        else:
            # set gridsize according to ckpt, since the scale relationship between different planes is broken
            # when using merged ckpts trained by plane parallel
            ckpt = torch.load(self.args.ckpt, map_location=self.args.device)
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                for j in range(len(self.resMode)):
                    planeSize = ckpt["state_dict"][f"density_plane.{j}"].shape[
                        -2:
                    ]  # just need to use density plane to get both density/app plane size
                    plane_coef.append(
                        torch.nn.Parameter(scale * torch.randn((1, n_component[i], planeSize[0], planeSize[1])))
                    )
                    line_coef.append(
                        torch.nn.Parameter(
                            scale
                            * torch.randn(
                                (
                                    1,
                                    n_component[i],
                                    gridSize[vec_id] * self.resMode[j],
                                    1,
                                )
                            )
                        )
                    )
            del ckpt
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    # obtimization
    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatial},
            {"params": self.density_plane, "lr": lr_init_spatial},
            {"params": self.app_line, "lr": lr_init_spatial},
            {"params": self.app_plane, "lr": lr_init_spatial},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if self.args.encode_app:
            grad_vars += [{"params": self.embedding_app.parameters(), "lr": lr_init_network}]
        if self.nonlinear_density:
            grad_vars += [{"params": self.basis_den.parameters(), "lr": lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{"params": self.renderModule.parameters(), "lr": lr_init_network}]
        if self.run_nerf:
            grad_vars += [{"params": self.nerf.parameters(), "lr": 5e-4}]
        return grad_vars

    # obtain feature grids
    def compute_densityfeature(self, xyz_sampled, return_feat=False, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::compute_densityfeature")

        N = self.ndims
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
        coordinate_line = (
            torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
        )
        # either direct summation over interpolation / mlp mapping
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = [], []
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for idx_plane in range(len(self.density_plane)):
            idx_dim = idx_plane // len(self.resMode)
            plane_coef = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_dim]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef = F.grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_dim]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            if self.nonlinear_density or return_feat:
                plane_coef_point.append(plane_coef)
                line_coef_point.append(line_coef)
            sigma_feature = sigma_feature + torch.sum(plane_coef * line_coef, dim=0)
        if self.nonlinear_density or return_feat:
            plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        if self.nonlinear_density:
            sigma_feature = F.relu(self.basis_den((plane_coef_point * line_coef_point).T))[  # pylint: disable=E1102
                ..., 0
            ]

        if return_feat:
            ret = (plane_coef_point * line_coef_point).T
            if profiler is not None:
                profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::compute_densityfeature")
            return sigma_feature, ret

        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::compute_densityfeature")

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, return_feat=False, profiler=None):
        if profiler is not None:
            profiler.op_time_collect_beg("render_test::evaluation::render_image::render_fn::compute_appfeature")

        N = self.ndims
        coordinate_plane = torch.stack([xyz_sampled[..., self.matMode[i]] for i in range(N)]).detach().view(N, -1, 1, 2)
        coordinate_line = torch.stack([xyz_sampled[..., self.vecMode[i]] for i in range(N)])
        coordinate_line = (
            torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(N, -1, 1, 2)
        )
        plane_coef_point, line_coef_point = [], []

        # for idx_plane in range(len(self.app_plane)):
        for idx_plane in range(len(self.app_plane)):
            idx_dim = idx_plane // len(self.resMode)
            plane_coef_point.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    coordinate_plane[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:-1])
            )
            line_coef_point.append(
                F.grid_sample(
                    self.app_line[idx_plane],
                    coordinate_line[[idx_dim]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        if return_feat:
            ret1 = self.basis_mat((plane_coef_point * line_coef_point).T)  # pylint: disable=E1102
            ret2 = (plane_coef_point * line_coef_point).T
            if profiler is not None:
                profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::compute_appfeature")
            return ret1, ret2

        ret = self.basis_mat((plane_coef_point * line_coef_point).T)  # pylint: disable=E1102

        if profiler is not None:
            profiler.op_time_collect_end("render_test::evaluation::render_image::render_fn::compute_appfeature")

        return ret

    # upsample/update grids
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            for j in self.resMode:
                plane_idx = i * len(self.resMode) + self.resMode.index(j)
                plane_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[plane_idx].data,
                        size=(res_target[mat_id_1] * j, res_target[mat_id_0] * j),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                line_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[plane_idx].data,
                        size=(res_target[vec_id] * j, 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)
        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
        print(self.density_plane)

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units  # pylint: disable=E0203
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :])
            self.app_line[i] = torch.nn.Parameter(self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :])
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]  # pylint: disable=E0203
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]  # pylint: disable=E0203
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def vectorDiffs(self, vector_comps):
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))
            )  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2  # + reg(self.app_line[idx]) * 1e-3
        return total
