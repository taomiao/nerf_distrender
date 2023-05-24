from unittest import TestCase

import torch

from model_manager.model_manager import ModelManager
import numpy as np
from render_utils.pose_to_rays import get_ray_directions_blender, get_rays_with_directions
from render_utils.ray_to_samples import sample_ray, sample_ray_within_hull, opt_sample_ray_within_hull, opt_sample_ray
import os
import imageio

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/PJLAB/taomiao/PycharmProjects/distrender/tests/inductor_codecache"


class TestLeftSmallMultiStages(TestCase):

    def test_infer(self):
        import torch._dynamo
        import torch._inductor
        import torch._dynamo.config
        import torch._inductor.config
        import logging
        torch._inductor.config.debug = True
        torch._dynamo.config.verbose = True
        torch._dynamo.config.suppress_errors = True
        mm = ModelManager()
        mm.load_models()
        tensorf = mm.get_model_by_name("model_leftsmall_multi_stages")

        c2w = torch.tensor([
            [
                -0.9999769644340181,
                0.006441397055757059,
                -0.0021398610458922534,
                -1532.05048640069,
                10652.0
            ],
            [
                -0.006444035841396001,
                -0.9999784822896628,
                0.0012285600258823703,
                472.14420678033,
                14204.0
            ],
            [
                -0.0021319013580488387,
                0.0012423210665824184,
                0.9999969558128491,
                297.105299808587,
                11165.851060556606
            ]
        ]
        )
        h, w, f = c2w[:, 4] / 10
        h = int(h)
        w = int(w)
        f = [f, f]
        c2w = c2w[:, :4]
        c2w[:, 3] /= 100

        directions = get_ray_directions_blender(h, w, f)
        rays_o, rays_d = get_rays_with_directions(directions, c2w)
        rays_o = rays_o.reshape([h * w, 3])
        rays_d = rays_d.reshape([h * w, 3])
        xyz_sampled, z_vals, ray_valid = sample_ray(rays_o, rays_d, 128, 0.0026, 0.4, 7,
                               torch.tensor([[-18., -6., 0.], [-6., 6., 1.]]))
        rays_d = rays_d.view(-1, 1, 3).expand(xyz_sampled.shape)
        rgb_depth = tensorf(xyz_sampled, rays_d, z_vals, ray_valid)
        rgb, depth = rgb_depth.to_here()
        img = rgb.clamp(0, 1.0).reshape([h, w, 3])*255
        img = img.to(torch.uint8).numpy()
        imageio.imwrite("./test.png", img)
