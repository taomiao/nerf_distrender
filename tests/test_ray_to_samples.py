from unittest import TestCase

import torch

from model_manager.model_manager import ModelManager
import numpy as np
from render_utils.pose_to_rays import get_ray_directions_blender, get_rays_with_directions
from render_utils.ray_to_samples import sample_ray, sample_ray_within_hull, opt_sample_ray, opt_sample_ray_within_hull

import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "distrender/tests/inductor_codecache"


class TestRayToSamples(TestCase):

    def test_infer(self):
        import torch._dynamo
        import torch._inductor
        import torch._dynamo.config
        import torch._inductor.config
        import logging
        torch._inductor.config.debug = True
        torch._dynamo.config.verbose = True
        torch._dynamo.config.suppress_errors = True

        c2w = torch.tensor([[9.72847700e-01, -2.10714117e-01, 9.57446322e-02, -8.93235672e-03],
                            [2.28060439e-01, 8.02241445e-01, -5.51722050e-01, -1.61061749e-01],
                            [3.94452736e-02, 5.58577001e-01, 8.28514159e-01, 8.68037283e-01]]
                           )
        h, w, f = 768, 1024, [859.23, 859.23]

        directions = get_ray_directions_blender(h, w, f)
        rays_o, rays_d = get_rays_with_directions(directions, c2w)
        rays_o = rays_o.reshape([h * w, 3])
        rays_d = rays_d.reshape([h * w, 3])
        rays = torch.cat([rays_o, rays_d], dim=1)[:100].cuda()
        xyz_sampled, z_vals, ray_valid = opt_sample_ray_within_hull(rays_o, rays_d, 128, 0.0026, 0.4, 7,
                                                    torch.tensor([[-18., -6., 0.], [-6., 6., 1.]]))
        print(xyz_sampled.shape)
