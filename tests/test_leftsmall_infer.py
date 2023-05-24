import os
from unittest import TestCase

import imageio

from model_manager.model_manager import ModelManager
from render_utils.pose_to_rays import get_ray_directions_blender, get_rays_with_directions

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/PJLAB/taomiao/PycharmProjects/distrender/tests/inductor_codecache"


class TestLeftSmall(TestCase):

    def test_infer(self):
        import torch._dynamo
        import torch._inductor
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.debug = True
        torch._dynamo.config.verbose = True
        torch._dynamo.config.suppress_errors = True
        mm = ModelManager()
        mm.load_models()
        tensorf = mm.get_model_by_name("model_leftsmall")

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
        rays = torch.cat([rays_o, rays_d], dim=1)[:100000]
        rgb, depth = tensorf(rays)
        img = rgb.clamp(0, 1.0).reshape([h, w, 3]) * 255
        img = img.to(torch.uint8).numpy()
        imageio.imwrite("./test.png", img)
