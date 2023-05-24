import torch


def sample_ray(rays_o, rays_d, N_samples, step_size, near, far, aabb):
    stepsize = step_size
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (aabb[1] - rays_o) / vec
    rate_b = (aabb[0] - rays_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
    rng = torch.arange(N_samples)[None].float()
    step = stepsize * rng.to(rays_o.device)
    interpx = t_min[..., None] + step
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
    mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
    return rays_pts, interpx, ~mask_outbbox


def sample_ray_within_hull(rays_o, rays_d, N_samples, step_size, near, far, aabb):
    o_z = rays_o[:, -1:] - aabb[0, 2].item()
    d_z = rays_d[:, -1:]
    far = -(o_z / d_z)
    far[rays_d[:, 2] >= 0] = far[-1]
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(rays_o)
    z_vals = near * (1.0 - t_vals) + far * (t_vals)
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
    return rays_pts, z_vals, ~mask_outbbox


opt_sample_ray = torch.compile(sample_ray)
opt_sample_ray_within_hull = torch.compile(sample_ray_within_hull)
