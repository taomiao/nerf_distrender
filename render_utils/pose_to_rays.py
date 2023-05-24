import torch
import numpy as np
from kornia import create_meshgrid
# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_ray_directions(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]#+0.5
    i,j = grid.unbind(-1)
    cent = center if center is not None else [W/2, H/2]
    directions = torch.stack([(i-cent[0])/focal[0], (j-cent[1])/focal[1], torch.ones_like(i)],-1)  # (H, W, 3)
    directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions

def get_ray_directions_blender(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]#+0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i-cent[0])/focal[0], -(j-cent[1])/focal[1], -torch.ones_like(i)],-1)  # (H, W, 3)
    directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions

def get_rays_with_directions(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

### others
def filtering_rays(aabb, all_rays, chunk=10240*5):
    N = torch.tensor(all_rays.shape[:-1]).prod()
    mask_filtered = []
    idx_chunks = torch.split(torch.arange(N), chunk)
    all_pts = []
    for idx_chunk in idx_chunks:
        rays_chunk = all_rays[idx_chunk]
        rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (aabb[1] - rays_o) / vec
        rate_b = (aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1) # least amount of steps needed to get inside bbox.
        t_max = torch.maximum(rate_a, rate_b).amin(-1) # least amount of steps needed to get outside bbox.
        mask_inbbox = t_max > t_min
        mask_filtered.append(mask_inbbox.cpu())
        d_z=rays_d[:, -1:]
        o_z=rays_o[:, -1:]
        far=-(o_z/d_z)
        pts = rays_o + rays_d*far
        all_pts.append(pts)
    all_pts = torch.cat(all_pts)
    mask_filtered = torch.cat(mask_filtered)
    ratio = torch.sum(mask_filtered) / N
    return mask_filtered, ratio, all_pts
