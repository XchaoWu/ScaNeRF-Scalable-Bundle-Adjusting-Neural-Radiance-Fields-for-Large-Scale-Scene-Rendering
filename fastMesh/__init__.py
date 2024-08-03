import torch 
from .lib.fastMesh import fastMesh
from tqdm import tqdm 
import sys 
sys.path += ['./', '../']
from cuda import ray_aabb_intersection
from cuda import background_sampling_cuda

class FastMesh:
    def __init__(self, path):
        self.fmesh = fastMesh()
        self.fmesh.build(path)

    def set(self, bbox_center, bbox_size):
        self.bbox_center = bbox_center 
        self.bbox_size = bbox_size
    
    def get_sceneinfo(self):
        return self.fmesh.getSceneBound()

    @torch.no_grad()
    def render_depth(self, rays_o, rays_d):
        depth = torch.zeros_like(rays_o[..., :1])
        self.fmesh.fisrtHit(rays_o, rays_d, depth)
        return depth 
    
    @torch.no_grad()
    def render_mask(self, rays_o, rays_d, trust_mesh=False):
        depth = torch.zeros_like(rays_o[..., :1])
        if trust_mesh==False:
            self.fmesh.firstEnter(rays_o, rays_d, depth)
        else:
            self.fmesh.fisrtHit(rays_o, rays_d, depth)
        # print(depth.min(), depth.max())
        # self.fmesh.fisrtHit(rays_o, rays_d, depth)
        bounds = torch.ones_like(rays_o[...,:2]) * -1
        ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
        inside = torch.all(torch.abs(rays_o - self.bbox_center) < (self.bbox_size / 2.), dim=-1, keepdim=True)

        # [FIXME]
        # return ((depth > bounds[..., :1]) & (bounds[..., :1] != -1) & (depth > 0 )) | inside

        # for outdoor scene 
        # return (depth > bounds[..., :1]) | (bounds[...,:1] == -1) | (depth == 0) | inside
        return ( (depth > bounds[..., :1]) & (bounds[..., :1] != -1) ) | (depth == 0) | inside
        
    @torch.no_grad()
    def sample_points(self, rays_o, rays_d, start, num_sample):
        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=rays_o.device)
        self.fmesh.sample_points(rays_o, rays_d, start, z_vals)
        return z_vals
    
    @torch.no_grad()
    def compute_bgdepth_batch(self, rays_o, rays_d):

        depth_z = torch.zeros_like(rays_o[...,:1])
        self.fmesh.fisrtHit(rays_o, rays_d, depth_z)

        bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
        ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
        valid = (bounds[:, 1] != -1) 
        # valid = torch.all(bounds!=-1, dim=-1)

        rays_o[valid] = rays_o[valid] + bounds[valid,1:] * rays_d[valid]

        bg_z = torch.zeros_like(rays_o[...,:1])

        self.fmesh.fisrtHit(rays_o, rays_d, bg_z)
        bg_z[depth_z == 0] = 1000 # for special case (no bg)

        has_bg = (bg_z[...,0] > 0) & valid 

        bg_z[valid] = bg_z[valid] + bounds[valid, 1:]

        return bg_z, has_bg, bounds
    
    @torch.no_grad()
    def background_sampling(self, rays_o, rays_d, num_sample, sample_range):

        bg_z, valid, bounds = self.compute_bgdepth_batch(rays_o.clone(), rays_d)

        z_vals = torch.full((rays_o.shape[0],num_sample), -1, dtype=torch.float32, device=rays_o.device)

        background_sampling_cuda(rays_o, rays_d, bounds[:,1:], bg_z, z_vals, num_sample, sample_range)

        return z_vals, valid

    # @torch.no_grad()
    # def background_sampling_v2(self, rays_o, rays_d, num_sample):
    #     bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
    #     ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
    #     z_vals = self.sample_points(rays_o, rays_d, bounds[:,1], num_sample)
    #     valid = torch.all(z_vals!=-1, dim=-1) & (bounds[:, 1] != -1) 
    #     return z_vals, valid 

    @torch.no_grad()
    def compute_bgdepth(self, poses, H, W):

        num_camera  = poses.ks.shape[0]
        device = poses.device

        bg_depths = torch.zeros(num_camera, H,W, dtype=torch.float32, device=device)
        # empty_rays = torch.zeros(num_camera, H,W, dtype=torch.bool, device=device)

        bounds = torch.full((H*W,2), -1, dtype=torch.float32, device=device)
        bg_z = torch.zeros((H*W,1), dtype=torch.float32, device=device)
        # fg_z = torch.zeros((H*W,1), dtype=torch.float32, device=device)

        all_rays_o, all_rays_d = poses.getRays(H,W)
        for idx in tqdm(range(num_camera)):
            rays_o = all_rays_o[idx]
            rays_d = all_rays_d[idx]

            # self.fmesh.fisrtHit(rays_o, rays_d, fg_z)

            ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
            valid = (bounds[:, 1] != -1) 

            rays_o[valid] = rays_o[valid] + bounds[valid,1:] * rays_d[valid]

            self.fmesh.fisrtHit(rays_o, rays_d, bg_z)
            has_no_bg = bg_z <= 0 

            # empty_rays[idx] = ((~has_no_bg) & (torch.abs(fg_z - bg_z) < 0.01)).reshape(H,W).clone()

            bg_z[valid] = bg_z[valid] + bounds[valid, 1:]
            bg_z[has_no_bg] = 0 
            bg_depths[idx] = bg_z.reshape(H,W).clone()
        
        del bounds, bg_z, all_rays_o, all_rays_d
        torch.cuda.empty_cache()

        return bg_depths

