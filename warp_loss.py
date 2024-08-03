import torch 
import torch.nn as nn 
import numpy as np 
import camera 
import torch.nn.functional as F
from torchvision import transforms
import sys 
from time import time 
from tools import tools 
from tools import utils
from cuda import computeViewcost
from cuda import grid_sample_forward_cuda, grid_sample_backward_cuda
from cuda import (
    gaussian_grid_sample_forward_cuda, 
    gaussian_grid_sample_backward_cuda,
    grid_sample_bool_cuda,
    proj2neighbor_forward,
    proj2neighbor_backward)

class SampleColorAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, grid):
        """
        src  N x H x W x 3
        grid N x B x 1 x 2
        """
        num_images = grid.shape[0]
        batch_size = grid.shape[1]
        device = grid.device
        out = torch.full((num_images, batch_size, 1, 3), 0, dtype=torch.float, device=device)
        mask = torch.full((num_images, batch_size, 1, 1), 0, dtype=torch.bool, device=device)
        grid_sample_forward_cuda(src, grid, out, mask)

        ctx.num_images = num_images
        ctx.batch_size = batch_size
        ctx.device = device
        ctx.src = src 
        ctx.grid = grid 

        return out, mask 
    
    @staticmethod
    def backward(ctx, grad_out, grad_mask):
        
        grad_grid = torch.full((ctx.num_images, ctx.batch_size, 1, 2), 0, dtype=torch.float, device=ctx.device)

        grid_sample_backward_cuda(ctx.src, ctx.grid, grad_out, grad_grid)

        return None, grad_grid

def sample_color(src, grid):
    return SampleColorAutoGrad.apply(src, grid)

class GaussianSampleColorAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, grid, sigma, max_dis):
        """
        src  N x H x W x 3
        grid N x B x 1 x 2
        """
        num_images = grid.shape[0]
        batch_size = grid.shape[1]
        device = grid.device
        out = torch.full((num_images, batch_size, 1, 3), 0, dtype=torch.float, device=device)
        mask = torch.full((num_images, batch_size, 1, 1), 0, dtype=torch.bool, device=device)
        gaussian_grid_sample_forward_cuda(src, grid, out, mask, sigma, max_dis)

        ctx.num_images = num_images
        ctx.batch_size = batch_size
        ctx.device = device
        ctx.src = src 
        ctx.grid = grid 
        ctx.sigma = sigma
        ctx.max_dis = max_dis

        return out, mask 

    
    @staticmethod
    def backward(ctx, grad_out, grad_mask):
        
        grad_grid = torch.full((ctx.num_images, ctx.batch_size, 1, 2), 0, dtype=torch.float, device=ctx.device)

        gaussian_grid_sample_backward_cuda(ctx.src, ctx.grid, grad_out, grad_grid, ctx.sigma, ctx.max_dis)

        return None, grad_grid, None, None 


def gaussian_sample_color(src, grid, sigma, max_dis):
    """
    sigma  Gaussian sigma 
    max_dis  高斯的影响范围，像素单位

    开始优化的情况下 sigma 大一些， max_dis 一些 
    sigma = 5.0   max_dis = 10.0 
    to 
    sigma = 1.0   max_dis = 0 
    """
    return GaussianSampleColorAutoGrad.apply(src, grid, sigma, max_dis)


class ProjNeighborViewAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, ks, rts, nei_views, nei_valid):
        """
        pts B x 3
        ks  N x 3 x 3
        rts N x 3 x 4 
        nei_views B x K 
        nei_valid B x K 
        """
        batch_size = pts.shape[0]
        num_neighbors = nei_views.shape[1]
        device = pts.device 
        
        grid =  torch.full((batch_size, num_neighbors, 3), 0, dtype=torch.float32, device=device)
        nei_origin = torch.full((batch_size, num_neighbors, 3), 0, dtype=torch.float32, device=device)
        nei_direction = torch.full((batch_size, num_neighbors, 3), 0, dtype=torch.float32, device=device)
        # proj_depth = torch.full((batch_size, num_neighbors, 1), -1, dtype=torch.float32, device=device)

        proj2neighbor_forward(pts, ks, rts, nei_views, nei_valid, nei_origin, nei_direction, grid)
        
        ctx.pts = pts 
        ctx.ks = ks 
        ctx.rts = rts 
        ctx.nei_views = nei_views
        ctx.nei_valid = nei_valid
        ctx.batch_size = batch_size
        ctx.device = device 
        ctx.num_camera = ks.shape[0]

        return grid, nei_origin, nei_direction 
    
    @staticmethod
    def backward(ctx, grad_grid, grad_nei_origin, grad_nei_direction):
        

        grad_pts = torch.full( (ctx.batch_size, 3), 0, dtype=torch.float32, device=ctx.device)
        grad_rts = torch.full( (ctx.num_camera, 3, 4), 0, dtype=torch.float32, device=ctx.device)

        proj2neighbor_backward(ctx.pts, ctx.ks, ctx.rts, ctx.nei_views, ctx.nei_valid, grad_grid, grad_pts, grad_rts)
        return grad_pts, None, grad_rts, None, None 

def proj_neighbor_view(pts, ks, rts, nei_views, nei_valid):
    return ProjNeighborViewAutoGrad.apply(pts, ks, rts, nei_views, nei_valid)

class WarpLoss:
    def __init__(self, cfg, block, z_thresh=0, num_sample=64):
        self.cfg = cfg 
        self.voxel_size = float(torch.max(block.tile_size / block.resolution[-1]).cpu())
        self.z_thresh = z_thresh
        self.H = block.H 
        self.W = block.W
        self.featureGrid = block.featureGrid
        self.decoder = block.decoder 
        self.poses = block.poses
        self.block = block 

        self.num_sample = 128
        self.alpha = cfg.TRAINING.LOSS.ALPHA  
        self.gamma = cfg.TRAINING.LOSS.GAMMA 

        self.device = self.featureGrid.device

        # self.images = block.train_data.images.permute(0,3,1,2)
        # self.images = (block.train_data.images * 255).type(torch.uint8).to(self.device)
        self.images = block.train_data.images

        self.lam = 0.5
        self.total_step = cfg.TRAINING.TOTAL_STEP

        self.topK = 10
        self.num_camera = self.poses.ks.shape[0]

        # self.sigma = 5.0 
        # self.max_dis = 10.0 

        # self.sigma_list = cfg.TRAINING.LOSS.SIGMA
        # self.radius_list = cfg.TRAINING.LOSS.RADIUS
        # self.adjust_step = cfg.TRAINING.LOSS.STEPS


        # self.occlusions = block.train_data.occlusions


        # self.confidence_weight = torch.ones((self.poses.ks.shape[0],), dtype=torch.float32, device=self.device)

        # self.normals = ((block.train_data.mono_normals + 1) / 2 * 255).type(torch.uint8).to(self.device)

        # self.max_level = 1

        # self.max_level = 1
        # self.current_level = self.max_level-1
        # self.build_pyramid()

        # torch.cuda.empty_cache()

        # self.pyramid_images = build_pyramid(input=self.images, max_level=self.max_level, align_corners=True)
        # self.blured_images = transforms.GaussianBlur(kernel_size=21, sigma=10)(self.images)

    def get_sigma(self, step):
        # return 0.1 + 4.9 * np.exp(-step/3000)
        return 3.0

    def get_radius(self, step):
        # return 0.5 + 7.5 * np.exp(-step/3000)
        return 5.0

    def toCPU(self):
        self.images = self.images.cpu()

    def toGPU(self):
        self.images = self.images.to(self.device)


    def build_pyramid(self):
        self.pyramid_images = [self.images.to(self.device)]
        for _ in range(self.max_level-1):
            self.pyramid_images += [transforms.GaussianBlur(kernel_size=21, sigma=10)(self.pyramid_images[-1])[..., ::2,::2]]
        print(f"Use pyramid images for warp loss level={len(self.pyramid_images)}\n")

    def soft_vis(self, depth_diff):
        return torch.exp(-self.alpha * depth_diff / self.voxel_size)
    
    def soft_diffuse(self, specular):
        """
        reflections = tint * specular   B x 3 
        """
        return torch.exp(-self.gamma *  torch.mean(specular, dim=-1, keepdim=True) )
        # return 1.0 / (1.0 + torch.exp(self.gamma * (torch.mean(specular, dim=-1, keepdim=True) - 0.5)))

    def proj_points2grid(self, x, rts):
        """
        x B x 3
        P N x 3 x 4 
        H height  W width 
        """

        # project to camera space  
        # N x 3 x 4
        # rts = self.poses.get_rts()
        # N x B x 3 
        x_cam = camera.world2cam(x, rts)
        # mark invalid points 
        invalid = torch.where(x_cam[..., 2] <= self.z_thresh)

        # compute ray direction here 
        # N x B x 3
        rays_d = x_cam.clone()
        rays_d /= (rays_d[..., 2:3] + 1e-8)
        rays_d = rays_d @ rts[:,:3,:3]

        # record sample far here N x B x 1 
        far = x_cam[..., 2].clone()

        # proj to pixel space 
        px = x_cam @ self.poses.ks.transpose(-2,-1)

        # N x B x 2
        grid = px[..., :2] / (px[..., 2:3] + 1e-8)
        grid[...,0] /= (self.W-1)
        grid[...,1] /= (self.H-1)
        grid = grid * 2. - 1.
        grid[invalid] = -2.

        # N x B x 1 x 2 
        # return grid[..., None, :], rays_d.detach(), far.detach()
        return grid[..., None, :], rays_d.detach(), far.detach()


    def fetch_color(self, grid, src):
        """
        grid N x B x 1 x 2 
        src N x 3 x H x W
        """
        mask = torch.ones_like(src[:,:1, ...])
        # N x H x W x 4
        src = torch.cat([src, mask], dim=1)
        # N x 4 x B x 1 
        out = F.grid_sample(src.detach(), grid, 
                            padding_mode="zeros",
                            mode="bilinear", align_corners=True)
        # N x B x 4
        out = out.squeeze(-1).permute(0,2,1)
        # N x B x 3
        color = out[..., :-1]
        # N x B x 1 
        mask = out[..., -1:]
        return color.contiguous(), mask.contiguous()

    def color_variance(self, color, score):
        """
        color  N x B x 3 
        score  N x B x 1
        """

        # B x 3 
        weighted_sum = torch.sum(color * score, dim=0)
        weights = torch.sum(score, dim=0)
        
        weighted_mean = weighted_sum / (weights + 1e-8)

        # B x 3 
        weighted_squre = torch.sum((color - weighted_mean[None, ...]) ** 2 * score, dim=0)
        mean_squre = weighted_squre / (weights + 1e-8)

        return torch.mean(mean_squre)
    
    def color_weight(self, color):
        """
        color N x B x 3 
        """
        # B x 3 
        mean_color = torch.mean(color, dim=0)

        # N x B x 1 
        return 1. - torch.mean((color - mean_color[None,...]) ** 2, dim=-1, keepdim=True)
    
    def color_weight2(self, color, score):
        """
        color N x B x 3 
        score N x B x 1 
        """
        # B x 3 
        weighted_sum = torch.sum(color * score, dim=0)
        weights = torch.sum(score, dim=0)
        weighted_mean = weighted_sum / (weights + 1e-8)

        # N x B x 1 
        return 1. - torch.mean((color - weighted_mean[None,...]) ** 2, dim=-1, keepdim=True)

    def compute_loss(self, c1, c2, score):
        """
        c     N x B x 3
        score N x B x 1 
        """
        return (torch.mean((c1-c2)**2, dim=-1, keepdim=True) * score).mean()

    @torch.no_grad()
    def select_neighbor_views(self, proj_mask, view_score, nei_occlusion):
        num_camera = self.poses.ks.shape[0]
        valid = (proj_mask==1) & nei_occlusion & (view_score >= 0.824)
        valid = valid.reshape(num_camera, -1)
        view_score = view_score.reshape(num_camera, -1)
        view_score[~valid] = 0
        # 对于每条光线，选择最好的前10个nighbor 
        val, idxs = torch.topk(view_score, self.topK, dim=0)
        mask = torch.zeros_like(view_score).scatter_(dim=0, index=idxs, src=torch.ones_like(view_score))
        mask = mask.reshape(-1,).bool()
        # num_camera x batch_size
        return mask 



    @torch.no_grad()
    def compute_visibility(self, rays_o, rays_d, proj_depth, steps, batch_size=2**14):

        depth = torch.zeros_like(rays_o[...,:1])
        specular = torch.zeros_like(rays_o)

        for l in range(0, rays_o.shape[0], batch_size):
            out, _ = self.block.render_rays(rays_o[l:l+batch_size], rays_d[l:l+batch_size],
                                            occlusion_mask=None, mode=2)
            
            # out, _ = self.featureGrid.render_batch_rays(rays_o[l:l+batch_size], rays_d[l:l+batch_size], 
            #                                             z_vals[l:l+batch_size], dist[l:l+batch_size],
            #                             self.decoder, 1, out_diffuse=True, out_normal=False, global_step=steps,
            #                             contract_func=self.featureGrid.contract_fore)
            if out != None:
                depth[l:l+batch_size] = out["pred_depth"]
                specular[l:l+batch_size] = out["pred_specular"]

        vis_score = self.soft_vis(torch.abs(depth - proj_depth))
        diffuse_score = self.soft_diffuse(specular)


        return vis_score, diffuse_score


    
    """
    算法优化思路
    1. 输入点 pts = rays_o + depth * rays_d   B x 3 
    2. 为每个点选择 neighbor view 自己实现可微操作 topK  `view_selection` ->  nei_views  B x K
    3. 投影， 每个点投影到对应neighbor view 上 

    """

    @torch.no_grad()
    def view_selection(self, rays_o, rays_d, pts):
        """
        这里还没有判断 nei_occlusions 
        return 
            nei_views  B x K  (Int)
            nei_valid  B x K  (Bool)
        """
        
        batch_size = pts.shape[0]

        # N x B
        view_cost = torch.full((self.num_camera, batch_size), 1, dtype=torch.float32, device=self.device)
        computeViewcost(rays_o, rays_d, pts, self.poses.ks, self.poses.get_rts(), view_cost, self.H, self.W)

        # K x B
        topK_cost, nei_views = torch.topk(view_cost, k=self.topK, dim=0, largest=False)

        nei_valid = torch.ones_like(nei_views).bool() 
        nei_valid[topK_cost > 0.176] = False 

        nei_views = nei_views.permute(1,0).int().contiguous()
        nei_valid = nei_valid.permute(1,0).contiguous()

        return nei_views, nei_valid

    def projection(self, pts, rts, nei_views, nei_valid):
        """
        pts B x 3 
        rts N x 3 x 4
        nei_views B x K 
        nei_valid B x K 
        return 
        grid      B x K x 2
        nei_origin B x K x 3
        nei_direction B x K x 3
        proj_depth    B x K x 1 
        """
        grid, nei_origin, nei_direction = \
            proj_neighbor_view(pts, self.poses.ks, rts, nei_views, nei_valid)
        # print(grid[:5])
        # print(proj_depth[:5])
        proj_depth = grid[..., 2:]
        grid = grid[..., :2] / (proj_depth + 1e-8)
        
        grid = grid - 0.5 # this is because _add(0.5) from barf 
        
        # grid[..., 0] = grid[..., 0] / (self.W - 1) * 2. - 1.
        # grid[..., 1] = grid[..., 1] / (self.H - 1) * 2. - 1.

        return grid, nei_origin, nei_direction, proj_depth
    
    def sample_neighbor_color(self, grid, nei_views, nei_valid, occlusions):
        """
        grid     B x K x 2
        nei_views  B x K 
        nei_valid  B x K 
        occlusions N x H x W x 1 
        return 
        
            nei_color  B x K x 3 
            nei_valid  B x K x 1
        """ 

        batch_size, num_neighbors = grid.shape[:2]

        to_index = lambda x: x[..., 1] * self.W  + x[..., 0]

        # B x K x 2 
        lt_ijk = grid.long() # left top

        # right top
        rt_ijk = lt_ijk.clone()
        rt_ijk[...,0] += 1

        # left bottom 
        lb_ijk = lt_ijk.clone()
        lb_ijk[..., 1] += 1

        # right bottom 
        rb_ijk = lb_ijk.clone()
        rb_ijk[..., 0] += 1

        # rt_ijk = rt_ijk[...,0] + 1 # right top
        # lb_ijk = lt_ijk[...,1] + 1 # left bottom 
        # rb_ijk = lb_ijk[...,0] + 1 # right bottom 

        # print(lt_ijk.shape, rt_ijk.shape)
        # exit()

        # B x K x 2 
        local_offset = grid - lt_ijk.float()
        # B x K x 2 
        nearest_ijk = (grid + 0.5).long()
        # print(grid.min(), grid.max())
        # print(nearest_ijk.min(), nearest_ijk.max())
        # exit()
        """
        images   N x H x W x 3
        """
        with torch.no_grad():

            # B*K
            select_view_idx = nei_views.cpu().flatten().long()

            nei_valid = nei_valid & occlusions[select_view_idx, nearest_ijk[..., 1].cpu().flatten(), nearest_ijk[..., 0].cpu().flatten()].reshape(batch_size, num_neighbors).to(self.device)

            # B x K x 3 sample color
            lt_color = self.images[select_view_idx, lt_ijk[..., 1].cpu().flatten(), lt_ijk[..., 0].cpu().flatten()].to(self.device)
            rt_color = self.images[select_view_idx, rt_ijk[..., 1].cpu().flatten(), rt_ijk[..., 0].cpu().flatten()].to(self.device)
            lb_color = self.images[select_view_idx, lb_ijk[..., 1].cpu().flatten(), lb_ijk[..., 0].cpu().flatten()].to(self.device)
            rb_color = self.images[select_view_idx, rb_ijk[..., 1].cpu().flatten(), rb_ijk[..., 0].cpu().flatten()].to(self.device)

            lt_color = lt_color.reshape(batch_size, num_neighbors, 3)
            rt_color = rt_color.reshape(batch_size, num_neighbors, 3)
            lb_color = lb_color.reshape(batch_size, num_neighbors, 3)
            rb_color = rb_color.reshape(batch_size, num_neighbors, 3)

        # B x K compute weight 
        lt_weight = (1. - local_offset[... ,0]) * (1. - local_offset[... ,1])
        rt_weight = local_offset[... ,0] * (1. - local_offset[... ,1])
        lb_weight = (1. - local_offset[... ,0]) * local_offset[... ,1]
        rb_weight = local_offset[... ,0] * local_offset[... ,1]

        neighbor_color  = lt_weight[..., None] * lt_color + rt_weight[... ,None] * rt_color + \
                          lb_weight[..., None] * lb_color + rb_weight[..., None] * rb_color
        
        # print(neighbor_color.min(), neighbor_color.max())
        # exit()

        return neighbor_color, nei_valid

        

    def __call__(self, steps, rays_o, rays_d, depth, diffuse, specular, ray_colors, valid, occlusions, ori_poses_idxs):

        batch_size = rays_o.shape[0]
        num_camera = self.poses.ks.shape[0]
        
        selected = valid 

        rays_o = rays_o[selected]
        rays_d = rays_d[selected]
        depth = depth[selected]
        diffuse = diffuse[selected]
        specular = specular[selected]
        ray_colors = ray_colors[selected]


        # rays_o = rays_o[:10000]
        # rays_d = rays_d[:10000]
        # depth = depth[:10000]

        batch_size = rays_o.shape[0]

        if rays_o.shape[0] == 0:
            return None 

        # B x 3
        pts = rays_o + depth * rays_d


        nei_views, nei_valid = self.view_selection(rays_o, rays_d, pts)

        """
        grid            B x K x 2 
        nei_origin      B x K x 3
        nei_direction   B x K x 3 
        proj_depth      B x K x 1 
        """

        """
        DEBUG =========================================
        """

        rts = self.poses.get_rts()

        grid, nei_origin, nei_direction, proj_depth = self.projection(pts, rts, nei_views, nei_valid)
        neighbor_color, nei_valid = self.sample_neighbor_color(grid, nei_views, nei_valid, occlusions)

        # k = torch.ones_like(neighbor_color, requires_grad=False, device=self.device)
        # g = torch.torch.autograd.grad(
        #     outputs=neighbor_color,
        #     inputs=rts,
        #     grad_outputs=k,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        # print(g)
        # print(neighbor_color)
        # exit()

        # B x K x 3,  B x K 
        # neighbor_color, nei_valid = self.sample_neighbor_color(grid, nei_views, nei_valid, occlusions)
        # print(neighbor_color)
        # print(neighbor_color[:5])
        # print(nei_direction[:5])
        
        # N x B x 1 x 2, N x B x 3,  N x B x 1
        # grid, direction, proj_depth = self.proj_points2grid(pts, rts)

        # bilinear_color, mask = sample_color((self.images * 255).type(torch.uint8).to(self.device), grid)
        # bilinear_color = bilinear_color.squeeze() / 255.
        # bilinear_color = bilinear_color.permute(1,0,2)
        # bilinear_color = bilinear_color[torch.arange(batch_size, device=self.device)[:,None].repeat(1,self.topK).flatten().long(), nei_views.flatten().long(), :]
        # bilinear_color = bilinear_color.reshape(batch_size, self.topK, 3)
        # # # print(bilinear_color.shape)
        # print(bilinear_color[:5])
        # # B x N x 2
        # grid = grid[...,0,:].permute(1,0,2)
        # # # B x N x 3 
        # # direction = direction.permute(1,0,2)

        # # # nei_views B x K 
        # out_grid = grid[torch.arange(batch_size, device=self.device)[:,None].repeat(1,self.topK).flatten().long(), nei_views.flatten().long(), :]
        # out_grid = out_grid.reshape(batch_size, self.topK, 2)

        # k = torch.ones_like(bilinear_color, requires_grad=False, device=self.device)
        # g = torch.torch.autograd.grad(
        #     outputs=bilinear_color,
        #     inputs=rts,
        #     grad_outputs=k,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        # print(g)
        # print(bilinear_color)
        # exit()

        # direction = direction[torch.arange(batch_size, device=self.device)[:,None].repeat(1,self.topK).flatten().long(), nei_views.flatten().long(), :]
        # direction = direction.reshape(batch_size, self.topK, 3)
        # # print(grid.shape)
        # # print(grid[-5:])
        # print(direction[:5])
        """
        DEBUG =========================================
        """

        # exit()
        # print(grid[-1:])
        # # print(proj_depth)
        # print(nei_views[-1:])
        # print(nei_valid[-1:])
        # # print(grid.shape, nei_origin.shape, nei_direction.shape, proj_depth.shape)


        nei_origin = nei_origin[nei_valid]
        nei_direction = nei_direction[nei_valid]
        proj_depth = proj_depth[nei_valid]

        # print(nei_origin.shape, nei_direction.shape, proj_depth.shape)
        # exit()

        vis_score, nei_diffuse_score = self.compute_visibility(nei_origin, nei_direction, proj_depth, steps)
        if vis_score == None or nei_diffuse_score == None:
            return None 
        # print("vis_score", vis_score)
        # print("nei_diffuse_score", nei_diffuse_score)

        warping_score = torch.zeros((batch_size, self.topK, 1), dtype=torch.float32, device=self.device)
        warping_score[nei_valid] = vis_score * nei_diffuse_score
        with torch.no_grad():        
            ref_diffuse_score = self.soft_diffuse(specular)
            warping_score = warping_score * ref_diffuse_score[:,None,:]
        #     print("ref_diffuse_score", ref_diffuse_score)
        # exit()

        pred_color = torch.clamp(diffuse + specular, 0, 1)
        wrp_loss = self.compute_loss(pred_color[:, None, :].repeat(1,self.topK,1), neighbor_color, warping_score)


        return wrp_loss 





