import torch
import torch.nn as nn 
import sys 
from tools import utils 
from tools import tools 


class DepthConsistencyLoss:
    # copy from MiDaS
    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1
    
    def toCPU(self):
        pass

    def toGPU(self):
        pass

    def __call__(self, rendered_depth, monocular_depth, mask):
        """
        N is the number of cameras
        B is the batchsize 
        rendered_depth  N x B x 1
        monocular_depth N x B x 1
        """
        h0, h1 = self.compute_scale_and_shift(rendered_depth, monocular_depth, mask)

        # N x B x 1
        scaled_depth = rendered_depth * h0[:,None,None] + h1[:,None,None]

        loss = utils.Mask_MSELoss(scaled_depth, monocular_depth, mask)
        
        return loss 


class DepthSmoothLoss:
    def __init__(self, cfg, block):
        self.poses = block.poses
        self.num_camera = block.num_camera

    def toCPU(self):
        pass 

    def toGPU(self):
        pass 
    
    def __call__(self, pixel_locs, rendered_depth, monocular_normal, mask):
        """
        patch size 2 x 2 
        pixel_locs       N x B x 2
        rendered_depth   N x B x 1   (patch size = 2)
        monocular_normal N x B x 3 
        mask             N x B x 1
        """
        
        # N x B x 3
        pixel_locs = torch.cat([pixel_locs, torch.ones_like(pixel_locs[...,:1])], -1)

        # N x B x 3
        pts_cam = torch.sum(self.poses.ks.inverse()[:,None,...] * pixel_locs[..., None, :],dim=-1) * rendered_depth

        # N x NP x 2 x 2 x 3 
        pts_cam = pts_cam.reshape(self.num_camera, -1, 2, 2, 3)
        # N x NP x 2 x 2 x 3 
        monocular_normal = monocular_normal.reshape(self.num_camera, -1, 2, 2, 3)
        # N x NP x 4
        mask = mask.reshape(self.num_camera, -1, 4)

        # the monocular normal of 4 pixels should be similar 
        # N x NP x 3 
        mean_normal = torch.mean(monocular_normal, dim=(2,3))
        # N x NP x 2 x 2
        sim = torch.sum(monocular_normal * mean_normal[..., None, None, :], dim=-1)
        # N x NP
        valid = torch.all(sim.reshape(self.num_camera, -1, 4) > 0.9, dim=-1) & torch.all(mask==True, dim=-1)

        mean_normal = torch.nn.functional.normalize(mean_normal[valid], p=2, dim=-1)

        # N x NP x 2 x 3
        grad_y = torch.nn.functional.normalize(pts_cam[..., 1, :, :] - pts_cam[..., 0, :, :], p=2, dim=-1)[valid]
        grad_x = torch.nn.functional.normalize(pts_cam[..., :, 1, :] - pts_cam[..., :, 0, :], p=2, dim=-1)[valid]
        
        # N x NP 
        cos = 0.5 * torch.abs(torch.sum(mean_normal[...,None,:]*grad_y, dim=-1)).mean(-1) + \
              0.5 * torch.abs(torch.sum(mean_normal[...,None,:]*grad_x, dim=-1)).mean(-1)

        return torch.mean(cos)
