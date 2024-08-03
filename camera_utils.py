from cv2 import norm
import torch 
import torch.nn as nn 
import camera 
import numpy as np 
from cuda import compute_ray_forward, compute_ray_backward
from easydict import EasyDict as edict



@torch.no_grad()
def prealign_cameras(pose,pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3,device=pose.device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=pose.device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    return pose_aligned,sim3

@torch.no_grad()
def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error

class CAM(nn.Module):
    def __init__(self, ks, c2ws, device, noise, gt_c2ws=None):

        super(CAM, self).__init__()

        self.device = device

        self.num_camera = c2ws.shape[0]

        self.ori_rts = camera.pose.invert(c2ws).to(self.device)

        self.se3_refine = nn.Parameter(torch.zeros((self.num_camera, 6), dtype=torch.float32, device=self.device))

        # if noise_weight == 0:
        #     self.rts = self.ori_rts.clone()
        # else:
        noise = noise.to(self.device)
        self.rts = camera.pose.compose([camera.lie.se3_to_SE3(noise), self.ori_rts.clone()])

        if gt_c2ws == None:
            self.gt_rts = self.ori_rts.clone()
        else:
            self.gt_rts = camera.pose.invert(gt_c2ws).to(self.device)
        
        self.ks = ks.to(self.device)

    def getRays(self, H, W, ray_idx=None, view_idx=None):
        rts = self.get_rts()

        if view_idx != None:
            rts = rts[view_idx]
            ks = self.ks[view_idx]
        else:
            ks = self.ks 
    

        # B x HW x 3 
        # rays_o, rays_d = camera.get_center_and_ray(H, W, rts, ks)
        if ray_idx != None:
            # B x batchsize x 3 
            # rays_o = rays_o[:, ray_idx]
            # rays_d = rays_d[:, ray_idx]
            return camera.get_center_and_ray_v2(H, W, rts, ks, ray_idx)
        else:
            return camera.get_center_and_ray(H, W, rts, ks)
        # return rays_o, rays_d 

    def get_rts(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        return rts     

    def getRays_v2(self, H, W, se3_refine, ray_idx=None):
        rts = self.get_rts_v2(se3_refine)

        # B x HW x 3 
        # rays_o, rays_d = camera.get_center_and_ray(H, W, rts, ks)
        if ray_idx != None:
            # B x batchsize x 3 
            # rays_o = rays_o[:, ray_idx]
            # rays_d = rays_d[:, ray_idx]
            return camera.get_center_and_ray_v2(H, W, rts, self.ks, ray_idx)
        else:
            return camera.get_center_and_ray(H, W, rts, self.ks)
        # return rays_o, rays_d

    def get_rts_v2(self, se3_refine):
        rts_refine = camera.lie.se3_to_SE3(se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        return rts   

    def get_poses(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        return camera.pose.invert(rts)

    def evaluate(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        rts_aligned, _ = prealign_cameras(rts, self.gt_rts)
        error = evaluate_camera_alignment(rts_aligned, self.gt_rts)

        return np.rad2deg(error.R.mean().cpu()), error.t.mean()


### =========================== For interpolate camera poses =======================================


def normalize(x):
    return x / np.linalg.norm(x)

def normalize_torch(x):
    return x / torch.norm(x)


def path360(focus, forward, radius, num):

    C_o = focus.clone()
    C_o[2] -= radius
    theta = torch.arange(num)/num*2*np.pi
    R_x = camera.angle_to_rotation_matrix((theta.sin()*0.05).asin(),"X")
    R_y = camera.angle_to_rotation_matrix((theta.cos()*0.05).asin(),"Y")
    R = R_y @ R_x

"""
give two poses, interpolate
"""
def interpolate_poses(c2w_a, c2w_b, num):
    poses = np.zeros((num, 3, 4), dtype=np.float32)
    up_axis = 0.5 * c2w_a[:, 1] + 0.5 * c2w_b[:, 1]

    idx = 0
    for step in np.linspace(0, 1, num):
        center = c2w_a[:, 3] * (1-step) + c2w_b[:, 3] * step 
        z_axis = c2w_a[:, 2] * (1-step) + c2w_b[:, 2] * step 
        x_axis = np.cross(up_axis, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = normalize(x_axis)
        y_axis = normalize(y_axis)
        z_axis = normalize(z_axis)
        c2w = np.stack([x_axis, y_axis, z_axis, center], axis=-1)
        poses[idx] = c2w
        idx += 1
    return poses 

def interpolate_poses_torch(c2w_a, c2w_b, num):
    poses = torch.zeros((num, 3, 4), dtype=torch.float32, device=c2w_a.device)
    up_axis = 0.5 * c2w_a[:, 1] + 0.5 * c2w_b[:, 1]

    idx = 0
    for step in np.linspace(0, 1, num):
        center = c2w_a[:, 3] * (1-step) + c2w_b[:, 3] * step 
        z_axis = c2w_a[:, 2] * (1-step) + c2w_b[:, 2] * step 
        x_axis = torch.cross(up_axis, z_axis)
        y_axis = torch.cross(z_axis, x_axis)
        x_axis = normalize_torch(x_axis)
        y_axis = normalize_torch(y_axis)
        z_axis = normalize_torch(z_axis)
        c2w = torch.stack([x_axis, y_axis, z_axis, center], dim=-1)
        poses[idx] = c2w
        idx += 1
    return poses 
