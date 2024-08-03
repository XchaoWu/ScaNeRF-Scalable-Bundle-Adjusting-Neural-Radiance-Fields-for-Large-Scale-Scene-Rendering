import torch 
import torch.nn as nn 
from cfg import * 

class ConsensusManager:
    def __init__(self, cfg, tile):

        self.cfg = cfg 

        self.H = tile.H 
        self.W = tile.W

        self.tile = tile 
        self.poses = tile.poses

        self.device = self.tile.device

        self.shared_se3 = self.poses.se3_refine.clone()
        self.delta_se3 = torch.zeros((self.tile.num_camera, 6), dtype=torch.float32, device=self.device)
        self.overlap_flags = torch.zeros((self.tile.num_camera), dtype=torch.bool, device=self.device)
        self.rho = torch.ones((6,), dtype=torch.float32, device=self.device) * self.cfg.RHO

        print("finished setting admm context")

    def export_check_point(self):
        shared_se3 = self.shared_se3.detach().cpu().numpy()
        delta_se3 = self.delta_se3.detach().cpu().numpy()
        overlap_flags = self.overlap_flags.detach().cpu().numpy()
        rho = self.rho.detach().cpu().numpy()
        return {"shared_se3": shared_se3, "delta_se3": delta_se3,
                "overlap_flags": overlap_flags, "rho": rho}

    def load_check_point(self, ckp):
        func = lambda x: torch.from_numpy(x).to(self.device)
        self.shared_se3 = func(ckp["shared_se3"])
        self.delta_se3 = func(ckp["delta_se3"])
        self.overlap_flags = func(ckp["overlap_flags"])
        self.rho = func(ckp["rho"])

    def update(self, shared_se3, overlap_idxs):
        self.toCPU()
        self.shared_se3 = shared_se3
        with torch.no_grad():
            # over-relaxation
            self.delta_se3 = self.delta_se3 + (1 + 0.5) * (self.poses.se3_refine.cpu() - self.shared_se3)
        # self.shared_pts = shared_pts
        if overlap_idxs.shape[0] > 0:
            self.overlap_flags[overlap_idxs] = True
        self.toGPU()
        print(f"TILE {self.cfg.TILEIDX} has {self.tile.num_camera} cameras, including {overlap_idxs.shape[0]} overlap cameras")
        

    def toCPU(self):
        self.shared_se3 = self.shared_se3.cpu()
        self.delta_se3 = self.delta_se3.cpu()
        self.overlap_flags = self.overlap_flags.cpu()
        self.rho = self.rho.cpu()
        # self.shared_pts = self.shared_pts.cpu()
        # self.ray_idx = self.ray_idx.cpu()
    
    def toGPU(self):
        self.shared_se3 = self.shared_se3.to(self.device)
        self.delta_se3 = self.delta_se3.to(self.device)
        self.overlap_flags = self.overlap_flags.to(self.device)
        self.rho = self.rho.to(self.device)
        # self.shared_pts = self.shared_pts.to(self.device)
        # self.ray_idx = self.ray_idx.to(self.device)

    
    def camera_loss(self):
        # with torch.no_grad():
        #     # over-relaxation
        #     self.delta_se3 = self.delta_se3 + (1 + 0.5) * (self.poses.se3_refine - self.shared_se3)
        admm_constrain =  (self.poses.se3_refine - self.shared_se3 + self.delta_se3) ** 2
        loss = torch.mean(self.rho[None, :] * admm_constrain[self.overlap_flags])
        return loss

    def __call__(self):
        if self.overlap_flags.sum() > 0:
            return self.camera_loss()
        else:
            return None 