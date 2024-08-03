import torch 
import numpy as np 
import os,sys,cv2
from datetime import datetime
import copy 
from fastMesh import FastMesh
from consensus import ConsensusManager
from cuda import ray_aabb_intersection
from easydict import EasyDict as edict
from hashgrid import HashGrid
from cfg import * 
from tqdm import tqdm 
from glob import glob 
import time 
import time, yaml, network, camera_utils 
from load_data import load_snisr, read_campara
from scheduler import Scheduler, SchedulerManager
from criterions import Criterions 
from tools import utils
from tools import tools 

class TILE:
    def __init__(self, cfg, tileIdx, gpuIdx, testIdxs, novelIdxs, fmesh, device, enable_admm):
        self.cfg = copy.deepcopy(cfg) 
        self.cfg.TILEIDX = tileIdx
        self.cfg.GPUIDX = gpuIdx
        self.cfg.TEST_IDX = testIdxs
        self.cfg.NOVEL_IDX = novelIdxs
        self.device = device
        self.fmesh = fmesh # global 
        self.enable_admm = enable_admm

        self.commit_shared_depth = False 

        self.ckp = None  

        try:
            if self.cfg.CKP != "":
                files = glob(os.path.join(self.cfg.CKP, f"{self.cfg.TILEIDX}-GPU*", f"checkpoint-*-{self.cfg.TILEIDX}.pt"))
                files.sort(key=lambda x:int(os.path.basename(x).split("-")[1]), reverse=True)
                path = files[0]
                print(f"start load ckp from {path} ...")
                self.ckp = torch.load(path) 
        except:
            pass

    def build_training_context(self):

        self.create_logdir()
        # input("here 1.625G Mem")
        self.create_training_model()
        # input("here 6.716G Mem")
        self.create_training_data()
        # input("here 6.716G Mem")
        self.create_optimizer()
        # input("here 23.373G Mem")
        # self.render_shared_depth()

        self.grow_th = 0.6
        self.prune_th = 0.1 # grow from 0.1 to 0.4
        # pruning
        self.dynamic_start = 0
        # no pruning 
        self.dynamic_end = self.cfg.TRAINING.TOTAL_STEP - 10000
        self.dynamic_step = 5000 # disable pruning  

        if self.ckp != None:
            self.global_step = self.ckp["global_step"]
        else:
            self.global_step = 1

        # camera confidence 
        self.confidence = torch.ones((self.num_camera), dtype=torch.float32)

    def create_logdir(self):
        # create log file 
        # runtime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        logdir = os.path.join(self.cfg.LOGDIR, f"{self.cfg.TILEIDX}-GPU{self.cfg.GPUIDX}")
        if os.path.exists(logdir):
            os.system(f"rm -rf {logdir}")
        os.mkdir(logdir)
        print(f"set log dir to {logdir}")

        # write descriptions and copy yaml file to logdir 
        # with open(os.path.join(logdir, "description.txt"), "w") as f:
        #     f.write(self.cfg.DESCRIPTION)
        # os.system(f"cp {self.cfg.yaml} {logdir}")
        # print(f"copy the yaml file to {logdir}")
        self.logdir = logdir
    
    def create_training_model(self, visualize=False):



        with open(os.path.join(self.cfg.DATADIR, "tiles", "training_views.txt"), "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if int(lines[i].strip()) == self.cfg.TILEIDX:
                    self.cfg.VISIBLE_POSES = [int(item) for item in lines[i+1].strip().split(" ")]
                    break

        with open(os.path.join(self.cfg.DATADIR, "tiles", "tile_info.txt"), "r") as f:
            lines = f.readlines()[1:]
            lines = [line.strip().split(" ") for line in lines]
            for line in lines:
                if int(line[0]) == self.cfg.TILEIDX:
                    self.tile_corner = torch.tensor([float(line[1]),float(line[2]), float(line[3])], dtype=torch.float32, device=self.device)
                    self.tile_size = torch.tensor([float(line[4]),float(line[5]), float(line[6])], dtype=torch.float32, device=self.device)
                    self.resolution = [int(line[7]), int(line[8])]
                    init_outside = int(line[9])
        
        try:
            near=self.cfg.TRAINING.NEAR
            far=self.cfg.TRAINING.FAR
        except:
            near=None; far=None 

        log2_hashmap_size = self.cfg.HASHGRID.LOG2_HASHMAP_SIZE

        self.featureGrid = HashGrid(self.device, self.tile_corner, self.tile_size, 
                                    log2_hashmap_size=log2_hashmap_size,
                                    grid_resolution=self.resolution,
                                    sampler_log2dim=self.cfg.TRAINING.GRID_LOG2DIM[0], 
                                    init_outside=init_outside==1, 
                                    model_path=self.cfg.MESH,
                                    near=near,
                                    far=far)
        if self.ckp != None:
            self.featureGrid.load_check_point(self.ckp["hashgrid"])
            print("successfully load ckp to hash grid!\n")

        print("finished building Hash tree")
        self.decoder = network.ShallowMLP(in_channel=32).to(self.device)

        if self.ckp != None:
            self.decoder.load_state_dict(self.ckp["decoder"])
            print("successfully load ckp to decoder!\n")
        else:
            network.init_model(self.decoder, "xavier")
        print("finished building decoder")

        self.featureGrid.vis_gird(os.path.join(self.logdir), bg=False)

        self.featureGrid.vis_gird(os.path.join(self.logdir), bg=True)

        # self.featureGrid.pruning_grid(self.decoder, 5, 0.4)
        # exit()

        if self.fmesh != None:
            self.fmesh.set(self.featureGrid.bbox_center, self.featureGrid.bbox_size / 2.0)


    def create_training_data(self):
        # if len(self.cfg.TEST_IDX) == 0:
        #     inference_idx = [self.cfg.VISIBLE_POSES[0]]
        #     self.cfg.VISIBLE_POSES = self.cfg.VISIBLE_POSES[1:]
        # else:
        #     inference_idx = self.cfg.TEST_IDX
        #     # self.cfg.VISIBLE_POSES = set(self.cfg.VISIBLE_POSES + self.cfg.TEST_IDX

        # # load testing data 
        # test_images, _, test_c2ws, test_ks, _, _, _, _, _ = load_snisr(self.cfg.DATADIR, inference_idx)
        # view_indices = list(set(self.cfg.VISIBLE_POSES))

        view_indices = list(set(self.cfg.VISIBLE_POSES))
        view_indices.sort(key=self.cfg.VISIBLE_POSES.index)
        # view_indices = list(set(self.cfg.VISIBLE_POSES + self.cfg.TEST_IDX))
        # view_indices.sort()
        view_indices = view_indices[:self.cfg.MAX_POSES]
        view_indices = list(set(view_indices + self.cfg.TEST_IDX))
        # print(view_indices)
        images, _, c2ws, ks, H, W, mono_depths, mono_normals, self.cfg.VISIBLE_POSES = \
                            load_snisr(self.cfg.DATADIR, view_indices, 
                            self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS > 0 or self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0, 
                            self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0)                                                             

        points = tools.cameras_scatter(c2ws[:,:3,:3].transpose(0,2,1), c2ws[:,:3,3], length=0.8)
        tools.points2obj(os.path.join(self.logdir, "camera.obj"), points)

        # split testing data from this 
        if len(self.cfg.TEST_IDX) == 0:
            inference_idx = [0]
        else:
            inference_idx = []
            for idx, item in enumerate(self.cfg.VISIBLE_POSES):
                if item in self.cfg.TEST_IDX:
                    inference_idx.append(idx)
        # print(inference_idx)
        # print(self.cfg.TEST_IDX)
        # print(self.cfg.VISIBLE_POSES)
        # exit()

        # # ========== TEST ===============
        # import cv2 
        # test_dir = os.path.join(self.logdir, "training_images")
        # os.mkdir(test_dir)
        # for idx,img in enumerate(images):
        #     cv2.imwrite(os.path.join(test_dir,f"{self.cfg.VISIBLE_POSES[idx]}.png"), img*255)
        # exit()    

        """
        All in CPU 
        """
        c2ws = torch.from_numpy(c2ws).float()
        ks = torch.from_numpy(ks).float()
        images = torch.from_numpy(images).float()
        # depths = torch.from_numpy(depths).float()
        if self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS > 0 or self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0:
            mono_depths = torch.from_numpy(mono_depths).float()
        if self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0:
            mono_normals = torch.from_numpy(mono_normals).float()
        self.num_camera = ks.shape[0]

        # test_c2ws = torch.from_numpy(test_c2ws).float()
        # test_ks = torch.from_numpy(test_ks).float()



        #  
        try:
            _, gt_c2ws, _, _ = tools.read_campara(os.path.join(self.cfg.DATADIR, "camera-gt.log"))
        except:
            gt_c2ws = None 
        #

        # gen poses
        self.poses = camera_utils.CAM(ks, c2ws, self.device, noise=self.cfg.NOISE[self.cfg.VISIBLE_POSES],
                                      gt_c2ws=gt_c2ws) 

        # """
        # occlusion only works for cameras outside the tile
        # """
        occlusions = []

        if self.fmesh != None:
            trust_mesh = False 
            print("\nCompute occlusion ...\n")
            for i in tqdm(range(self.num_camera)):
                rays_o, rays_d = self.poses.getRays(H, W, view_idx=[i])
                occlusions += [self.fmesh.render_mask(rays_o[0], rays_d[0], trust_mesh=trust_mesh).reshape(H,W,1).cpu()]
            occlusions = torch.stack(occlusions, 0)
            # print(occlusions.shape, occlusions.dtype)
            torch.cuda.empty_cache()
        else:
            # files = glob(os.path.join(self.cfg.DATADIR, "masks", "*.png"))
            # files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            # print("\nLoading occlusion ...\n")
            # for f in tqdm(files):
            #     occlusions += [torch.from_numpy(cv2.imread(f)[...,:1] / 255.).bool()]
            # occlusions = torch.stack(occlusions, 0)
            occlusions = torch.ones((self.num_camera, H, W, 1)).bool()

        # num_camera
        # 防止 patch 溢出图片 
        # occlusions[:,:,-1,0] = False 
        # occlusions[:,-1,:,0] = False 
        # self.num_valid_pixel_per_view = occlusions.reshape(self.num_camera,-1).sum(-1).int()
        # self.sample_pidxs = torch.ones((self.num_camera, H*W), dtype=torch.int32) * -1
        # for idx,occl in enumerate(tqdm(occlusions)):
        #     ys, xs = torch.where(occl[...,0]==True)
        #     pidxs = ys * W + xs
        #     self.sample_pidxs[idx, :len(pidxs)] = pidxs
        # print(sample_pidxs[0])
        # exit()

        # ======== TEST =======
        # import cv2 
        # test_dir = os.path.join(self.logdir, "occlusions")
        # os.mkdir(test_dir)
        # for idx in range(images.shape[0]):
        #     temp_mask = occlusions[idx].float().detach()
        #     temp_mask[temp_mask == 0] = 0.2
        #     out = (images[idx] * temp_mask).numpy()*255
        #     cv2.imwrite(os.path.join(test_dir,f"{self.cfg.VISIBLE_POSES[idx]}.png"), out)
        # exit()
        
        self.H = H 
        self.W = W 

        if len(self.cfg.NOVEL_IDX) > 0:
            novel_ks, novel_c2ws = tools.read_campara(os.path.join(self.cfg.DATADIR, "renderPath.log"))
            novel_H =  720
            novel_W = 1280
            novel_ks[:,0,2] = novel_W / 2.
            novel_ks[:,1,2] = novel_H / 2. 
            self.novel = edict({"ks": torch.from_numpy(novel_ks)[self.cfg.NOVEL_IDX], 
                                "c2ws": torch.from_numpy(novel_c2ws)[self.cfg.NOVEL_IDX], 
                                "H": novel_H, "W": novel_W})


        self.train_data = edict({"c2ws": c2ws, "ks": ks, "images": images,
                                #   "occlusions": occlusions.to(self.device),
                                  "occlusions": occlusions,
                                  "mono_depths": mono_depths, "mono_normals": mono_normals})
        # self.test_data = edict({"c2ws": test_c2ws, "ks": test_ks, "images": test_images})
        self.inference_idx = inference_idx
        print("finished loading training data")

    def create_optimizer(self):

        self.featureGrid_optimizer = torch.optim.Adam([{"params": self.featureGrid.parameters(), "lr": self.cfg.TRAINING.ETA.HASH_FEATURE, "betas": (0.9, 0.99), "eps":1e-15}])
        if self.ckp != None:
            self.featureGrid_optimizer.load_state_dict(self.ckp["featureGrid_optimizer"])
            print("successfully load ckp to featureGrid_optimizer!\n")

        self.featureGrid_sche = SchedulerManager([Scheduler("featureGrid", self.cfg.TRAINING.ETA.HASH_FEATURE, 0.1 * self.cfg.TRAINING.ETA.HASH_FEATURE, self.cfg.TRAINING.TOTAL_STEP)])
        # input("here 10.802G")

        params = [{"params": self.decoder.parameters(), "lr": self.cfg.TRAINING.ETA.DECODER, "weight_decay":1e-6}]

        scheduler_list = [Scheduler("decoder", self.cfg.TRAINING.ETA.DECODER, 0.1 * self.cfg.TRAINING.ETA.DECODER, self.cfg.TRAINING.TOTAL_STEP, groups=[0])]
        

        index = 1
        # input("here 10.802G")
        if self.cfg.TRAINING.CAMOPT.ENABLE:
            params += [{"params":self.poses.se3_refine, "lr":self.cfg.TRAINING.ETA.CAM}]
            scheduler_list += [Scheduler("cam", self.cfg.TRAINING.ETA.CAM, 
                                        0.1 * self.cfg.TRAINING.ETA.CAM,
                                        self.cfg.TRAINING.TOTAL_STEP, groups=[index], 
                                        start_itr=self.cfg.TRAINING.CAMOPT.START_STEPS, 
                                        end_itr=self.cfg.TRAINING.TOTAL_STEP)]
            index+=1


        self.optimizer = torch.optim.Adam(params)
        if self.ckp != None:
            self.optimizer.load_state_dict(self.ckp["optimizer"])
            print("successfully load ckp to optimizer!\n")


        self.sche = SchedulerManager(scheduler_list)
        # input("here 10.802G")

        if self.enable_admm:
            self.consensus_manager = ConsensusManager(self.cfg, self)
            if self.ckp != None:
                self.consensus_manager.load_check_point(self.ckp["admm"])
                print("successfully load ckp to admm!\n")
            
        # loss 
        self.crit = Criterions(self.cfg, self)
        # input("here 14.757G if no muti level")
    
    def set(self, shared_info):
    #     self.shared_se3 = self.poses.se3_refine.clone()
    #     self.delta_se3 = torch.zeros((self.num_camera, 6), dtype=torch.float32, device=self.device)
    #     self.overlap_flags = torch.zeros((self.num_camera), dtype=torch.bool, device=self.device)
    #     self.rho = torch.ones((6,), dtype=torch.float32, device=self.device) * self.cfg.RHO
        self.shared_info = shared_info
    #     print("finished setting admm context")

    @torch.no_grad()
    def update_confidence(self, pred, gt, valid):
        # N x B 
        score = torch.zeros_like(gt[...,0]).cpu().reshape(self.num_camera, -1)
        # -1 x 1 
        socre_itr = 1. - torch.abs(pred.detach() - gt).mean(-1)
        score.reshape(-1)[valid] = socre_itr[valid].cpu()
        self.confidence = 0.9 * self.confidence + 0.1 * score.mean(-1)

    # @torch.no_grad()
    # def rendering_overlapped_depth(self):
    #     for idx in self.overlap_idxs:

    @torch.no_grad()
    def update_occlusion_mask(self):
        
        kernel_size = 91
        # kernel_size = 3
        kernel = torch.ones((1,1,kernel_size,kernel_size), dtype=torch.float32)
        # N x H x W x 1
        occlusions = torch.ones((self.num_camera,self.H,self.W,1)).bool()
        bbox_center = self.featureGrid.bbox_center
        bbox_size = self.featureGrid.bbox_size / 2.

        for idx, ori_idx in enumerate(tqdm(self.cfg.VISIBLE_POSES, desc="updating mask")):
            # idx 当前tile内的相机index -> ori_idx对应所有相机的index 
            if self.shared_info.shared_depth[ori_idx] == None:
                continue 
 
            rays_o, rays_d = self.poses.getRays(self.H, self.W, 
                                        ray_idx=None, view_idx=[idx])
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)

            inside = torch.all(torch.abs(rays_o[0] - bbox_center) < (bbox_size / 2.), dim=-1)
            if inside.float().sum() == 1:
                continue 
            depth = self.shared_info.shared_depth[ori_idx]
            # depth = depth.reshape(-1,1)
            depth = depth.repeat_interleave(2,0).repeat_interleave(2,1).reshape(-1,1)
            bounds = torch.ones_like(rays_o[...,:2]) * -1
            ray_aabb_intersection(rays_o, rays_d, bbox_center, bbox_size, bounds)
            bounds = bounds.detach().cpu()
            occlusion = ((depth > bounds[..., :1]) & (bounds[..., :1] != -1)).reshape(1, 1, self.H, self.W)
            occlusion = 1.0 - torch.nn.functional.conv2d(1.0 - occlusion.float(), kernel, padding=(kernel_size//2, kernel_size//2)).clamp(0,1)
            occlusion = occlusion.bool()

            occlusions[idx] = occlusion.reshape(self.H, self.W, 1)

        # occlusions[:,:,-1,0] = False 
        # occlusions[:,-1,:,0] = False 
        # self.num_valid_pixel_per_view = occlusions.reshape(self.num_camera,-1).sum(-1).int()
        # self.sample_pidxs = torch.ones((self.num_camera, self.H*self.W), dtype=torch.int32) * -1
        # for idx,occl in enumerate(tqdm(occlusions)):
        #     ys, xs = torch.where(occl[...,0]==True)
        #     pidxs = ys * self.W + xs
        #     self.sample_pidxs[idx, :len(pidxs)] = pidxs
        #     # self.sample_pidxs[idx, len(pidxs):] =  pidxs
        # self.train_data.occlusions = occlusions.to(self.device)
        self.train_data.occlusions = occlusions 

        # ======== TEST =======
        # import cv2 
        # images = self.train_data.images
        # test_dir = os.path.join(self.logdir, "occlusions")
        # os.mkdir(test_dir)
        # for idx in range(images.shape[0]):
        #     temp_mask = occlusions[idx].float().detach()
        #     temp_mask[temp_mask == 0] = 0.2
        #     depth = self.shared_info.shared_depth[self.cfg.VISIBLE_POSES[idx]]
        #     if depth != None:
        #         depth = depth.detach().numpy()
        #         depth = depth / depth.max() * 255 
        #         cv2.imwrite(os.path.join(test_dir,f"depth_{self.cfg.VISIBLE_POSES[idx]}.png"), depth)

        #     out = (images[idx] * temp_mask).numpy()*255
        #     cv2.imwrite(os.path.join(test_dir,f"{self.cfg.VISIBLE_POSES[idx]}.png"), out)
        # exit()

    @torch.no_grad()
    def render_shared_depth(self):
        bbox_center = self.featureGrid.bbox_center
        bbox_size = self.featureGrid.bbox_size / 2.
        for idx, ori_idx in enumerate(tqdm(self.cfg.VISIBLE_POSES, desc="render shared depth")):
            if idx not in self.overlap_idxs:
                continue 
            rays_o, rays_d = self.poses.getRays(self.H, self.W, 
                                        ray_idx=None, view_idx=[idx])
            rays_o = rays_o.reshape(self.H,self.W,3)[::2,::2].reshape(-1,3)
            rays_d = rays_d.reshape(self.H,self.W,3)[::2,::2].reshape(-1,3)
            # rays_o = rays_o.reshape(-1,3)
            # rays_d = rays_d.reshape(-1,3)
            inside = torch.all(torch.abs(rays_o[0] - bbox_center) < (bbox_size / 2.), dim=-1)
            # print(ori_idx, inside)
            if inside.float().sum() == 0:
                continue 

            # bounds = torch.ones_like(rays_o[...,:2]) * -1
            # ray_aabb_intersection(rays_o, rays_d, bbox_center, bbox_size / 2.0, bounds)
            # valid = (bounds[:, 1] != -1) 

            # first enter 
            # depth = torch.ones_like(rays_o[...,:1]) * 1e8
            # z_vals, _ = self.featureGrid.samplePoints(rays_o, rays_d, 128)
            # valid = torch.all(z_vals != -1, dim = -1) 
            # if torch.all(z_vals[valid,:1] == 0):
            #     self.fmesh.fmesh.firstEnter(rays_o, rays_d, depth)
            # else:
            #     depth[valid] = z_vals[valid,:1]
            
            # depth = depth.detach().cpu()
            depth = self.render_depth_rays(rays_o, rays_d)

            # with self.shared_info.shared_lock:
            #     shared_depth = self.shared_info.shared_depth[ori_idx]
            #     shared_depth = shared_depth.reshape(-1,1)
            #     update_idx = shared_depth > depth
            #     shared_depth[update_idx] = depth[update_idx]
            #     self.shared_info.shared_depth[ori_idx] = shared_depth.reshape(self.H, self.W, 1)
            # depth = xxx 
            # self.shared_info.shared_depth[ori_idx] = depth.reshape(self.H, self.W, 1).detach().cpu()
            self.shared_info.shared_depth[ori_idx] = depth.reshape(self.H//2, self.W//2, 1).detach().cpu()
        self.commit_shared_depth = True 

    def commit(self):
        """
        commit info to master process 
        """
        # with torch.no_grad():
        #     pts, valid = self.consensus_manager.backproj2points(INFERENCE)
        #     if pts != None:
        #         pts = pts.cpu().detach()
        #         valid = valid.cpu().detach()
        # self.shared_info.shared_mem[self.cfg.TILEIDX] = {"pose": self.poses.se3_refine.cpu().detach(),
        #                                                   "idx": self.cfg.VISIBLE_POSES,
        #                                                   "pts": pts, "valid": valid}
        self.shared_info.shared_mem[self.cfg.TILEIDX] = {"pose": self.poses.se3_refine.cpu().detach(),
                                                          "idx": self.cfg.VISIBLE_POSES,
                                                          "confidence": self.confidence}
        
        with self.shared_info.shared_lock:
            self.shared_info.shared_count.value += 1 
            print(f"Tile {self.cfg.TILEIDX} finished commiting count {self.shared_info.shared_count.value}")

    def synchronize(self):
        """
        syn info to master process
        """
        # self.consensus_manager.update(self.shared_info.shared_mem[self.cfg.TILEIDX]["shared_poses"],
        #                  self.shared_info.shared_mem[self.cfg.TILEIDX]["shared_pts"],
        #                  self.shared_info.shared_mem[self.cfg.TILEIDX]["overlap_idxs"])
        self.overlap_idxs = self.shared_info.shared_mem[self.cfg.TILEIDX]["overlap_idxs"]

        self.consensus_manager.update(self.shared_info.shared_mem[self.cfg.TILEIDX]["shared_poses"], self.overlap_idxs)
        
        print(f"Tile {self.cfg.TILEIDX} global step {self.global_step} -> finished synchronizing")

    def export_tile(self):
        """
        export tile topology 
        export tile atlas 
        export decoder MLP parameters 
        """
        output_dir = os.path.join(self.cfg.LOGDIR,f"tile-{self.cfg.TILEIDX}")
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)

        self.featureGrid.export(output_dir)
        torch.save(self.decoder.state_dict(), os.path.join(output_dir, "decoder.pth"))

        # if self.cfg.SCENE == "outdoor":
        #     torch.save(self.sky_mlp.state_dict(), os.path.join(output_dir, "sky.pth"))

        with torch.no_grad():
            c2ws = self.poses.get_poses().detach().cpu().numpy()
        np.savez(os.path.join(output_dir, f"cams.npz"), 
            c2ws=c2ws, ks=self.poses.ks.detach().cpu().numpy(), idxs=np.array(self.cfg.VISIBLE_POSES))
        
        print(f"export Tile to {output_dir}")
    

    def export_check_point(self, output_dir):


        print("start export checkpoint ... \n")
        """
        export check 
        """
        check_point = {
            "global_step": self.global_step,

        }

        """
        feature grid 
        features & occupied grid 
        """
        check_point["hashgrid"] = self.featureGrid.export_check_point()

        """
        ADMM shared
        """
        check_point["admm"] = self.consensus_manager.export_check_point()

        """
        decoder
        """
        check_point["decoder"] = self.decoder.state_dict()

        """
        Optimizer
        """
        check_point["featureGrid_optimizer"] = self.featureGrid_optimizer.state_dict()
        check_point["optimizer"] = self.optimizer.state_dict()

        save_path = os.path.join(output_dir, f"checkpoint-{self.global_step}-{self.cfg.TILEIDX}.pt")
        torch.save(check_point, save_path)


        print(f"Successfully export check point to {save_path}\n")

    def toCPU(self):
        """
        All to CPU
        """
        self.featureGrid.toCPU()

        for group in self.featureGrid_optimizer.param_groups:
            for p in group["params"]:
                state = self.featureGrid_optimizer.state[p]
                if len(state) > 0:
                    state["exp_avg"] = state["exp_avg"].cpu()
                    state["exp_avg_sq"] = state["exp_avg_sq"].cpu()
                p.data = p.data.cpu()

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                if len(state) > 0:
                    state["exp_avg"] = state["exp_avg"].cpu()
                    state["exp_avg_sq"] = state["exp_avg_sq"].cpu()
                p.data = p.data.cpu()
        self.crit.toCPU()
        if self.enable_admm:
            self.consensus_manager.toCPU()
            # self.shared_se3 = self.shared_se3.cpu()
            # self.delta_se3 = self.delta_se3.cpu()
            # self.overlap_flags = self.overlap_flags.cpu()
            # self.rho = self.rho.cpu()

        torch.cuda.empty_cache()
        print(f"Tile {self.cfg.TILEIDX} all in CPU")

    def toGPU(self):
        """
        All to GPU
        """
        torch.cuda.empty_cache()
        self.featureGrid.toGPU()

        for group in self.featureGrid_optimizer.param_groups:
            for p in group["params"]:
                state = self.featureGrid_optimizer.state[p]
                if len(state) > 0:
                    state["exp_avg"] = state["exp_avg"].to(self.device)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(self.device)
                p.data = p.data.to(self.device)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state[p]
                if len(state) > 0:
                    state["exp_avg"] = state["exp_avg"].to(self.device)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(self.device)
                p.data = p.data.to(self.device)
        self.crit.toGPU()

        if self.enable_admm:
            self.consensus_manager.toGPU()
            # self.shared_se3 = self.shared_se3.to(self.device)
            # self.delta_se3 = self.delta_se3.to(self.device)
            # self.overlap_flags = self.overlap_flags.to(self.device)
            # self.rho = self.rho.to(self.device)
        print(f"Tile {self.cfg.TILEIDX} all in GPU")

    # for unbounded scene 
    def render_rays(self, rays_o, rays_d, occlusion_mask=None, mode=TRAIN,):
        
        out_dict = {}

        out_dict["rays_o"] = rays_o
        out_dict["rays_d"] = rays_d

        num_fg_sample = self.cfg.TRAINING.NUM_SAMPLE

        out_fore,ret_fg = self.featureGrid.render_fore_rays(rays_o, rays_d, num_fg_sample,
                                                     self.decoder, mode, occlusion_mask=occlusion_mask,
                                                     global_step=self.global_step)
        out_dict["ret_fg"] = ret_fg

        if ret_fg:
            out_dict.update(out_fore)
        else:
            out_dict["fore_valid"] = torch.zeros(rays_d[...,0].shape, dtype=torch.bool, device=self.device)

        out_bg, ret_bg = self.featureGrid.render_bg_rays(rays_o, rays_d, self.cfg.TRAINING.NUM_BG_SAMPLE, 
                                                     self.decoder, mode, occlusion_mask=occlusion_mask,
                                                     global_step=self.global_step,
                                                     bg_mode = self.cfg.TRAINING.BG_MODE,
                                                     infinity=True, 
                                                     fmesh=self.fmesh,
                                                     invalid_underground=self.cfg.INVALID_UNDERGROUND)

        if ret_fg == False and ret_bg == False:
            return None, False
        out_dict["ret_bg"] = ret_bg

        if ret_bg == True:
            out_dict["bg_valid"] = out_bg["valid"]

            if ret_fg:
                out_dict["pred_color"] += out_fore["T_left"] * out_bg["rgb"]
                out_dict["pred_depth"] += out_fore["T_left"] * out_bg["depth"]
                out_dict["pred_specular"] = out_fore["specular"] + out_fore["T_left"] * out_bg["specular"]
                out_dict["pred_diffuse"] = out_fore["diffuse"] + out_fore["T_left"] * out_bg["diffuse"]
                if mode == TRAIN:
                    out_dict["l2_reg_specular"] += out_bg["l2_reg_specular"]
            else:
                out_dict["pred_color"] = out_bg["rgb"]
                out_dict["pred_depth"] = out_bg["depth"]
                out_dict["pred_specular"] = out_bg["specular"]
                out_dict["pred_diffuse"] = out_bg["diffuse"]
                if mode == TRAIN:
                    out_dict["l2_reg_specular"] = out_bg["l2_reg_specular"]
        else:
            out_dict["pred_specular"] = out_fore["specular"]
            out_dict["pred_diffuse"] = out_fore["diffuse"]


        return out_dict, True 


    def render_normals(self, rays_o, rays_d):
        with torch.no_grad():

            z_vals, dists = self.featureGrid.samplePoints(rays_o, rays_d, self.cfg.TRAINING.NUM_SAMPLE)
            fore_valid = torch.any(z_vals != -1, dim = -1) 

            rays_o = rays_o[fore_valid]
            rays_d = rays_d[fore_valid]
            z_vals = z_vals[fore_valid]
            dists = dists[fore_valid]
            if z_vals.shape[0] == 0:
                return None 
            samples = rays_o[:, None, :] + z_vals[..., None] * rays_d[:,None,:]
        normals, sigma = self.featureGrid.compute_normal(samples, self.decoder)
        with torch.no_grad():
            weights, _ = self.featureGrid.cal_integrate_weight(sigma, z_vals, dists, rays_d, infinity=False)
            pred_normal = self.featureGrid.accumulate(weights, normals)
        return (pred_normal + 1) / 2.

    def render_depth_rays(self, rays_o, rays_d, batch_size=2**14):
        pred_depth = torch.zeros_like(rays_o[...,:1])
        for i in range(0, rays_o.shape[0], batch_size):
            with torch.no_grad():
                out, ret = self.render_rays(rays_o[i:i+batch_size], rays_d[i:i+batch_size], None, INFERENCE)
            if out == None:
                continue 
            pred_depth[i:i+batch_size] = out['pred_depth']
        return pred_depth
        
    def render_image_rays(self, rays_o, rays_d, occlusion_mask=None, batch_size=2**14):

        pred_color = torch.zeros_like(rays_o)
        pred_depth = torch.zeros_like(rays_o[...,:1])
        fore_depth = torch.zeros_like(rays_o[...,:1])
        pred_diffuse = torch.zeros_like(rays_o)
        pred_specular = torch.zeros_like(rays_o)
        pred_fore = torch.zeros_like(rays_o)
        pred_normal = torch.zeros_like(rays_o)
        pred_tint = torch.zeros_like(rays_o)

        for i in range(0, rays_o.shape[0], batch_size):
            with torch.no_grad():
                out, ret = self.render_rays(rays_o[i:i+batch_size], rays_d[i:i+batch_size], occlusion_mask[i:i+batch_size], INFERENCE)
            if out == None:
                continue 
            # normal = self.render_normals(rays_o[i:i+batch_size], rays_d[i:i+batch_size])
            fore_valid = out['fore_valid']
            # bg_valid = out['bg_valid']

            pred_color[i:i+batch_size] = out['pred_color']
            pred_depth[i:i+batch_size] = out['pred_depth']
            if out["ret_fg"]:
                fore_depth[i:i+batch_size][fore_valid] =  out["depth"]
                pred_diffuse[i:i+batch_size] = out['diffuse']
                pred_specular[i:i+batch_size] = out['specular']
                pred_fore[i:i+batch_size][fore_valid] = out['rgb']
                # pred_normal[i:i+batch_size][fore_valid] = normal
                pred_tint[i:i+batch_size][fore_valid] = out["tint"]
        
        return {"rgb": pred_color, "fore_depth":fore_depth, "depth": pred_depth,  "diffuse": pred_diffuse, 
                "specular":pred_specular, "fore": pred_fore, "normal": pred_normal,
                "tint":pred_tint}


    def train(self, iterations):
        
        self.batch_size = 2**self.cfg.TRAINING.BS_LOG2DIM


        for _ in tqdm(range(iterations)):

            # if self.commit_shared_depth:
            #     self.update_occlusion_mask()
            #     self.commit_shared_depth = False 

            self.train_one_step()

            # [FIXME] Only for Mega Data (OOM case in warping loss)
            # if self.global_step >= self.cfg.TRAINING.LOSS.WARP_LOSS_START:
            #     self.cfg.TRAINING.NUM_SAMPLE = min(256, self.cfg.TRAINING.NUM_SAMPLE)
            
            # if self.global_step % 10000 == 0:
            # # if self.global_step == 10000 or self.global_step == 20000:
            #     self.export_tile()
                # self.export_check_point(self.logdir)

            if self.global_step % 100 == 0:
                info = f"============ TILE {self.cfg.TILEIDX}\tGPU {self.cfg.GPUIDX} ============\n"
                info += f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}"
                info += f"STEP: {self.global_step}/{self.cfg.TRAINING.TOTAL_STEP}\n"
                info += self.sche.getInfo() + "\n"
                info += self.crit.getInfo()

                # R_error, t_error = self.poses.evaluate()
                # info += f"\nR_error: {R_error}\tt_error: {t_error}\n"
                print(info)
                with open(os.path.join(self.logdir, "training.log"), "a") as f:
                    f.write(info)
                # print(self.consensus_manager.confidence)
                # temp = torch.tensor(self.cfg.VISIBLE_POSES)
                # print(temp[torch.argsort(self.consensus_manager.confidence,descending=True)])

            if self.cfg.UPDATE_MASK_STEP != -1 and self.global_step % self.cfg.UPDATE_MASK_STEP == 0:
                self.render_shared_depth()

            if self.global_step == 1 or self.global_step % 200 == 0:
                cam_files = os.path.join(self.logdir, "cams")
                if os.path.exists(cam_files) is False:
                    os.mkdir(cam_files)
                with torch.no_grad():
                    c2ws = self.poses.get_poses().detach().cpu().numpy()
                np.savez(os.path.join(cam_files, f"cams-{self.global_step}.npz"), 
                    c2ws=c2ws, ks=self.poses.ks.detach().cpu().numpy(), idxs=np.array(self.cfg.VISIBLE_POSES))
                utils.write_camera(os.path.join(cam_files, f"cams-{self.global_step}.txt"), self.cfg.VISIBLE_POSES, c2ws)

            if self.global_step % 1000 == 0:
                torch.cuda.empty_cache()
                count = 0

                for view_idx in self.inference_idx:
                    rays_o, rays_d = self.poses.getRays(self.H, self.W, ray_idx=None, view_idx=[view_idx])
                    rays_o = rays_o.reshape(-1,3).detach()
                    rays_d = rays_d.reshape(-1,3).detach()


                    out = self.render_image_rays(rays_o, rays_d, self.train_data.occlusions[view_idx].reshape(-1,1).to(self.device))
                    target = self.train_data.images[view_idx].reshape(self.H, self.W, 3).numpy()
                    pred_color = utils.get_image_v2(out['rgb'], self.H, self.W)
                    pred_fore = utils.get_image_v2(out['fore'], self.H, self.W)
                    pred_diffuse = utils.get_image_v2(out['diffuse'], self.H, self.W)
                    pred_specular = utils.get_image_v2(out['specular'], self.H, self.W)
                    fore_depth = utils.get_image_v2(out['fore_depth'].repeat(1,3), self.H, self.W) 
                    fore_depth = fore_depth / fore_depth.max()
                    depth = utils.get_image_v2(out['depth'].repeat(1,3), self.H, self.W) 
                    depth = depth / depth.max()
                    pred_normal = utils.get_image_v2(out['normal'], self.H, self.W)[...,::-1]
                    pred_tint = utils.get_image_v2(out['tint'], self.H, self.W)

                    psnr, ssim = utils.Metric()(pred_color, target)
                    frame = np.concatenate([np.concatenate([pred_color, pred_diffuse, pred_specular], 1),
                                            np.concatenate([depth, fore_depth, target], 1)], 0)

                    cv2.imwrite(f"{self.logdir}/{count}-{self.global_step}-{psnr:.2f}-{ssim:.3f}.png", frame * 255)
                    count += 1
                
                if len(self.cfg.NOVEL_IDX) > 0:
                    count = 0
                    for k,c2w in zip(self.novel["ks"], self.novel["c2ws"]):
                        rays_o, rays_d = utils.get_rays_torch_v2(self.novel.H, self.novel.W, k, c2w)
                        rays_o = rays_o.reshape(-1,3).to(self.device)
                        rays_d = rays_d.reshape(-1,3).to(self.device)

                        out = self.render_image_rays(rays_o, rays_d)
                        pred_color = utils.get_image_v2(out['rgb'], self.novel.H, self.novel.W)
                        pred_fore = utils.get_image_v2(out['fore'], self.novel.H, self.novel.W)
                        pred_diffuse = utils.get_image_v2(out['diffuse'], self.novel.H, self.novel.W)
                        pred_specular = utils.get_image_v2(out['specular'], self.novel.H, self.novel.W)
                        fore_depth = utils.get_image_v2(out['fore_depth'].repeat(1,3),  self.novel.H, self.novel.W) 
                        fore_depth = fore_depth / fore_depth.max()
                        depth = utils.get_image_v2(out['depth'].repeat(1,3), self.novel.H, self.novel.W) 
                        depth = depth / depth.max()
                        pred_normal = utils.get_image_v2(out['normal'], self.novel.H, self.novel.W)[...,::-1]
                        pred_tint = utils.get_image_v2(out['tint'], self.novel.H, self.novel.W)

                        frame = np.concatenate([np.concatenate([pred_fore, pred_diffuse, pred_specular], 1),
                                                np.concatenate([pred_normal, fore_depth, pred_color], 1)], 0)
                        cv2.imwrite(f"{self.logdir}/novel-{count}-{self.global_step}.png", frame * 255)
                        count += 1

                torch.cuda.empty_cache()

            if self.global_step >= self.dynamic_start and self.global_step <= self.dynamic_end and self.global_step % self.dynamic_step == 0:
                
                torch.cuda.empty_cache()
                loc_idx = min(self.global_step // self.cfg.TRAINING.ADJUST_STEP, len(self.cfg.TRAINING.GRID_LOG2DIM)-1)
                log2dim = self.cfg.TRAINING.GRID_LOG2DIM[loc_idx]
                loc_idx = min(self.global_step // self.cfg.TRAINING.ADJUST_STEP, len(self.cfg.TRAINING.PRUNING_TH)-1)
                self.prune_th = self.cfg.TRAINING.PRUNING_TH[loc_idx]
                self.featureGrid.pruning_grid(self.global_step, self.decoder, log2dim, self.prune_th)
                # if self.global_step % 30000 == 0:
                #     self.featureGrid.vis_gird(os.path.join(self.logdir))
                print("finished purning grids")
                torch.cuda.empty_cache()


    def train_one_step(self):

        # num_sampled_rays = self.batch_size//self.num_camera
        # patch_size = 2 # FIXED 
        # num_patch = num_sampled_rays // (patch_size**2)

        # num_sampled_rays = num_patch * (patch_size**2)

        # # num_camera x num_patch_per_view  random select left-up pixel of patch 
        # sidx = (torch.rand(self.num_camera,num_patch) * self.num_valid_pixel_per_view[:, None]).long()
        # ori_pidx = self.sample_pidxs[torch.arange(self.num_camera,dtype=torch.long)[:,None].repeat(1,num_patch).reshape(-1), sidx.reshape(-1)]
        # # num_camera x num_patch_per_view x patch_size 
        # ray_idx = torch.stack([ori_pidx, ori_pidx+1, ori_pidx+self.W, ori_pidx+self.W+1], -1).reshape(self.num_camera, -1)
        # ray_idx = ray_idx.to(self.device).long()

        # # N x B 
        # loc_x = ray_idx % self.W 
        # # N x B 
        # loc_y = (ray_idx / self.W).int()
        # # N x B x 2 
        # pixel_locs = torch.stack([loc_x, loc_y], dim=-1)

        num_sampled_rays = self.batch_size//self.num_camera
        patch_size = 2 # FIXED 
        num_patch = num_sampled_rays // (patch_size**2)

        num_sampled_rays = num_patch * (patch_size**2)
        patch_x = torch.randperm(self.W-patch_size, device=self.device)[:num_patch]
        patch_y = torch.randperm(self.H-patch_size, device=self.device)[:num_patch]
        patch_idx = patch_y * self.W + patch_x
        ray_idx = utils.get_ray_idx(patch_idx, patch_size, self.H, self.W)

        loc_x = ray_idx % self.W 
        loc_y = (ray_idx / self.W).int()
        # N x B x 2 
        pixel_locs = torch.stack([loc_x, loc_y], dim=-1)[None,...].repeat(self.num_camera,1,1)


        # ray_idx = (torch.arange(self.num_camera, dtype=torch.long, device=self.device)[:,None].repeat(1,ray_idx.shape[1]).reshape(-1),ray_idx.reshape(-1))

        # if self.global_step <= self.cfg.TRAINING.LOSS.WARP_LOSS_START:
        #     rays_o, rays_d = self.poses.getRays(self.H, self.W, ray_idx)
        #     rays_o = rays_o.reshape(-1,3)
        #     rays_d = rays_d.reshape(-1,3)
        #     gt_color = self.train_data.images.reshape(-1, self.H*self.W,3)[:, ray_idx].reshape(-1,3).to(self.device)
        #     occlusion_mask = self.train_data.occlusions.reshape(-1, self.H*self.W,1)[:, ray_idx].reshape(-1,1)
        # else:
        #     print(self.cfg.VISIBLE_POSES)
        #     rays_o, rays_d = self.poses.getRays(self.H, self.W, view_idx=[0])
        #     rays_o = rays_o.reshape(self.H, self.W, 3)[::4,::4]
        #     rays_d = rays_d.reshape(self.H, self.W, 3)[::4,::4]
        #     rays_o = rays_o.reshape(-1,3)
        #     rays_d = rays_d.reshape(-1,3)
        #     gt_color = self.train_data.images[0,::4,::4].reshape(-1,3).to(self.device)
        #     occlusion_mask = self.train_data.occlusions[0,::4,::4].reshape(-1,1)

        rays_o, rays_d = self.poses.getRays(self.H, self.W, ray_idx)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        gt_color = self.train_data.images.reshape(-1, self.H*self.W,3)[:, ray_idx].reshape(-1,3).to(self.device)
        occlusion_mask = self.train_data.occlusions.reshape(-1, self.H*self.W,1)[:, ray_idx].reshape(-1,1).to(self.device)
        # print(rays_o.shape)
        # print(rays_d.shape)
        # print(gt_color.shape)
        # print(occlusion_mask.shape)

        self.cfg.global_step = self.global_step

        # if self.global_step <= self.cfg.TRAINING.LOSS.WARP_LOSS_START:
        #     out, ret = self.render_rays(rays_o, rays_d, occlusion_mask, TRAIN)
        # else:
        #     with torch.no_grad():
        #         out, ret = self.render_rays(rays_o, rays_d, occlusion_mask, INFERENCE)

        out, ret = self.render_rays(rays_o, rays_d, occlusion_mask, TRAIN)

        if out == None:
            self.global_step += 1
            return 

        if self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS > 0:
            # N x B x 1 
            monocular_depth = self.train_data.mono_depths.reshape(-1, self.H*self.W,1)[:,ray_idx].reshape(self.num_camera,-1,1).to(self.device)
            # # N x B x 1
            # rendered_depth = torch.zeros_like(monocular_depth)
            # rendered_depth.reshape(-1,1)[out["fore_valid"]] = out["depth"]
            rendered_depth = out["pred_depth"].reshape(monocular_depth.shape)
            out["rendered_depth"] = rendered_depth
            out["monocular_depth"] = monocular_depth
            valid_mask = torch.zeros_like(monocular_depth).bool()


        if self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0:
            # N x B x 3 
            monocular_normal = self.train_data.mono_normals.reshape(-1, self.H*self.W,3)[:,ray_idx].reshape(self.num_camera,-1,3).to(self.device)

            rendered_depth = out["pred_depth"].reshape(monocular_normal[...,:1].shape)
            out["rendered_depth"] = rendered_depth
            out["monocular_normal"] = monocular_normal
            out["pixel_locs"] = pixel_locs
            valid_mask = torch.zeros_like(monocular_normal[...,:1]).bool()

        if self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS > 0 or self.cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0:
            # N x B x 1 
            valid_mask.reshape(-1,1)[out["fore_valid"] | out["bg_valid"]] = 1.
            out["valid_mask"] = valid_mask

        # update confidence 
        # self.update_confidence(out['pred_color'], gt_color, out["fore_valid"])

        out.update({"input": out['pred_color'], 
                    "target": gt_color, 
                    # "ref_grid": ref_grid,
                    # "init_depth": init_depth,
                    "ori_poses_idxs": self.cfg.VISIBLE_POSES,
                    "occlusions": self.train_data.occlusions,
                    "global_step":self.global_step})
        loss = self.crit(**out)

        loss = loss + 0.01 * out['l2_reg_specular'] 


        self.featureGrid_optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        loss.backward()

        # print(torch.abs(self.poses.se3_refine.grad).mean())


        self.featureGrid_optimizer.step()
        self.optimizer.step()
        self.featureGrid_sche.step(self.global_step, self.featureGrid_optimizer)
        self.sche.step(self.global_step, self.optimizer)

        self.global_step += 1




