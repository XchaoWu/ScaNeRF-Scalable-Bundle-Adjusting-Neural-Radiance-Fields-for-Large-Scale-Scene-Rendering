import torch 
import numpy as np 
import os,sys,cv2,time,random
import multiprocessing 
from multiprocessing import Process, Queue, Pool, Lock, Value, Manager
from load_data import load_snisr, read_campara
from fastMesh import FastMesh
from datetime import datetime
from tile import TILE 
import camera
from easydict import EasyDict as edict
import os,sys,cv2 
from tools import utils
from tools import tools
import ctypes 
from cfg import * 

class ADMM_TRAINER:
    def __init__(self, cfg):

        self.cfg = cfg 
        
        self.cfg.LOGDIR = os.path.join(self.cfg.DATADIR, "logs")
        self.cfg.MESH = os.path.join(self.cfg.DATADIR, "mesh/mesh.ply")

        self.shared_info = edict({"shared_mem": Manager().list([None for _ in range(np.max(cfg.TILES) + 1)]),
                                  "shared_lock": Lock(), "shared_count": Value(ctypes.c_int, -1),
                                  "shared_info": Manager().dict(),
                                  "shared_depth": Manager().list()})

        self.syn_itrs = self.cfg.SYN_ITERS

        # number of process 
        self.num_process = len(self.cfg.GPU)
        # gpu list 
        self.gpu_list = self.cfg.GPU
        # decide tile to training 
        self.tileIdx_list = self.cfg.TILES

        self.test_idxs = dict()
        for bidx in self.tileIdx_list:
            self.test_idxs[f"{bidx}"] = []
        try:
            f = open(os.path.join(cfg.DATADIR, "blocks", "test.txt"), "r")
        except:
            pass
        else:
            lines = f.readlines()
            f.close()
            for line in lines:
                line = line.strip().split(" ")
                self.test_idxs[line[0]] = list(map(lambda x:int(x), line[1:]))
        
        self.novel_idxs = dict()
        for bidx in self.tileIdx_list:
            self.novel_idxs[f"{bidx}"] = []
        try:
            f = open(os.path.join(cfg.DATADIR, "blocks", "novel.txt"), "r")
        except:
            pass
        else:
            lines = f.readlines()
            f.close()
            for line in lines:
                line = line.strip().split(" ")
                self.novel_idxs[line[0]] = list(map(lambda x:int(x), line[1:]))

        self.num_tile = len(self.tileIdx_list)

        self.allocate_tiles()

        
    
    def allocate_tiles(self):

        self.infos = [{"tidxs": list(), "gpuidx": -1, "test": list(), "novel": list()} for _ in range(self.num_process)]
        index = 0
        for tileIdx in self.tileIdx_list:
            self.infos[index]["tidxs"].append(tileIdx)
            self.infos[index]["test"].append(self.test_idxs[f"{tileIdx}"])
            self.infos[index]["gpuidx"] = self.gpu_list[index]
            self.infos[index]["novel"].append(self.novel_idxs[f"{tileIdx}"])
            index = (index + 1) % self.num_process

    def master_process(self):
        """
        preprocess before training 
        """
        ks, c2ws, H, W = read_campara(os.path.join(self.cfg.DATADIR, "camera.log"), True)  
        num_camera = c2ws.shape[0]
        ks = torch.from_numpy(ks)
        c2ws = torch.from_numpy(c2ws)
        ori_rts = camera.pose.invert(c2ws)
        runtime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        if self.cfg.PREFIX != "":
            root_dir = os.path.join(self.cfg.LOGDIR, f"{self.cfg.PREFIX}-{runtime}")
        else:
            root_dir = os.path.join(self.cfg.LOGDIR, f"{runtime}")
            
        if os.path.exists(root_dir):
            os.system(f"rm -rf {root_dir}")
        os.mkdir(root_dir)
        print(f"set root log dir to {root_dir}")
        self.shared_info.shared_info["root_dir"] = root_dir
        os.system(f"cp {self.cfg.yaml} {root_dir}")
        print(f"copy the yaml file to {root_dir}")

        cam_weight = torch.ones((np.max(self.tileIdx_list)+1, num_camera)).float() # avg
        shared_poses = torch.zeros((num_camera, 6), dtype=torch.float32)

        noise_camera = int(num_camera * 1.0)
        noise_idxs = torch.randperm(num_camera)[:noise_camera]
        self.shared_info.shared_info["noise"] = torch.zeros((num_camera, 6), dtype=torch.float32)
        self.shared_info.shared_info["noise"][noise_idxs] = self.cfg.TRAINING.CAMOPT.NOISE * torch.randn((noise_camera, 6), dtype=torch.float32)

        for i in range(num_camera):
            self.shared_info.shared_depth.append(None)
            # self.shared_info.shared_depth.append(torch.ones((H,W,1),dtype=torch.float32)*1e8)

        # allow other processes to start 
        self.shared_info.shared_count.value = 0

        while self.shared_info.shared_count.value != -1:
            if self.shared_info.shared_count.value == self.num_tile:
                

                # debug 
                # for idx,item in enumerate(self.shared_info.shared_depth):
                #     if item != None:
                #         item[item==1e8] = 0
                #         item = item / 100 * 255
                #         item = item.numpy()
                #         cv2.imwrite(os.path.join(root_dir, f"{idx}.png"), item)


                temp_shared_poses = torch.zeros((num_camera, 6), dtype=torch.float32)
                overlap_count = torch.zeros((num_camera), dtype=torch.int32)
                accumulate_weight = torch.zeros((num_camera), dtype=torch.float32)


                for tidx in self.tileIdx_list:
                    out = self.shared_info.shared_mem[tidx]
                    pose_idxs = out['idx']
                    tile_poses = out['pose']
                    confidence = out["confidence"]


                    overlap_count[pose_idxs] += 1
                    accumulate_weight[pose_idxs] += confidence
                    temp_shared_poses[pose_idxs] += (confidence[...,None] * tile_poses)

                overlap_pose_idxs = torch.where(overlap_count >= 2)[0]
                accumulate_weight[accumulate_weight == 0] = 1
                temp_shared_poses /= accumulate_weight[...,None]

                dual_residual = torch.mean(torch.abs(shared_poses - temp_shared_poses))
                shared_poses = temp_shared_poses

                # compute variance 
                primal_residual = 0
                for tidx in self.tileIdx_list: 
                    out = self.shared_info.shared_mem[tidx]
                    pose_idxs = out['idx']
                    tile_poses = out['pose']
                    # M x 6
                    primal_residual += torch.mean(torch.abs(tile_poses - shared_poses[pose_idxs]))
                primal_residual = primal_residual / len(self.tileIdx_list)
                with open(os.path.join(root_dir, "admm_error.txt"),"a") as f:
                    f.write(f"primal_residual: {primal_residual:.8f}\tdual_residual: {dual_residual:.8f}\n")

                # broadcaset to all process 
                for tidx in self.tileIdx_list:
                    out = self.shared_info.shared_mem[tidx]
                    pose_idxs = out['idx']
                    overlap_idxs = torch.tensor([pose_idxs.index(idx) for idx in pose_idxs if idx in overlap_pose_idxs])
                    self.shared_info.shared_mem[tidx] = {"shared_poses": shared_poses[pose_idxs],
                                                          "overlap_idxs": overlap_idxs}
                self.shared_info.shared_count.value = 0  

        # finished training here 
        refined_rts = camera.pose.compose([camera.lie.se3_to_SE3(shared_poses), ori_rts])
        refined_c2ws = camera.pose.invert(refined_rts)
        tools.write_campara(os.path.join(root_dir, "refined_camera.log"), ks, refined_c2ws, H, W)


    def admm_training_process(self, groupIdx):

        # fix random seed 
        np.random.seed(self.cfg.SEED)
        torch.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed_all(self.cfg.SEED)
        random.seed(self.cfg.SEED)
        print(f"fixed random seed to {self.cfg.SEED}")

        gpuIdx = self.infos[groupIdx]["gpuidx"]
        tileIdxs = self.infos[groupIdx]["tidxs"]
        testIdxs = self.infos[groupIdx]["test"]
        novelIdxs = self.infos[groupIdx]["novel"]
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuIdx}"
        device = torch.device("cuda:0")

        if self.cfg.MESH != "":
            fmesh = FastMesh(self.cfg.MESH)
        else:
            fmesh = None 

        tiles = []
        for tileIdx,tidx,nidx in zip(tileIdxs,testIdxs, novelIdxs):
            tiles += [TILE(self.cfg, tileIdx, gpuIdx, tidx, nidx, fmesh, device, True)]
        
        if len(tiles) == 1:
            exchange = False
        else:
            exchange = True 
    
        for t in tiles:
            t.set(self.shared_info)
            t.build_training_context()
            t.commit()
            if exchange:
                t.toCPU()
        # print("finished building training context for all tiles")

        # wait the master process to finish its job 
        while self.shared_info.shared_count.value != 0:
            continue 

        for t in tiles:
            t.synchronize()

        syn_steps = [self.cfg.SYN_START, self.syn_itrs]
        syn_steps = [item for item in syn_steps if item > 0]

        left_iterations = self.cfg.TRAINING.TOTAL_STEP
        index = 0
        while left_iterations > 0:
            training_itrs = syn_steps[index]

            for t in tiles:
                if exchange:
                    t.toGPU()
                if t.commit_shared_depth:
                    t.update_occlusion_mask()
                    t.commit_shared_depth = False
                t.train(training_itrs)
                if exchange:
                    t.toCPU()
                t.commit()
            
            # wait the master process to finish its job 
            while self.shared_info.shared_count.value != 0:
                continue 
            
            # self.shared_info.shared_count.value = 0 
            # broadcast to training process 
            for t in tiles:
                t.synchronize()

            left_iterations = left_iterations - training_itrs
            index = min(index + 1, len(syn_steps) - 1)
            
        # store to disk 
        for t in tiles:
            if exchange:
                t.toGPU()
            t.export_tile()
            if exchange:
                t.toCPU()
    
    def independent_training_process(self, groupIdx):

        # fix random seed 
        np.random.seed(self.cfg.SEED)
        torch.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed_all(self.cfg.SEED)
        random.seed(self.cfg.SEED)
        print(f"fixed random seed to {self.cfg.SEED}")

        gpuIdx = self.infos[groupIdx]["gpuidx"]
        tileIdxs = self.infos[groupIdx]["bidxs"]
        testIdxs = self.infos[groupIdx]["test"]
        novelIdxs = self.infos[groupIdx]["novel"]
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuIdx}"
        device = torch.device("cuda:0")

        if self.cfg.MESH != "":
            fmesh = FastMesh(self.cfg.MESH)
        else:
            fmesh = None 

        tiles = []
        for bidx,tidx,nidx in zip(tileIdxs,testIdxs, novelIdxs):
            tiles += [TILE(self.cfg, bidx, gpuIdx, tidx, nidx, fmesh, device, False)]
        
        for b in tiles:
            b.build_training_context()
            b.set(self.shared_info)
            b.train(self.cfg.TRAINING.TOTAL_STEP)
            b.commit()
            b.export_tile()
            b.toCPU()

    def training_process(self, groupIdx, enable_admm):
        if enable_admm:
            self.admm_training_process(groupIdx)
        else:
            self.independent_training_process(groupIdx)

    def run(self, enable_admm=True, groupIdx=None):

        # if enable_admm:
        main_process = Process(target=self.master_process, args=())
        main_process.start()
        
        while self.shared_info.shared_count.value != 0:
            continue 
        self.cfg.LOGDIR = self.shared_info.shared_info["root_dir"]
        self.cfg.NOISE = self.shared_info.shared_info["noise"]
        print("start distributed training ... ")

        if groupIdx != None:
            self.training_process(groupIdx, enable_admm)
        else:
            process_list = []
            for groupIdx in range(len(self.infos)):
                process_list += [Process(target=self.training_process, args=(groupIdx, enable_admm))]
            for p in process_list:
                p.start()
            for p in process_list:
                p.join()
        
        self.shared_info.shared_count.value = -1

        print("finished training")

if __name__ == "__main__":

    cfg = utils.parse_yaml(sys.argv[1])

    # fix random seed 
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    random.seed(cfg.SEED)
    print(f"fixed random seed to {cfg.SEED}")

    admm = ADMM_TRAINER(cfg)
    admm.run(cfg.RHO > 0)