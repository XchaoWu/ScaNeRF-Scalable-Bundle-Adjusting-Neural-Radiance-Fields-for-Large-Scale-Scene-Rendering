import torch 
import torch.nn as nn 
import numpy as np 
import network
import sys,os,cv2
from glob import glob 
from time import time 
from tqdm import tqdm 
from tools import tools
from tools import utils
from load_data import load_snisr, read_campara
from hashgrid import (
    ray_block_intersection,
    sample_points,
    prepare_points,
    sort_by_key,
    pts_inference,
    accumulate_color,
    ray_firsthit_block,
    inverse_z_sampling,
    bg_pts_inference,
    get_last_block,
    update_outgoing_bidx,
    update_outgoing_bidx_v2,
    bg_pts_inference_v2,
    process_occupied_grid)
from tools.ssim import SSIM 


class RenderingHashGrid:
    def __init__(self, datadir, demo_name, gpuIdx, mode):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuIdx}"
        self.device = torch.device("cuda:0")

        assert mode in ["val", "inference"]

        self.sample_step = 0.0125
        self.demo_name = demo_name
        self.datadir = datadir
        self.demo_dir = os.path.join(cfg.DATADIR, "demo", f"{demo_name}")
        self.mode = mode 

        self.load_camera()
        self.parse()
        self.parse_blocks()

    def cal_psnr(self,I1,I2):
        mse = torch.mean((I1-I2)**2)
        if mse < 1e-10:
            return 100
        return 10 * float(torch.log10(255.0**2/mse))

    def load_camera(self):
        if self.mode == "val":
            ks, c2ws, H, W  = read_campara(os.path.join(self.demo_dir, "refined_camera.log"), True)
            with open(os.path.join(cfg.DATADIR, "val_new.txt"), "r") as f:
                lines = f.readlines()
                val_idx = [int(line.strip()) for line in lines]
            ks = ks[val_idx]
            c2ws = c2ws[val_idx]
            self.val_idx = val_idx

            self.ssim_cal = SSIM(window_size=11).to(self.device)
            self.gt_path = os.path.join(self.datadir, "images")
        else:
            ks, c2ws, H, W = read_campara(os.path.join(cfg.DATADIR, "renderPath.log"), True)
            H = 720
            W = 1280 
            ks[:, 0, 2] = 640
            ks[:, 1, 2] = 360
        self.H = H 
        self.W = W 
        self.ks = torch.from_numpy(ks).to(self.device) 
        self.c2ws = torch.from_numpy(c2ws).to(self.device) 

    def parse(self):
        
        # STEP2 search trained blocks 
        extract_idx = lambda x: int(os.path.basename(x).split("-")[-1])
        files = glob(os.path.join(self.datadir, "demo", f"{self.demo_name}", "tile-*"))
        files = [item for item in files if os.path.isdir(item)]
        files.sort(key=extract_idx)
        self.trained_blockDir = files
        self.trained_blockIdx = [extract_idx(item) for item in files]
        print(f"find trained tile index:\n{self.trained_blockIdx}")

    def parse_single_block(self, block_dir):
        decoder_path = os.path.join(block_dir, "decoder.pth")
        hash_path = os.path.join(block_dir, "feature.npz")
        self.decoder_dirs += [decoder_path]
        self.hash_dirs += [hash_path]

    def parse_blocks(self):
        self.decoder_dirs = []
        self.hash_dirs = []

        for block_dir in self.trained_blockDir:
            self.parse_single_block(block_dir)

        # network 
        self.params = []
        for path in self.decoder_dirs:
            weights, bias = utils.extract_MLP_para(path)
            temp_params = []
            for w,b in zip(weights, bias):
                # print(b.shape, w.shape)
                temp_params += [b, w.transpose(1,0).flatten()]
                # temp_params += [torch.cat([b[...,None], w.transpose(1,0)], -1).reshape(-1,)]
            temp_params = torch.cat(temp_params, 0)
            self.params += [temp_params]
        # self.params = torch.stack(self.params, 0).half()
        self.params = torch.stack(self.params, 0)
        print("params shape", self.params.shape)

        num_blocks = len(self.hash_dirs)
        """
        block_corner 
        grid_occupied, grid_size, grid_log2dim 
        features, resolution, 
        """
        for idx,path in enumerate(tqdm(self.hash_dirs)):
            npz_file = np.load(path)
            features = npz_file["features"] 
            grid = npz_file["occupied_grid"]

            # grid[:,-int(11 / 38. * grid.shape[1]):,:] = False

            resolution = npz_file["resolution"]
            if idx == 0:            
                self.feature_tables = np.zeros((num_blocks, *list(features.shape)), dtype=np.float16)
                # self.occupied_grid = np.zeros((num_blocks, *list(grid.shape)) ,dtype=bool)
                self.occupied_grid = []
                self.block_corner = np.zeros((num_blocks, 3), dtype=np.float32)
                self.block_size = np.zeros((num_blocks, 3), dtype=np.float32)
                self.grid_log2dim = np.zeros((num_blocks, 3), dtype=np.int32)
                self.resolution = np.zeros((num_blocks, *list(resolution.shape)), dtype=np.int32)
                # self.grid_starts = np.zeros((num_blocks), dtype=np.int64)
                # self.grid_starts[0] = 0
                grid_sizes = np.zeros((num_blocks+1), dtype=np.int64)
            
            self.feature_tables[idx] = features
            self.occupied_grid.append(grid.reshape(-1,))
            self.block_corner[idx] = npz_file["block_corner"]
            self.block_size[idx] = npz_file["block_size"]
            self.grid_log2dim[idx] = npz_file["grid_log2dim"]
            self.resolution[idx] = resolution
            grid_sizes[idx+1] = np.cumprod(grid.shape)[-1]
            
        self.block_corner = torch.from_numpy(self.block_corner).to(self.device)
        self.block_size = torch.from_numpy(self.block_size).to(self.device)
        self.feature_tables = torch.from_numpy(self.feature_tables).to(self.device)
        self.occupied_grid = torch.from_numpy(np.concatenate(self.occupied_grid)).to(self.device)
        self.grid_starts = np.cumsum(grid_sizes)[:-1]
        self.grid_starts = torch.from_numpy(self.grid_starts).to(self.device)
        self.grid_log2dim = torch.from_numpy(self.grid_log2dim).to(self.device)
        self.resolution = torch.from_numpy(self.resolution).to(self.device)
        self.params = self.params.to(self.device)

        scene_min_corner = torch.min(self.block_corner, dim=0)[0]
        scene_max_corner = torch.max(self.block_corner + self.block_size, dim=0)[0]
        self.scene_size = scene_max_corner - scene_min_corner
        self.scene_center = (scene_max_corner + scene_min_corner) / 2. 

        self.block_corner = self.block_corner + self.block_size / 4. 
        self.block_size = self.block_size / 2.
        print(self.grid_log2dim.shape)
        print("process grid")
        self.fake_occupied_grid = self.occupied_grid.clone()
        for i in range(self.block_corner.shape[0]):
            process_occupied_grid(i, int(torch.prod(2 ** self.grid_log2dim[i]).cpu()),
                                  self.block_corner, self.block_size, 
                                  self.occupied_grid, self.grid_starts, 
                                  self.grid_log2dim, self.fake_occupied_grid)
        print("finished processing grid")


    def vis_occupied_grid(self, idx_list=[]):
        for idx in idx_list:
            log2dim = self.grid_log2dim[idx]
            block_size = self.block_size[idx]
            block_corner = self.block_corner[idx]
            grid_size = block_size / (2**log2dim)

            X,Y,Z = torch.meshgrid(torch.arange(0, 2**log2dim[0], 1, device=self.device),
                                torch.arange(0, 2**log2dim[1], 1, device=self.device),
                                torch.arange(0, 2**log2dim[2], 1, device=self.device))
            centers = torch.stack([X,Y,Z], -1).reshape(-1,3) * grid_size + grid_size / 2.

            grid = self.occupied_grid[self.grid_starts[idx]:self.grid_starts[idx]+centers.shape[0]]
            centers = centers[grid.reshape(-1)] + block_corner
            size = torch.ones_like(centers) * grid_size
            vertices, faces = tools.draw_AABB(centers.cpu().numpy(), size.cpu().numpy())
            tools.mesh2obj(os.path.join(self.demo_dir,f"grid_{idx}.obj"), vertices, faces)

    def rendering(self, startIdx, endIdx):
        
        logdir = os.path.join(self.demo_dir,f"output_{self.mode}")
        if os.path.exists(logdir) is False:
            # os.system(f"rm -rf {logdir}")
            os.mkdir(logdir)

        time_record = []
        if self.mode == "val":
            psnrs = []
            ssims = []
            f = open(os.path.join(self.demo_dir, "metric.txt"), "w")
        

        if endIdx == -1:
            endIdx = self.ks.shape[0]

        for index in tqdm(range(startIdx,endIdx,1)):
            # index = self.ks.shape[0] - index - 1
        # for index in tqdm(range(632,832,1)):
            # if testIdx != -1:
            #     if self.mode == "val":
            #         index = self.val_idx.index(testIdx)
            #     else:
            #         index = testIdx
        # for index in tqdm(range(0,,1)):
            # index = 100
            k = self.ks[index]
            c2w = self.c2ws[index]
            s = time()
            diffuse, specular, depth, transparency = self.render_rays_base(self.H, self.W, k, c2w)
            torch.cuda.synchronize()
            e = time()
            time_record += [e-s]
            # exit()

            final = torch.clamp(diffuse+specular, 0.0, 1.0)
            # final_v2 = torch.cat([final, 1-transparency], -1)

            if self.mode == "val":   
                img_idx = self.val_idx[index]
                gt = cv2.imread(os.path.join(self.gt_path, f"{img_idx}.png"))    
                gt = torch.from_numpy(gt).float()
                psnr = self.cal_psnr(gt, final.detach().cpu().numpy()*255)
                ssim = float(self.ssim_cal(gt[None,...].permute(0,3,1,2).to(self.device)/255,
                              final[None,...].permute(0,3,1,2)).cpu())
                f.write(f"img {img_idx} psnr {psnr:.2f}\tssim {ssim:.3f}\n")
                psnrs.append(psnr)
                ssims.append(ssim)
                cv2.imwrite(os.path.join(logdir, f"{img_idx}.png"), final.detach().cpu().numpy()*255)
                # depth = depth.detach().cpu().numpy()
                # depth = depth / 30 * 255 
                # cv2.imwrite(os.path.join(logdir, f"depth_{img_idx}.png"), depth)
            else:
                cv2.imwrite(os.path.join(logdir, f"{index}.png"), final.detach().cpu().numpy()*255)
                # cv2.imwrite(os.path.join(logdir, f"diffuse_{index}.png"), diffuse.detach().cpu().numpy()*255)
                # cv2.imwrite(os.path.join(logdir, f"specular_{index}.png"), specular.detach().cpu().numpy()*255)
                # cv2.imwrite(os.path.join(logdir, f"{index}_rgba.png"), final_v2.detach().cpu().numpy()*255)
                # depth = depth.detach().cpu().numpy()
                # depth = depth / 20 * 255 
                # cv2.imwrite(os.path.join(logdir, f"depth_{index}.png"), depth)
                # transparency = transparency.detach().cpu().numpy() * 255 
                # cv2.imwrite(os.path.join(logdir, f"T_{index}.png"), transparency)
                # depth = depth.detach().cpu().numpy()
                # depth = depth / 30 * 255 
                # cv2.imwrite(os.path.join(logdir, f"depth_{index}.png"), depth)
            # if testIdx != -1:
            #     exit()

        if self.mode == "val":
            mean_psnr = np.mean(psnrs)
            mean_ssim = np.mean(ssims)
            f.write(f"mean psnr {mean_psnr:.2f}\tmean ssim {mean_ssim:.3f}\n")
            f.close()

        print( "render time ", np.mean(time_record) * 1000 ) 

    def compute_rays(self, H, W, K, c2w):
        """
        K   3 x 3
        c2w 3 x 4 
        return 
        rays_o H x W x 3
        rays_d H x W x 3 
        """
        j,i = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device))
        dirs = torch.stack([(i+0.5-K[0,2])/K[0,0], (j+0.5-K[1,2])/K[1,1], torch.ones_like(i, device=self.device)], -1) 
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].clone()[None,None,].repeat(H,W,1)
        return rays_o, rays_d
    
    def render_rays_base(self, H, W, k, c2w):
        
        s = time()
        rays_o, rays_d = self.compute_rays(H, W, k, c2w)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        # rays_o = rays_o.reshape(H,W,3)[162:163, 891:892].reshape(-1,3)
        # rays_d = rays_d.reshape(H,W,3)[162:163, 891:892].reshape(-1,3)
        torch.cuda.synchronize()
        e = time()
        print(f"compute_rays {(e-s) * 1000:.2f} ms" ) 

        s = time()
        intersections = torch.full((rays_d.shape[0], self.block_corner.shape[0], 2), 1e7, dtype=torch.float32, device=self.device)
        ray_block_intersection(rays_o, rays_d, self.block_corner, self.block_size, intersections)
        torch.cuda.synchronize()
        e = time()
        print(f"ray block intersection {(e-s) * 1000:.2f} ms" ) 

        # print(intersections)
        # exit()
        s = time()
        tracing_blocks = torch.argsort(intersections[...,0], dim=-1).int()
        torch.cuda.synchronize()
        e = time()
        print(f"tracing_blocks {(e-s) * 1000:.2f} ms" ) 
 
        max_tracing_block = int(torch.mean((intersections != 1e7).float(), dim=-1).sum(dim=-1).max().cpu())
        # print(max_tracing_block)
        # exit()
        # s = time()
        # hit_blockIdxs = torch.full((rays_d.shape[0],1), -1, dtype=torch.int16, device=self.device)
        # ray_firsthit_block(rays_o, rays_d, self.block_corner, self.block_size, self.occupied_grid,
        #                 self.grid_starts, self.grid_log2dim, tracing_blocks, intersections,
        #                 hit_blockIdxs)
        # torch.cuda.synchronize()
        # e = time()
        # print(f"ray_firsthit_block {(e-s) * 1000:.2f} ms" ) 

        # temp = hit_blockIdxs.float().cpu().numpy().reshape(H,W,1) / 7. * 255
        # cv2.imwrite("a.png",temp)
        # exit()

        # print(tracing_blocks)
        # print(intersections)

        transparency = torch.ones((rays_d.shape[0], 1), dtype=torch.float32, device=self.device)
        diffuse = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        specular = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        depth = torch.zeros((rays_d.shape[0], 1),dtype=torch.float32, device=self.device)
        tracing_idx = torch.zeros((rays_d.shape[0], 1), dtype=torch.int32, device=self.device)
        z_start = torch.full((rays_d.shape[0], 1), 0, dtype=torch.float32, device=self.device)
        # render_time = 0.0
        # print("max_tracing_block", max_tracing_block)
        s = time()  
        for step in range(max_tracing_block):
            # print(tracing_idx, transparency)
            # print(tracing_idx)
            # runing_mask = transparency > 1e-5
            runing_mask = (tracing_idx < max_tracing_block) & (transparency > 1e-5)
            if runing_mask.sum() == 0:
            #     print(f"break in {tracing_idx}")
                break

            # s = time()  
            num_sample = 128
            z_vals = torch.full((rays_d.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
            dists = torch.full((rays_d.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
            sample_points(rays_o, rays_d, self.block_corner, self.block_size, self.fake_occupied_grid, 
                        self.grid_starts, self.grid_log2dim, tracing_blocks, intersections, 
                        tracing_idx, z_start, z_vals, dists)
            # print(z_vals)
            # torch.cuda.synchronize()
            # e = time()
            # render_time += e-s
            # print(f"sample_points {(e-s) * 1000:.2f} ms" ) 
            # print(z_vals)
            # exit()
            # print(step, max_tracing_block, tracing_idx)
            # samples = rays_o[:,None,:] + z_vals[:,:,None] * rays_d[:,None,:]
            # from tools import tools 
            # tools.points2obj("/home/yons/4TB/sig23/coffee/sample.obj", samples.detach().cpu().numpy().reshape(-1,3))
            # exit()
            # s = time()
            block_idxs = torch.full((rays_d.shape[0],num_sample,4), -1, dtype=torch.int16, device=self.device)
            prepare_points(z_vals, runing_mask, intersections, block_idxs)
            # block_idxs[..., 1:] = -1
            # print(block_idxs)

            # print(block_idxs)
            # torch.cuda.synchronize()
            # e = time()
            # render_time += e-s
            # print(f"prepare_points {(e-s) * 1000:.2f} ms" ) 

            # update_outgoing_bidx(block_idxs, outgoing_bidxs)
            # print(outgoing_bidxs)

            # s = time()
            pts_diffuse = torch.full((rays_d.shape[0],num_sample,3), 0, dtype=torch.float32, device=self.device)
            pts_specular = torch.full((rays_d.shape[0],num_sample,3), 0, dtype=torch.float32, device=self.device)
            pts_alpha = torch.full((rays_d.shape[0],num_sample,1), 0, dtype=torch.float32, device=self.device)
            pts_inference(rays_o, rays_d, z_vals, dists, 
                          block_idxs, self.feature_tables, self.params, self.resolution,
                          self.occupied_grid, self.grid_starts, self.grid_log2dim,
                          self.block_corner, self.block_size, pts_diffuse, pts_specular, pts_alpha)
            
            # B x num_sample
            # temp = (block_idxs != -1).sum(dim=-1).reshape(-1,)
            # temp = temp > 1
            # pts_diffuse.reshape(-1,3)[temp, 1] = 1.0
            # pts_specular.reshape(-1,3)[temp, 1] = 1.0
            # pts_diffuse = pts_diffuse.reshape(-1,num_sample,3)
            # pts_specular = pts_specular.reshape(-1,num_sample,3)

            # torch.cuda.synchronize()
            # e = time()
            # render_time += e-s
            # print(f"pts_inference {(e-s) * 1000:.2f} ms" ) 

            # s = time()
            accumulate_color(pts_diffuse, pts_specular, pts_alpha, transparency, 
                             z_vals, diffuse, specular, depth)

            # torch.cuda.synchronize()
            # e = time()
            # render_time += e-s
            # print(f"accumulate {(e-s) * 1000:.2f} ms" ) 
            
        torch.cuda.synchronize()
        e = time()
        print(f"render_time {(e-s) * 1000:.2f} ms" ) 

        del pts_diffuse, pts_specular, pts_alpha
        torch.cuda.empty_cache()
        
        # del pts_diffuse, pts_specular, pts_alpha
        # torch.cuda.empty_cache()
        # print(diffuse)
        # exit()

        # diffuse = diffuse.reshape(H,W,3)
        # specular = specular.reshape(H,W,3)
        # depth = depth.reshape(H,W,1)
        # transparency = transparency.reshape(H,W,1)
        # return diffuse, specular, depth, transparency
    
        # cv2.imwrite("a.png", transparency.cpu().numpy()*255)
        # exit()
        num_bg_sample = 128
        sample_range = 1e6


        # # ============================= multi-bg ================================
        # exit point 
        bg_bidxs = torch.full((rays_d.shape[0], 4), -1, dtype=torch.int16, device=self.device)
        bg_blend_weights = torch.full((rays_d.shape[0], 4), 0, dtype=torch.float, device=self.device)
        update_outgoing_bidx(rays_o, rays_d, self.block_corner, self.block_size,
                             tracing_blocks, intersections, bg_bidxs, bg_blend_weights, 0.12, False)
        # bg_bidxs[..., 1:] = -1
        # bg_blend_weights[..., 1:] = 0
        # bg_blend_weights[bg_blend_weights!=0] = 1.
        # print(bg_blend_weights)
        # print(bg_bidxs)
        bg_blend_weights = bg_blend_weights / torch.sum(bg_blend_weights, dim=-1, keepdim=True)
        num_bg_blend = int((bg_blend_weights > 0).sum(dim=-1).max().cpu())
        # print("num_bg_blend", num_bg_blend)
        # print(bg_bidxs)
        # print(bg_blend_weights)
        # exit()
        # bg_bidxs[..., 1:] = -1
        # bg_blend_weights[..., :] = 1.
        # print(intersections)
        # print(tracing_blocks)
        # print(bg_bidxs)
        # print(bg_blend_weights)
        # exit()
        # temp = bg_bidxs + 1
        # temp = temp / temp.max() * 255 
        # temp = temp.cpu().numpy()
        # cv2.imwrite("temp.png", temp[...,:3].reshape(self.H, self.W, 3))
        # diffuse = diffuse.reshape(H,W,3)
        # specular = specular.reshape(H,W,3)
        # depth = depth.reshape(H,W,1)
        # temp = torch.clamp(diffuse + specular, 0, 1).cpu().numpy() * 255
        # cv2.imwrite("temp2.png", temp[...,:3].reshape(self.H, self.W, 3))
        # exit()

        bg_diffuse = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        bg_specular = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        bg_depth = torch.zeros((rays_d.shape[0], 1),dtype=torch.float32, device=self.device)
        for i in range(num_bg_blend):
            s = time()
            bg_z_vals = torch.full((rays_d.shape[0], num_bg_sample), -1, dtype=torch.float32, device=self.device)
            inverse_z_sampling(intersections, bg_bidxs[...,i].contiguous(), bg_z_vals, sample_range)
            torch.cuda.synchronize()
            e = time()
            print(f"inverse_z_sampling {(e-s) * 1000:.2f} ms" ) 
            # print(bg_z_vals)

            # samples = rays_o[:,None,:] + bg_z_vals[:,:,None] * rays_d[:,None,:]
            # from tools import tools 
            # tools.points2obj("/data/wxc/data/sig23/street/sample.obj", samples.detach().cpu().numpy().reshape(-1,3))
            # print("done")
            # exit()

            s = time()
            pts_diffuse = torch.full((rays_d.shape[0],num_bg_sample,3), 0, dtype=torch.float32, device=self.device)
            pts_specular = torch.full((rays_d.shape[0],num_bg_sample,3), 0, dtype=torch.float32, device=self.device)
            pts_alpha = torch.full((rays_d.shape[0],num_bg_sample,1), 0, dtype=torch.float32, device=self.device)
            # print(tracing_blocks[:,0].contiguous())
            bg_pts_inference_v2(rays_o, rays_d, bg_z_vals, bg_bidxs, i,
                                self.block_corner, self.block_size, self.resolution,
                                self.feature_tables, self.params, pts_diffuse, pts_specular, pts_alpha)
            torch.cuda.synchronize()
            e = time()
            print(f"bg_pts_inference {(e-s) * 1000:.2f} ms" ) 
            # print(pts_diffuse)
            # print(pts_alpha)

            s = time()
            bg_transparency = torch.ones((rays_d.shape[0], 1), dtype=torch.float32, device=self.device)
            bg_temp_diffuse = torch.full((rays_d.shape[0],3), 0, dtype=torch.float32, device=self.device)
            bg_temp_specular = torch.full((rays_d.shape[0],3), 0, dtype=torch.float32, device=self.device)
            bg_temp_depth = torch.full((rays_d.shape[0],1), 0, dtype=torch.float32, device=self.device)
            accumulate_color(pts_diffuse, pts_specular, pts_alpha, bg_transparency, 
                             bg_z_vals, bg_temp_diffuse, bg_temp_specular, bg_temp_depth)
            torch.cuda.synchronize()
            e = time()
            print(f"accumulate {(e-s) * 1000:.2f} ms" ) 

            bg_diffuse += bg_temp_diffuse * bg_blend_weights[:,i:i+1]
            bg_specular += bg_temp_specular * bg_blend_weights[:,i:i+1]
            # print(bg_temp_diffuse, bg_temp_specular, bg_blend_weights[:,i:i+1])
            bg_depth += bg_temp_depth * bg_blend_weights[:,i:i+1]
        
        # print(bg_diffuse, bg_specular)
        # s = time()
        # accumulate_color(pts_diffuse, pts_specular, pts_alpha, transparency, 
        #                 bg_z_vals, diffuse, specular, depth)
        # torch.cuda.synchronize()
        # e = time()
        # print(f"accumulate {(e-s) * 1000:.2f} ms" ) 
        # print(diffuse, bg_diffuse)

        del pts_diffuse, pts_specular, pts_alpha
        torch.cuda.empty_cache()

        diffuse = diffuse + transparency * bg_diffuse
        specular = specular + transparency * bg_specular
        depth = depth + transparency * bg_depth

        diffuse = diffuse.reshape(H,W,3)
        specular = specular.reshape(H,W,3)
        depth = depth.reshape(H,W,1)
        transparency = transparency.reshape(H,W,1)
    

        return diffuse, specular, depth, transparency
        # # # ========================================================================
        
        # bg_bidxs = torch.full((rays_d.shape[0], 4), -1, dtype=torch.int16, device=self.device)
        # bg_blend_weights = torch.full((rays_d.shape[0], 4), 0, dtype=torch.float, device=self.device)

        # update_outgoing_bidx_v2(rays_o, rays_d, self.block_corner, self.block_size,
        #                        tracing_blocks, intersections, bg_bidxs, bg_blend_weights)
        # # # print(outgoing_bidxs)
        # # # print(bg_blend_weights)
        # # # exit()
        # blend_num_bg = 0
        # while bg_bidxs[0,blend_num_bg] != -1:
        #     blend_num_bg += 1
        
        # bg_blend_weights = bg_blend_weights[:, :blend_num_bg]
        # bg_blend_weights = bg_blend_weights / torch.sum(bg_blend_weights, dim=-1, keepdim=True)
        # # # print(bg_blend_weights)

        # bg_diffuse = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        # bg_specular = torch.zeros((rays_d.shape[0], 3),dtype=torch.float32, device=self.device)
        # bg_depth = torch.zeros((rays_d.shape[0], 1),dtype=torch.float32, device=self.device)
        # for i in range(blend_num_bg):
        #     s = time()
        #     bg_z_vals = torch.full((rays_d.shape[0], num_bg_sample), -1, dtype=torch.float32, device=self.device)
        #     inverse_z_sampling(intersections, bg_bidxs[...,i].contiguous(), bg_z_vals, sample_range)
        #     torch.cuda.synchronize()
        #     e = time()
        #     print(f"inverse_z_sampling {(e-s) * 1000:.2f} ms" ) 

        #     assert (bg_z_vals == -1).sum() == 0

        #     s = time()
        #     pts_diffuse = torch.full((rays_d.shape[0],num_bg_sample,3), 0, dtype=torch.float32, device=self.device)
        #     pts_specular = torch.full((rays_d.shape[0],num_bg_sample,3), 0, dtype=torch.float32, device=self.device)
        #     pts_alpha = torch.full((rays_d.shape[0],num_bg_sample,1), 0, dtype=torch.float32, device=self.device)
        #     # print(tracing_blocks[:,0].contiguous())
        #     bg_pts_inference_v2(rays_o, rays_d, bg_z_vals, bg_bidxs, i,
        #                        self.block_corner, self.block_size, self.resolution,
        #                       self.feature_tables, self.params, pts_diffuse, pts_specular, pts_alpha)
        #     torch.cuda.synchronize()
        #     e = time()
        #     print(f"bg_pts_inference_v2 {(e-s) * 1000:.2f} ms" ) 

        #     s = time()
        #     bg_transparency = torch.ones((rays_d.shape[0], 1), dtype=torch.float32, device=self.device)
        #     bg_temp_diffuse = torch.full((rays_d.shape[0],3), 0, dtype=torch.float32, device=self.device)
        #     bg_temp_specular = torch.full((rays_d.shape[0],3), 0, dtype=torch.float32, device=self.device)
        #     bg_temp_depth = torch.full((rays_d.shape[0],1), 0, dtype=torch.float32, device=self.device)
        #     accumulate_color(pts_diffuse, pts_specular, pts_alpha, bg_transparency, 
        #                      bg_z_vals, bg_temp_diffuse, bg_temp_specular, bg_temp_depth)
        #     torch.cuda.synchronize()
        #     e = time()
        #     print(f"accumulate {(e-s) * 1000:.2f} ms" ) 

        #     bg_diffuse += bg_temp_diffuse * bg_blend_weights[:,i:i+1]
        #     bg_specular += bg_temp_specular * bg_blend_weights[:,i:i+1]
        #     bg_depth += bg_temp_depth * bg_blend_weights[:,i:i+1]

        # diffuse = diffuse + transparency * bg_diffuse
        # specular = specular + transparency * bg_specular
        # depth = depth + transparency * bg_depth

        # diffuse = diffuse.reshape(H,W,3)
        # specular = specular.reshape(H,W,3)
        # depth = depth.reshape(H,W,1)
        # transparency = transparency.reshape(H,W,1)
        # # # print(diffuse.reshape(-1,3)[0])
        # # # exit()

        # # # print(pts_diffuse)
        # # # print(pts_specular)
        # # # print(pts_alpha)
        
        # # # samples = rays_o[:,None,:] + bg_z_vals[:,:,None] * rays_d[:,None,:]
        # # # samples = samples.reshape(H,W,num_bg_sample,3)[:300,200:600]
        # # # from tools import tools 
        # # # tools.points2obj("/home/yons/4TB/sig23/street/sample.obj", samples.detach().cpu().numpy().reshape(-1,3))
        # # # exit()

        # return diffuse, specular, depth, transparency

    
if __name__ == "__main__":
    import camera_utils

    cfg = utils.parse_yaml(sys.argv[1])
    gpuIdx = int(sys.argv[2])
    mode = "val"
    demo_name = sys.argv[3]
    try:
        startIdx = int(sys.argv[4])
        endIdx = int(sys.argv[5])
    except:
        startIdx = 0
        endIdx = -1

    print(startIdx, endIdx)

    sr = RenderingHashGrid(cfg.DATADIR, demo_name, gpuIdx, mode)
    sr.rendering(startIdx, endIdx)
