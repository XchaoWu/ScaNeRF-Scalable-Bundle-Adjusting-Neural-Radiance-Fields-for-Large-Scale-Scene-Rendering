from math import gamma
import torch 
import torch.nn as nn 
import warp_loss
from mono_loss import (
    DepthConsistencyLoss, 
    DepthSmoothLoss, )

import numpy as np 
from easydict import EasyDict as edict


def depth_weight_decay_func(weight, step):
    return weight * (0.1 ** (step / 30000))

def smooth_weight_decay_func(weight, step):
    return weight * (0.1 ** (step / 30000))

def warp_weight_warming_func(weight, step):
    # return weight * (1 - np.exp(-step / 2000))
    # return weight * min(np.exp((step-10000) / 2000), 1.0)
    return weight * max(min(step / 10000, 1.0),0.0)

class LossItem:
    def __init__(self, loss_name, loss_func, loss_weight, start_step=0, end_step=1000000000,
                 decay_func = None):
        self.loss_name = loss_name 
        self.loss_func = loss_func 
        self.start_weight = loss_weight
        self.loss_weight = loss_weight 
        self.record_list = []
        self.start_step = start_step
        self.end_step = end_step
        self.decay_func = decay_func
    
    def toCPU(self):
        try:
            self.loss_func.toCPU()
        except:
            pass
            # print(f"{self.loss_name} has no func to CPU")
        # else:
        #     print(f"{self.loss_name} to CPU successfully!")
    
    def toGPU(self):
        try:
            self.loss_func.toGPU()
        except:
            pass
            # print(f"{self.loss_name} has no func to GPU")
        # else:
            # print(f"{self.loss_name} to GPU successfully!")

    def calMeanloss(self):

        if len(self.record_list) > 0:
            meanloss = np.mean(self.record_list)
            self.record_list = []
        else:
            meanloss = None 
        return meanloss 
    
    def getInfo(self):
        meanloss = self.calMeanloss()
        if meanloss != None:
            info = "%-10s\t%.8f\tweight: %.8f\n" % (self.loss_name, meanloss, self.loss_weight)
        else:
            info = ""
        return info 

    def __call__(self, loss, global_step, **kwargs):

        if self.decay_func != None and global_step > self.start_step:
            self.loss_weight = self.decay_func(self.start_weight, global_step-self.start_step)
        
        if global_step > self.start_step and global_step < self.end_step:
            this_loss = self.loss_func(**kwargs)

            if this_loss != None: 
                loss = loss + self.loss_weight * this_loss 
                self.record_list += [float(this_loss)]

        return loss 

class Criterions:
    def __init__(self, cfg, tile):
        
        item_list = []
        if cfg.TRAINING.LOSS.WEIGHT_RGB_LOSS > 0:
            item_list+= [LossItem("RGB Loss", nn.MSELoss(), cfg.TRAINING.LOSS.WEIGHT_RGB_LOSS, cfg.TRAINING.LOSS.RGB_LOSS_START)]

        if cfg.TRAINING.LOSS.WEIGHT_WARP_LOSS > 0:
            decay_func = None
            if cfg.TRAINING.LOSS.WARP_WARPING:
                decay_func = warp_weight_warming_func
            item_list += [LossItem("Warp Loss", warp_loss.WarpLoss(cfg, tile), cfg.TRAINING.LOSS.WEIGHT_WARP_LOSS, cfg.TRAINING.LOSS.WARP_LOSS_START,
                                    decay_func=decay_func)]

        if cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS > 0:
            item_list += [LossItem("Depth Loss", DepthConsistencyLoss(), cfg.TRAINING.LOSS.WEIGHT_DEPTH_LOSS, cfg.TRAINING.LOSS.DEPTH_LOSS_START,
                                    end_step=cfg.TRAINING.TOTAL_STEP, decay_func=depth_weight_decay_func)]
        
        if cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS > 0:
            item_list += [LossItem("Smooth Loss", DepthSmoothLoss(cfg, tile), cfg.TRAINING.LOSS.WEIGHT_DEPTH_SMOOTH_LOSS, cfg.TRAINING.LOSS.DEPTH_SMOOTH_LOSS_START,
                                    end_step=cfg.TRAINING.TOTAL_STEP, decay_func=smooth_weight_decay_func)]
        
        if cfg.RHO > 0:
            item_list += [LossItem("Admm Loss", tile.consensus_manager, 1.0, start_step=cfg.SYN_START, end_step=cfg.TRAINING.TOTAL_STEP)]

        self.item_list = item_list

        self.record_list = []
    
    def toCPU(self):
        for i in range(len(self.item_list)):
            self.item_list[i].toCPU()
    
    def toGPU(self):
        for i in range(len(self.item_list)):
            self.item_list[i].toGPU()        

    def __call__(self, **kwargs):
        loss = 0
        global_step = kwargs['global_step']
        
        valid = None 
        if kwargs["ret_fg"]:
            valid = kwargs["fore_valid"]
        if kwargs["ret_bg"]:
            bg_valid = kwargs["bg_valid"]
            if valid != None:
                valid = valid | bg_valid
            else:
                valid = bg_valid

        for item in self.item_list:
            if item.loss_name == "RGB Loss":
                # if kwargs["ret_fg"]:
                #     valid = fore_valid | bg_valid
                # else:
                #     valid = bg_valid
                loss = item(loss, global_step, 
                            input=kwargs["input"][valid], 
                            target=kwargs["target"][valid])
            elif item.loss_name == "GNormal Loss":
                loss = item(loss, global_step, 
                            output_normal=kwargs["output_normal"],
                            grad_normal=kwargs["grad_normal"],
                            viewdirs=kwargs["rays_d"],
                            weights=kwargs["weights"])
            elif item.loss_name == "Warp Loss":
                # only being used in foreground 
                # if kwargs["ret_fg"]:
                # if kwargs["ret_fg"]:
                #     valid = fore_valid | bg_valid
                # else:
                #     valid = bg_valid
                loss = item(loss, global_step,
                            steps=global_step,
                            rays_o=kwargs["rays_o"], 
                            rays_d=kwargs["rays_d"],
                            depth=kwargs['pred_depth'], 
                            diffuse=kwargs["pred_diffuse"], 
                            specular=kwargs["pred_specular"],
                            ray_colors=kwargs['target'],
                            valid=valid,
                            occlusions=kwargs["occlusions"],
                            ori_poses_idxs=kwargs["ori_poses_idxs"])
            elif item.loss_name == "Depth Loss":
                loss = item(loss, global_step, 
                            rendered_depth = kwargs["rendered_depth"],
                            monocular_depth = kwargs["monocular_depth"],
                            mask = kwargs["valid_mask"])
            elif item.loss_name == "Normal Loss":
                loss = item(loss, global_step,
                            rendered_normal = kwargs["rendered_normal"],
                            monocular_normal = kwargs["monocular_normal"],
                            mask = kwargs["valid_mask"])
            elif item.loss_name == "Smooth Loss":
                loss = item(loss, global_step,
                            pixel_locs = kwargs["pixel_locs"],
                            rendered_depth = kwargs["rendered_depth"],
                            monocular_normal = kwargs["monocular_normal"],
                            mask = kwargs["valid_mask"])
            elif item.loss_name == "Admm Loss":
                loss = item(loss, global_step)
            elif item.loss_name == "BF Loss":
                raise NotImplementedError
                # loss = item(loss, global_step,
                #             pred_fore = kwargs["rgb"],
                #             gt = kwargs["target"][fore_valid],
                #             pred_depth = kwargs['pred_depth'][fore_valid],
                #             bound = kwargs["bound"])

        self.record_list += [loss.item()]
        return loss 
    
    def getInfo(self):
        info = ""
        for item in self.item_list:
            info += item.getInfo()
        meanloss = np.mean(self.record_list)
        info += "%-10s\t%.8f\n" % ("Total Loss", meanloss)
        self.record_list = []
        
        return info 

