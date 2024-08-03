import numpy as np 
import cv2,os,sys 
import torch 
import torch.nn as nn 
from glob import glob 
from tqdm import tqdm 
import re 

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = c2w @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    return c2w

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def read_campara(path, return_shape=False):
    """
    read camera paras of Indoor Scene 
    format  

    index 
    fx fy cx cy
    width height near far 
    r11 r12 r13 t1
    r21 r22 r23 t2
    r31 r32 r33 t3 (camera2world)
    0   0   0   1
    """
    trans = lambda x:float(x)
    Ks = []
    C2Ws = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for i in range(0,len(lines), 7):
        item = lines[i:i+7]
        name = item[0].strip()
        fx,fy,cx,cy = map(trans, re.split(r"\s+",item[1].strip()))
        width, height, near, far = map(trans, re.split(r"\s+",item[2].strip()))
        r11,r12,r13,t1 = map(trans, re.split(r"\s+",item[3].strip()))
        r21,r22,r23,t2 = map(trans, re.split(r"\s+",item[4].strip()))
        r31,r32,r33,t3 = map(trans, re.split(r"\s+",item[5].strip()))
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        RT = np.array([[r11,r12,r13,t1],
                       [r21,r22,r23,t2],
                       [r31,r32,r33,t3]],dtype=np.float32)
        Ks += [K]
        C2Ws += [RT]
    Ks = np.stack(Ks, 0)
    C2Ws = np.stack(C2Ws, 0)
    print('\n=== Finish Loading camera ==')
    print(f'Ks shape: {Ks.shape}\tC2Ws shape: {C2Ws.shape}')
    if return_shape == False:
        return Ks, C2Ws
    else:
        return Ks, C2Ws, int(height), int(width)

def read_images(path, idx_list=None):

    # extract_id = lambda x: int(os.path.splitext(os.path.basename(x))[0])

    file_list = [os.path.join(path, f"{idx}.png") for idx in idx_list]

    images = []
    for file in tqdm(file_list):
        img = cv2.imread(file) / 255.
        images.append(img)
    
    images = np.array(images)

    return images 

def read_depth(path, idx_list=None):
    file_list = [os.path.join(path, f"{idx}.npy") for idx in idx_list]
    depths = []
    for file in tqdm(file_list):
        dep = np.load(file)
        depths.append(dep)
    
    depths = np.array(depths)

    return depths 

def load_snisr(data_dir, idx_list=None, omni_depth=False, omni_normal=False):


    ignore = []
    try:
        f = open(os.path.join(data_dir, "ignore.log"), 'r')
    except:
        pass
    else:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 1:
                ignore += [int(line[0])]
            elif len(line) == 2:
                ignore += list(np.arange(int(line[0]), int(line[1])))
    

    ks, c2ws, H, W = read_campara(os.path.join(data_dir, "camera.log"), True)

    # c2ws = c2ws @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    if idx_list == None:
        idx_list = [ _ for _ in range(ks.shape[0])]
    
    idx_list = [item for item in idx_list if item not in ignore]

    ks = ks[idx_list]
    c2ws = c2ws[idx_list]

    images = read_images(os.path.join(data_dir, "images"), 
                         idx_list)
    
    # depths = read_depth(os.path.join(data_dir, "depths"),
                        #  idx_list)
    
    if omni_depth:
        mono_depths = read_depth(os.path.join(data_dir, "mono_depths"), idx_list)
    else:
        mono_depths  = None
    
    if omni_normal:
        mono_normals = read_depth(os.path.join(data_dir, "mono_normals"), idx_list)
    else:
        mono_normals = None 



    # render_poses = np.stack([pose_spherical(angle, -45, 13) for angle in np.linspace(90,270,40+1)[:-1]], 0)

    return images, None, c2ws, ks, H, W, mono_depths, mono_normals, idx_list

# if __name__ == "__main__":
#     images, c2ws, ks, H, W  = load_snisr("/data/wxc/data/sig23/wangchi_scene/", 
#                                          [1,17,18,23])
#     print(images.shape, c2ws.shape, ks.shape)


