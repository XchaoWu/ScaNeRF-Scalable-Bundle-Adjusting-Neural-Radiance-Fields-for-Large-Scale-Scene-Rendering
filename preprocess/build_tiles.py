import cv2,sys,os 
import torch 
from glob import glob 
import numpy as np 
from tqdm import tqdm 
import time  
sys.path += ["./", "../"]
from tools import tools 
from tools import utils 
from load_data import load_snisr, read_campara
from cfg import * 
from fastMesh import FastMesh
from cuda import ray_aabb_intersection_v2
from glob import glob


# data_dir = "/data/wxc/data/sig23/community_debug"
# tile_size = [20,13,30] # [X, Y(height), Z]
# overlap_ratio = 0.2
# offset = [-2, 7.5, 8]
# expect_num = 4
# min_num_image = 60 # for each tile, less will be dropped 
# max_num_tile = [100000, 1, 1]

# data_dir = "/data/wxc/data/sig23/park"
# tile_size = [25,25,25] # [X, Y(height), Z]
# overlap_ratio = 0.2
# offset = [5,12,-12]
# expect_num = 2
# min_num_image = 60 # for each tile, less will be dropped 
# max_num_tile = [100000, 1, 100000]


# data_dir = "/data/wxc/data/sig23/shady_path"
# tile_size = [25,12,25] # [X, Y(height), Z]
# overlap_ratio = 0.2
# offset = [-15,0,5]
# expect_num = 3
# min_num_image = 60 # for each tile, less will be dropped 
# max_num_tile = [100000, 1, 100000]


# data_dir = "/data/wxc/data/sig23/street_debug"
# tile_size = [16,10,16] # [X, Y(height), Z]
# overlap_ratio = 0.2
# offset = [9,4,5]
# expect_num = 9
# min_num_image = 60 # for each tile, less will be dropped 
# max_num_tile = [100000, 1, 100000]


cfg = utils.parse_yaml(sys.argv[1])
data_dir = cfg.DATADIR
tile_size = cfg.ALLOCATION.TILE_SIZE
overlap_ratio = cfg.ALLOCATION.OVERLAP_RATIO
offset = cfg.ALLOCATION.OFFSET
expect_num = cfg.ALLOCATION.EXPECT_NUM
min_num_image = cfg.ALLOCATION.MIN_NUM_IMAGE
max_num_tile = cfg.ALLOCATION.MAX_DIM_TILE
scene_type=cfg.ALLOCATION.SCENE_TYPE

GPU_idx = sys.argv[2]
thresh = 0.1




tile_dir = os.path.join(data_dir, "tiles")
if os.path.exists(tile_dir) is False:
    os.mkdir(tile_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_idx}"
device = torch.device("cuda:0")

tile_size = torch.tensor(tile_size, dtype=torch.float32)
print(tile_size)

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


fmesh = FastMesh(os.path.join(data_dir,"mesh/mesh.ply"))
scene_bound = fmesh.get_sceneinfo()
print(scene_bound)

scene_min_corner = scene_bound[:3] + torch.tensor(offset, dtype=torch.float32)
scene_max_corner = scene_bound[3:]
tile_side = torch.ceil((scene_max_corner - scene_min_corner) / tile_size).int()
print("scene_min_corner", scene_min_corner)

tile_side = [min(tile_side[0], max_num_tile[0]),
              min(tile_side[1], max_num_tile[1]),
              min(tile_side[2], max_num_tile[2])]
print("tile_side", tile_side)

xs,ys,zs = torch.meshgrid(torch.arange(tile_side[0]), 
                          torch.arange(tile_side[1]), 
                          torch.arange(tile_side[2]))
grid = torch.stack([xs,ys,zs], -1).reshape(-1,3)

# num_tile x 3
tile_corners = scene_min_corner + grid * (1-overlap_ratio) * tile_size 
# print(tile_corners)

ks, c2ws, H, W  = read_campara(os.path.join(data_dir, "camera.log"), True)

tools.points2obj(os.path.join(tile_dir,"camera.obj"),
                tools.cameras_scatter(c2ws[:,:3,:3].transpose(0,2,1), c2ws[:,:3,3]))

ks = torch.from_numpy(ks).to(device)
c2ws = torch.from_numpy(c2ws).to(device)


# num_tile x num_camera 
related_matrix = torch.zeros((tile_corners.shape[0], ks.shape[0]), dtype=torch.float32)

scale = 4
# scale = 2
box_centers = (tile_corners + tile_size / 2.).to(device)
box_sizes = torch.ones_like(box_centers) * tile_size.to(device)[None,:]
for cidx in tqdm(range(ks.shape[0])):
    k = ks[cidx]
    k = k / scale
    k[-1, -1] = 1.
    c2w = c2ws[cidx]
    rays_o, rays_d = utils.get_rays_torch_v2(H // scale, W // scale , k, c2ws[cidx])
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)

    # B X K x 2 
    bounds = torch.full((rays_d.shape[0], box_centers.shape[0], 2), -1, dtype=torch.float32, device=device)
    ray_aabb_intersection_v2(rays_o, rays_d, box_centers, box_sizes, bounds)
    
    # B x K 
    # valid = torch.all(bounds != -1, dim=-1)
    bounds[bounds == -1] = 1e7
    # B x 1
    depth = fmesh.render_depth(rays_o, rays_d)
    depth[depth == 0] = 1e5 # sky ? 
    # valid = depth.squeeze() > 0
    
    # K 
    occupied_ratio = torch.sum(bounds[..., 0] < depth, dim=0) / (H * W) * (scale ** 2)
    related_matrix[:, cidx] = occupied_ratio.cpu()


# num_camera x 3 
camera_centers = c2ws[:,:,3].cpu()



# tile_score = torch.zeros_like(tile_corners)
# num_tile x num_camera
tile_score = torch.norm(camera_centers[None, ...] - (tile_corners[:,None,:] + tile_size / 2.), dim=-1).mean(-1)
# tile_score = related_matrix.mean(-1)
# print(distance)
# exit()

# num_tile x num_camera x 3
cam_loc = (camera_centers[None,...] - tile_corners[:,None,:]) / tile_size 
# num_tile x num_camera
inside = torch.all( (cam_loc >= 0) & (cam_loc < 1), dim=-1)

tile_ignore = torch.where(torch.all(inside == False, dim=-1)==True)[0].numpy().tolist()

valid_tile_list = list(range(tile_corners.shape[0]))
valid_tile_list = [_ for _ in valid_tile_list if _ not in tile_ignore]

print(f"first select: Num {len(valid_tile_list)}")

if len(valid_tile_list) < expect_num:
    print("less than expect ...")
    candidate = np.array(tile_ignore)[torch.argsort(tile_score[tile_ignore], descending=False)].tolist()
    valid_tile_list = valid_tile_list + candidate[:expect_num-len(valid_tile_list)]
elif len(valid_tile_list) > expect_num:
    print("more than expect ...")
    candidate = np.array(valid_tile_list)[torch.argsort(tile_score[valid_tile_list], descending=False)].tolist()
    valid_tile_list = candidate[:expect_num]
valid_tile_list.sort()

# indoor scenes this is better 
if scene_type == "indoor":
    final_score = related_matrix
else:
    final_score = thresh * inside + related_matrix

final_score[:, ignore] = 0
sorted_cores, sorted_images = torch.sort(final_score, dim=1, descending=True)


f = open(os.path.join(tile_dir, "training_views.txt"), "w")
# valid_tile_list = []
new_valid_tile_list = []
for tidx, i in enumerate(valid_tile_list):
    # if i in tile_ignore:
    #     continue 
    scores = sorted_cores[i]
    images = sorted_images[i]

    select_images = images[scores > thresh].cpu().numpy().tolist()

    if len(select_images) > min_num_image:
        print(len(select_images))
        new_valid_tile_list.append(i)
        f.write(f"{len(new_valid_tile_list)-1}\n")
        f.write(" ".join([str(item) for item in select_images]) + "\n")

f.close()

tile_corners = tile_corners[new_valid_tile_list]
tile_centers = tile_corners + tile_size / 2. 

geometry = tools.draw_AABB(tile_centers.numpy(), 
                         (torch.ones_like(tile_centers) * tile_size).numpy())
tools.mesh2obj(os.path.join(tile_dir, "tiles.obj"), geometry[0], geometry[1])

if scene_type == "outdoor":
    resolution = 8192 
else:
    resolution = 4096

with open(os.path.join(tile_dir, "tile_info.txt"), "w") as f:
    f.write("# TILEID(1) BBOX_CORNER(3) BBOX_SIZE(3) RESOLUTION(2) FLAG(1)\n")
    for i in range(tile_corners.shape[0]):
        f.write(f"{i} {tile_corners[i][0]:.2f} {tile_corners[i][1]:.2f} {tile_corners[i][2]:.2f} {tile_size[0]:.2f} {tile_size[1]:.2f} {tile_size[2]:.2f} 32 {resolution} 0\n")