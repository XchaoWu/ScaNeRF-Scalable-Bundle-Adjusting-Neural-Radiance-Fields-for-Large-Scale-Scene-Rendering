import os,sys,cv2 
import numpy as np 
from tqdm import tqdm 
from multiprocessing import Process

data_dir = ""
output_dir = ""
image_dir = os.path.join(data_dir, "bundler")
bundler_file = os.path.join(data_dir, "bundler", "bundler.out")
num_core = 20

with open(bundler_file, "r") as f:
    lines = f.readlines()

while lines[0][0] == "#":
    lines = lines[1:]

num_camera = int(lines[0].split(" ")[0])
lines = lines[1:]

print(f"num camera: {num_camera}")


f = open(os.path.join(output_dir, "coarse_camera.log"), "w")

WIDTH = 1000
HEIGHT = 680
crop_left = 40
crop_top = 30
out_img_dir = os.path.join(output_dir, "images")
if os.path.exists(out_img_dir) is True:
    os.system(f"rm -rf {out_img_dir}")
os.mkdir(out_img_dir)

for i in tqdm(range(num_camera)):
    item = lines[i*5 : (i+1)*5]
    
    focal = float(item[0].strip().split(" ")[0])
    r1 = np.array(list(map(lambda x: float(x), item[1].strip().split(" "))))
    r2 = np.array(list(map(lambda x: float(x), item[2].strip().split(" "))))
    r3 = np.array(list(map(lambda x: float(x), item[3].strip().split(" "))))
    t = np.array(list(map(lambda x: float(x), item[4].strip().split(" "))))
    R = np.stack([r1,r2,r3], 0)
    R = R.transpose(1,0)
    c = -1 * R @ t[:, None]

    R[:, 1] *= -1
    R[:, 2] *= -1
    c2w = np.concatenate([R,c], -1)

    # 4 x 4
    c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], 0)
    global_trans_1 = np.eye(4)
    global_trans_1[1,1] *= -1
    global_trans_2 = np.eye(4)
    global_trans_2[1,1] = 0
    global_trans_2[1,2] = 1
    global_trans_2[2,1] = 1
    global_trans_2[2,2] = 0

    c2w = np.linalg.inv(global_trans_2) @ np.linalg.inv(global_trans_1) @ c2w 

    img = cv2.imread(os.path.join(image_dir, "%05d.png" % i))

    H, W = img.shape[:2]

    img = img[crop_top:crop_top+HEIGHT,crop_left:crop_left+WIDTH]

    # W 
    col_selection = np.bool_(np.min(img.mean(-1), axis=0))
    # H 
    row_selcetion = np.bool_(np.min(img.mean(-1), axis=1))
    # print(list(col_selection))
    # assert np.bool_(np.min(img.mean(-1))).min() == True 


    if col_selection[0] == False or col_selection[-1] == False or \
        row_selcetion[0] == False or row_selcetion[-1] == False:

        temp_img = img.copy()
        temp_img[np.where(img.mean(-1) == 0)] = (0,0,255)
        cv2.imwrite(os.path.join(out_img_dir, f"{i}_check.png"), temp_img)
        # exit()
    # print(np.bool_(np.min(img.mean(-1))).min())
    # img[np.where(img.mean(-1) == 0)] = (0,0,255)
    cv2.imwrite(os.path.join(out_img_dir, f"{i}.png"), img)

    cx = W / 2. - crop_left 
    cy = H / 2. - crop_top 

    f.write(f"{i}\n")
    f.write(f"{focal:.2f} {focal:.2f} {cx} {cy}\n")
    f.write(f"{WIDTH} {HEIGHT} 0 1000\n")
    f.write(f"{c2w[0,0]:.8f} {c2w[0,1]:.8f} {c2w[0,2]:.8f} {c2w[0,3]:.8f}\n")
    f.write(f"{c2w[1,0]:.8f} {c2w[1,1]:.8f} {c2w[1,2]:.8f} {c2w[1,3]:.8f}\n")
    f.write(f"{c2w[2,0]:.8f} {c2w[2,1]:.8f} {c2w[2,2]:.8f} {c2w[2,3]:.8f}\n")
    f.write(f"0 0 0 1\n")

f.close()

print("finished extracting camera log from reality capture!")
