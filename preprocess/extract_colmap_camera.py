import os,sys,cv2 
import numpy as np 
from tools import tools 
from tools.poses import colmap_read_model as read_model


"""
Extract Colmap output to camera.log
Colmap project datadir 
"""
data_dir = ""

def load_colmap_data(data_dir):

    camerasfile = os.path.join(data_dir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    imagesfile = os.path.join(data_dir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    points3dfile = os.path.join(data_dir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    return camdata, imdata, pts3d

camdata, imdata, pts3d = load_colmap_data(data_dir)


list_of_keys = list(camdata.keys())
cam = camdata[list_of_keys[0]]
H, W, focal, cx, cy = cam.height, cam.width, cam.params[0], cam.params[1], cam.params[2]
print(H, W, focal, cx, cy)



images_dir = os.path.join(data_dir, "new_images")
if os.path.exists(images_dir) is False:
    os.mkdir(images_dir)

file = open(os.path.join(data_dir, "camera.log"), "w")


count = 0
for k in imdata:
    im = imdata[k]
    os.system(f"cp {os.path.join(data_dir, 'images', im.name)} {os.path.join(images_dir, f'{count}.png')}")
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])
    R = R.transpose(1,0)
    C = -1 * R @ t 
    file.write(f"{count}\n")
    file.write(f"{focal:.2f} {focal:.2f} {cx} {cy}\n")
    file.write(f"{W} {H} 0 1000\n")
    file.write(f"{R[0,0]:.8f} {R[0,1]:.8f} {R[0,2]:.8f} {C[0,0]:.8f}\n")
    file.write(f"{R[1,0]:.8f} {R[1,1]:.8f} {R[1,2]:.8f} {C[1,0]:.8f}\n")
    file.write(f"{R[2,0]:.8f} {R[2,1]:.8f} {R[2,2]:.8f} {C[2,0]:.8f}\n")
    file.write(f"0 0 0 1\n")
    count += 1
file.close()

ks, c2ws = tools.read_campara(os.path.join(data_dir, "camera.log"))


Rs = c2ws[:,:3,:3]
Cs = c2ws[:,:3,3]

print(Cs.max(), Cs.min())
points = tools.cameras_scatter(Rs, Cs, length=0.5, step=0.01)
tools.points2obj(os.path.join(data_dir, "camera.obj"), points)