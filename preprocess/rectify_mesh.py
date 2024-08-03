import os,sys,cv2 
import numpy as np 
sys.path += ["./","../"]
from tools import tools 
import trimesh
import pyembree


"""

"""

mesh_path = ""
cam_path = ""

angle = (-90, 20, 0)
mesh_center = np.array([0,0,0])
scale = 2

def Rx(theta):
    return np.array([[1, 0, 0], 
                    [0, np.cos(theta), -np.sin(theta)], 
                    [0, np.sin(theta), np.cos(theta)]], 
                    dtype=np.float32)

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], 
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]], 
                    dtype=np.float32)

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]], 
                    dtype=np.float32)


mesh = trimesh.load(mesh_path)
Ks, C2Ws, H, W = tools.read_campara(cam_path, True)

vertices = mesh.vertices
if np.any(mesh_center) == None:
    mesh_center = np.mean(vertices, axis=0)


Rs = C2Ws[:,:3,:3]
Cs = C2Ws[:,:3,3]

rotation = Rz(angle[2] / 180 * np.pi) @ Ry(angle[1] / 180 * np.pi) @ Rx(angle[0] / 180 * np.pi)
vertices = scale * ((vertices - mesh_center) @ rotation.transpose())

Cs = scale * ((Cs - mesh_center) @ rotation.transpose())
Rs = rotation @ Rs 

mesh.vertices = vertices
C2Ws = np.concatenate([Rs, Cs[...,None]],axis=-1)


with open(os.path.join(os.path.split(mesh_path)[0], "align_info.txt"), "w") as f:
    f.write(f"center: {mesh_center[0]} {mesh_center[1]} {mesh_center[2]}\n")
    f.write(f"angle: {angle[0]} {angle[1]} {angle[2]}\n")
    f.write(f"scale: {scale}\n")

mesh.export(os.path.join(os.path.split(mesh_path)[0], "mesh_align.ply"), "ply")
# tools.mesh2obj(os.path.join(os.path.split(mesh_path)[0], "mesh_align.obj"), mesh.vertices, mesh.faces+1)

file = open(os.path.join(os.path.split(cam_path)[0], "camera_align.log"), "w")

count = 0
for k, c2w in zip(Ks, C2Ws):
    file.write(f"{count}\n")
    file.write(f"{k[0,0]:.2f} {k[1,1]:.2f} {k[0,2]} {k[1,2]}\n")
    file.write(f"{W} {H} 0 1000\n")
    file.write(f"{c2w[0,0]:.8f} {c2w[0,1]:.8f} {c2w[0,2]:.8f} {c2w[0,3]:.8f}\n")
    file.write(f"{c2w[1,0]:.8f} {c2w[1,1]:.8f} {c2w[1,2]:.8f} {c2w[1,3]:.8f}\n")
    file.write(f"{c2w[2,0]:.8f} {c2w[2,1]:.8f} {c2w[2,2]:.8f} {c2w[2,3]:.8f}\n")
    file.write(f"0 0 0 1\n")
    count += 1

file.close()

Rs = C2Ws[:,:3,:3]
Cs = C2Ws[:,:3,3]

# print(Cs.max(), Cs.min())
# points = tools.cameras_scatter(Rs, Cs, length=0.5, step=0.01)
# tools.points2obj(os.path.join(data_dir, "camera.obj"), points)



