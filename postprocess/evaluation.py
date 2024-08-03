import sys,os,cv2 
from glob import glob 
import torch 
import numpy as np 
# import lpips 
sys.path += ["./", "../"]
from tools import utils
from tools.ssim import SSIM 

pred_path = sys.argv[1]
gt_path = sys.argv[2]

ssim_cal = SSIM(window_size=11).cuda()

def cal_psnr(I1,I2):
    mse = torch.mean((I1-I2)**2)
    if mse < 1e-10:
        return 100
    return 10 * float(torch.log10(255.0**2/mse))


files = glob(os.path.join(pred_path, '*.png'))
files.sort(key = lambda x: int(os.path.splitext(os.path.basename(x))[0]) )


psnrs = []
ssims = []
lpips_list = []

for file in files:
    name = os.path.basename(file)
    gt = cv2.imread(os.path.join(gt_path, name))
    pred = cv2.imread(file)
    gt = gt[:pred.shape[0], :pred.shape[1]]

    gt = torch.from_numpy(gt).float()
    pred = torch.from_numpy(pred).float()
    psnr = cal_psnr(gt,pred)
    psnrs.append(psnr)

    ssim = ssim_cal(gt[None,...].permute(0,3,1,2).cuda() / 255. , pred[None,...].permute(0,3,1,2).cuda() / 255.)
    ssim = float(ssim.cpu())
    ssims.append(ssim)


    print(name, psnr, ssim)

print(f"mean psnr: {np.mean(psnrs):.5f} mean SSIM: {np.mean(ssims):.5f}")
