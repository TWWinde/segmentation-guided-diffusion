import os
from PIL import Image
import numpy as np
import pytorch_msssim
import lpips
import torch
from pytorch_fid import fid_score

def compute_metrics( ):
    pips, ssim, psnr, rmse  = [], [], [], []
    loss_fn_alex = lpips.LPIPS(net='vgg')
    loss_fn_alex = loss_fn_alex.to('cuda:0')
    path_real_root = "/data/private/autoPET/autopet_2d/image/test"
    path_fake_root = "/data/private/autoPET/ddim-AutoPET-256-segguided/samples_many_320"
    path_list = os.listdir(path_fake_root)
    for item in path_list:
        path_fake = os.path.join(path_fake_root, item)
        real_name = item.replace("condon_", "")
        path_real = os.path.join(path_real_root, real_name)
        input1 = Image.open(path_real)
        input1 = np.array(input1)/255.0
        input2 = Image.open(path_fake)
        input2 = np.array(input2)/255.0

        input3 = torch.tensor(input1, dtype=torch.float32)
        input4 = torch.tensor(input2, dtype=torch.float32)
        input3 = input3.unsqueeze(0).unsqueeze(0).to('cuda:0')  # (1, 1, 256, 256)
        input4 = input4.unsqueeze(0).unsqueeze(0).to('cuda:0')

        ssim_value = pytorch_msssim.ssim(input3, input4)
        ssim.append(ssim_value.mean().item())
        # PIPS lpips
        d = loss_fn_alex(input3, input4)
        pips.append(d.mean().item())
        # PSNR, RMSE
        mse = torch.nn.functional.mse_loss(input3, input4)
        max_pixel_value = 1.0
        psnr_value = 10 * torch.log10((max_pixel_value ** 2) / mse)
        rmse_value = torch.sqrt(mse)
        psnr.append(psnr_value.mean().item())
        rmse.append(rmse_value.mean().item())

    total_samples = len(pips)
    avg_pips = sum(pips) / total_samples
    avg_ssim = sum(ssim) / total_samples
    avg_psnr = sum(psnr) / total_samples
    avg_rmse = sum(rmse) / total_samples
    avg_pips = np.array(avg_pips)
    avg_ssim = np.array(avg_ssim)
    avg_psnr = np.array(avg_psnr)
    avg_rmse = np.array(avg_rmse)
    fid_value = fid_score.calculate_fid_given_paths([path_real_root, path_fake_root], batch_size=50, device='cuda', dims=2048)
    print(f"FID: {fid_value}")
    print(avg_pips, avg_ssim, avg_psnr, avg_rmse)

if __name__ == "__main__":
    compute_metrics()