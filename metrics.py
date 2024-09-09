import os
from PIL import Image
import numpy as np
import pytorch_msssim
import lpips
import torch
from pytorch_fid import fid_score
from scipy import linalg # For numpy FID
from pathlib import Path
from fid_folder.inception import InceptionV3


def compute_metrics( ):
    pool1, pool2 = [], []
    pips, ssim, psnr, rmse = [], [], [], []
    loss_fn_alex = lpips.LPIPS(net='vgg')
    loss_fn_alex = loss_fn_alex.to('cuda:0')
    path_real_root = "/data/private/autoPET/autopet_2d/image/test"
    path_fake_root = "/data/private/autoPET/ddim-AutoPET-256-segguided/samples_many_320"
    path_list = os.listdir(path_fake_root)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model_inc = InceptionV3([block_idx])
    model_inc.cuda()
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
        input3_rgb = input3.expand(-1, 3, -1, -1)
        input4_rgb = input4.expand(-1, 3, -1, -1)
        pool_real = model_inc(input3_rgb.float())[0][:, :, 0, 0]
        pool1 += [pool_real]
        pool_fake = model_inc(input4_rgb.float())[0][:, :, 0, 0]
        pool2 += [pool_fake]


    total_samples = len(pips)
    real_pool = torch.cat(pool1, 0)
    mu_real, sigma_real = torch.mean(real_pool, 0), torch_cov(real_pool, rowvar=False)
    fake_pool = torch.cat(pool2, 0)
    mu_fake, sigma_fake = torch.mean(real_pool, 0), torch_cov(fake_pool, rowvar=False)
    fid = numpy_calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6)
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
    print(avg_pips, avg_ssim, avg_psnr, avg_rmse, fid)


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        Taken from https://github.com/bioinf-jku/TTUR
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1, sigma1, mu2, sigma2 = mu1.detach().cpu().numpy(), sigma1.detach().cpu().numpy(), mu2.detach().cpu().numpy(), sigma2.detach().cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            #print('wat')
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #print('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return out


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


if __name__ == "__main__":
    compute_metrics()