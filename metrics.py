import os
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch_fidelity import calculate_metrics
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def compute_fid_kid(folder1, folder2, fid=True, kid=True):
    metrics = calculate_metrics(input1=folder1, input2=folder2, fid=fid, kid=kid)
    if fid and kid:
        return metrics["frechet_inception_distance"], metrics["kernel_inception_distance_mean"]
    elif fid:
        return metrics["frechet_inception_distance"], np.nan
    elif kid:
        return np.nan, metrics["kernel_inception_distance_mean"]

def compute_ssim_lpips_psnr(folder1, folder2, SSIM=True, LPIPS=True, PSNR=False):

    image_names = sorted(os.listdir(folder1), key=lambda x: int(os.path.splitext(x)[0]))

    if LPIPS:
        lpips_metric = lpips.LPIPS(net='alex')

    ssim_scores = []
    lpips_scores = []
    psnr_scores = []
    for name in tqdm(image_names):
        name = name.replace("png", "jpg")
        if os.path.exists(os.path.join(folder1, name)) == False:
            name = name.replace("jpg", "png")
        img1 = np.array(Image.open(os.path.join(folder1, name)).convert('RGB').resize((384, 512)))
        if os.path.exists(os.path.join(folder2, name)) == False:
            name = name.replace("png", "jpg")
        img2 = np.array(Image.open(os.path.join(folder2, name)).convert('RGB').resize((384, 512)))

        if SSIM:
            score = structural_similarity(img1, img2, data_range=255, channel_axis=2)
            ssim_scores.append(score)

        if LPIPS:
            img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 255
            img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 255
            score = lpips_metric(img1, img2).item()
            lpips_scores.append(score)

        if PSNR:
            score = peak_signal_noise_ratio(img1, img2, data_range=255)
            psnr_scores.append(score)

    return np.mean(ssim_scores), np.mean(lpips_scores), np.mean(psnr_scores)
    

if __name__ == "__main__":
    gt_folder = ""  # the path to a clothing or image with a resolution of 512×384
    res_pair_path = ""

    # =============== FID and KID ==================
    fid_score, kid_score = compute_fid_kid(res_pair_path, gt_folder, fid=True, kid=True)
    
    # =============== SSIM, LPIPS, and PSNR ==================
    ssim_score, lpips_score, psnr_score = compute_ssim_lpips_psnr(res_pair_path, gt_folder, SSIM=True, LPIPS=False, PSNR=True)
    
    print(f"# FID: {fid_score:.4f}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.4f}")