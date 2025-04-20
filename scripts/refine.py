import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

def apply_fade_to_limb(limb_input, transition_height=20):
    """
    mask: (B, 1, H, W)
    limb_input: (B, 3, H, W)
    return: limb_aug: (B, 3, H, W)
    """
    mask = (limb_input[:, 0:1, :, :] > 0).float()  # shape: (B, 1, H, W)

    B, _, H, W = mask.shape
    device = mask.device

    limb_aug = limb_input.clone()

    for b in range(B):
        # Find the valid region
        mask_b = mask[b, 0]  # shape: H x W
        ys, xs = torch.where(mask_b > 0)
        if len(ys) > 0:
            min_y = ys.min().item()
            end_y = min(H, min_y + transition_height)

            # Create fade_map: a gradient from 1 to 0, shape: H x 1
            fade_weights = torch.linspace(1, 0, end_y - min_y, device=device).unsqueeze(1)  # shape: (T, 1)
            fade_weights = 1 - fade_weights  # Convert to gradient from 0 to 1
            fade_map = torch.ones((H, W), dtype=torch.float32, device=device)

            fade_map[min_y:end_y, :] = fade_weights

            # Create fade_mask (only within valid limb region)
            y_range = torch.arange(H, device=device).unsqueeze(1)  # H x 1
            fade_mask = (mask_b > 0) & (y_range >= min_y) & (y_range < end_y)  # H x W, bool

            # Expand fade_map to 3 channels
            fade_map_3ch = fade_map.unsqueeze(0).expand(3, H, W)  # 3 x H x W

            # Apply fade
            limb_aug[b][:, fade_mask] *= fade_map_3ch[:, fade_mask]
        else:
            limb_aug[b] = limb_input[b]

    return limb_aug


def refine_res(dst, name, dataroot, mode='test', pair_mode=None, pre_data_dir=None):
    # src
    src_path = os.path.join(dataroot, mode, "image", name)
    src = cv2.imread(src_path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (384, 512))
    # dst
    dst = dst.astype("uint8")
    # parse
    parse_s = Image.open(os.path.join(dataroot, mode, "image-parse-v3", name.replace("jpg", "png"))).convert("L").resize((384, 512))
    parse_s = np.array(parse_s)
    parse_t = Image.open(os.path.join(pre_data_dir, pair_mode, "parse", name)).convert("L").resize((384, 512))
    parse_t = np.array(parse_t)
    # mask
    mask = (parse_s!= 0).astype(np.float32) + (parse_t!=0).astype(np.float32)
    mask = np.clip(mask, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=5)
    mask = 255 - mask

    output = cv2.seamlessClone(src, dst, mask, (192,256), cv2.NORMAL_CLONE)
    return output