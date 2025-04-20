import os
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)

class VITONDataset(data.Dataset):
    def __init__(self, dataroot, image_size=512, mode='train', semantic_nc=13, pair_mode="unpaired", pre_data_dir=None):
        super(VITONDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.pair_mode = pair_mode
        self.datamode = mode  # train or test or self-defined
        self.data_list = mode + '_pairs.txt'
        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 256)
        self.semantic_nc = semantic_nc
        self.data_path = osp.join(dataroot, mode)
        self.pre_data_dir = osp.join(pre_data_dir, pair_mode)
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(dataroot, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = 'image/' + im_name

        c_name = {}
        c = {}
        cm = {}

        key = self.pair_mode

        c_name[key] = self.c_names[key][index]
        c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
        c[key] = transforms.Resize(self.crop_size, interpolation=2)(c[key])
        c_img = c[key]
        cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
        cm[key] = transforms.Resize(self.crop_size, interpolation=0)(cm[key])
        cm_img = cm[key]

        c[key] = self.transform(c[key])  # [-1,1]
        cm_array = np.array(cm[key])
        cm_array = (cm_array >= 128).astype(np.float32)
        cm[key] = torch.from_numpy(cm_array)  # [0,1]
        cm[key].unsqueeze_(0)
        # c[key] = c[key] * cm[key]

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.crop_size, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # parse map
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        inpaint_mask = Image.open(osp.join(self.data_path, "agnostic-mask", os.path.basename(im_name).replace(".jpg", "_mask.png")))
        inpaint_mask = transforms.Resize(self.crop_size, interpolation=2)(inpaint_mask)
        inpaint_mask = self.toTensor(inpaint_mask)
        inpaint_mask = 1 - inpaint_mask[0].unsqueeze(0)  # (1, 512, 512)

        warped_cloth = Image.open(osp.join(self.pre_data_dir, "warp_cloth", os.path.basename(c_name[key])))
        warped_cloth = transforms.Resize(self.crop_size, interpolation=2)(warped_cloth)
        warped_cloth = self.transform(warped_cloth)
        warped_cloth_mask = Image.open(osp.join(self.pre_data_dir, "warp_mloth", os.path.basename(c_name[key])))
        warped_cloth_mask = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST) \
            (warped_cloth_mask)
        warped_cloth_mask = self.toTensor(warped_cloth_mask)
        warped_cloth = warped_cloth * warped_cloth_mask

        feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask

        down, up, left, right = mask2bbox(cm[key][0].numpy())
        ref_image = c[key][:, down:up, left:right]
        ref_image = (ref_image + 1.0) / 2.0
        ref_image = transforms.Resize((224, 224))(ref_image)
        ref_image = self.clip_normalize(ref_image)

        agnostic_name = im_name.replace('image', 'agnostic-v3.2')
        agnostic = Image.open(osp.join(self.data_path, agnostic_name))
    
        agnostic = transforms.Resize(self.crop_size, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)
        
        # load image-parse-agnostic
        parse_name = im_name.replace('image', 'image-parse-agnostic-v3.2').replace('.jpg', '.png')
        image_parse_agnostic = Image.open(osp.join(self.data_path, parse_name))
        image_parse_agnostic = transforms.Resize(self.crop_size, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
        hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)

        inpaint = feat * (1 - hands_mask) + agnostic * hands_mask   # inpaint: torch.Size([3, 512, 512]) torch.float32 tensor(-1.) tensor(1.)

        # ---------------- limb ----------------
        limb_mask = Image.open(osp.join(self.pre_data_dir, "limb_w", os.path.basename(im_name)))
        limb_mask = cv2.equalizeHist(np.array(limb_mask))
        limb_mask = Image.fromarray(limb_mask)
        limb_mask = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST)(limb_mask)
        limb_mask = self.toTensor(limb_mask)
        limb_mask = limb_mask[0].unsqueeze(0)   # (1, 512, 384)
        limb_mask = limb_mask * (1-warped_cloth_mask)

        limb = Image.open(osp.join(self.pre_data_dir, "limb_r", os.path.basename(im_name)))
        limb = transforms.Resize(self.crop_size, interpolation=2)(limb)
        limb = self.transform(limb)

        noise = torch.randn_like(limb) * 0.5
        inpaint = inpaint * (1 - limb_mask) + limb * limb_mask + noise * limb_mask  # (3, 512, 384)
        # feat = feat * (1 - limb_mask) + limb * limb_mask  # (3, 512, 384)

        result = {
            "GT": im,
            "inpaint_image": inpaint,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": ref_image,
            'warp_feat': feat,
            "file_name": self.im_names[index]
        }
        return result

    def __len__(self):
        return len(self.im_names)


if __name__ == '__main__':
    dataset = VITONDataset(dataroot="data", image_size=512, mode='test', pair_mode="unpaired", pre_data_dir="./results")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    for data in loader:
        pass
