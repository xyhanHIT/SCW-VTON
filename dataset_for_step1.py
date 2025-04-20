import torch
import torch.utils.data as data
from PIL import Image
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def ParseOneHot(x, num_class=None):
    h, w = x.shape
    x = x.reshape(-1)
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((x.shape[0], num_class))
    ohx[range(x.shape[0]), x] = 1
    ohx = ohx.reshape(h,w, ohx.shape[1])
    return ohx.transpose(2,0,1)

def ColorNorm(cloth, mask):
    res = cloth * np.expand_dims(mask, axis=2)
    item_num = np.sum(mask)
    for i in range(3):
        item_sum = np.sum(res[:,:,i])
        res[:,:,i] = res[:,:,i] - item_sum / item_num + mask
        res = np.clip(res, 0, 1)
    return res

def ParseFine(parse):
    parse_background = (parse==0)
    parse_hair = (parse==2)
    parse_neck = (parse==10)
    parse_cloth1 = (parse==5)
    parse_cloth2 = (parse==6)
    parse_cloth3 = (parse==7)
    parse_low_cloth1 = (parse==8)
    parse_low_cloth2 = (parse==9)
    # parse_cloth4 = (parse==10)
    parse_cloth5 = (parse==11)
    parse_low_cloth3 = (parse==12)
    parse_face = (parse==13)
    parse_left_hand = (parse==14)
    parse_right_hand = (parse==15)
    parse_leg1 = (parse==16)    
    parse_leg2 = (parse==17)   
    parse_shoe1 = (parse==18)    
    parse_shoe2 = (parse==19) 

    # -------------
    # 0：bg
    # 1：hair
    # 2：face
    # 3：top / mask
    # 4：left arm
    # 5：right arm
    # 6：bottom
    # -------------
    parse = (parse_background + parse_neck) * 0 + \
        parse_hair * 1 + \
        parse_face * 2 + \
        (parse_cloth1 + parse_cloth2 + parse_cloth3 + parse_cloth5) * 3 + \
        parse_left_hand * 4 + \
        parse_right_hand * 5 + \
        (parse_low_cloth1 + parse_low_cloth2 + parse_low_cloth3 + parse_leg1 + parse_leg2 + parse_shoe1 + parse_shoe2) * 6
        
    return parse.astype("uint8")


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.pair_mode = opt.pair_mode
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                if self.pair_mode == "paired":
                    c_name = im_name
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def __getitem__(self, index):
        # cloth name
        c_name = self.c_names[index]
        # image name
        im_name = self.im_names[index]
        # cloth
        cloth_arr = np.array(Image.open(osp.join(self.data_path, 'cloth', c_name)).resize([self.fine_width,self.fine_height])).astype(np.float32)/255
        cloth = torch.from_numpy(cloth_arr.astype(np.float32).transpose(2,0,1))
        # mloth
        mloth_arr = np.array(Image.open(osp.join(self.data_path, 'cloth-mask', c_name)).resize([self.fine_width,self.fine_height])).astype(np.float32)/255
        _, mloth_arr = cv2.threshold(mloth_arr, 0.5, 1, cv2.THRESH_BINARY)
        mloth = torch.from_numpy(mloth_arr.astype(np.float32)).unsqueeze(0)
        # cloth_norm
        cloth_norm_arr = ColorNorm(cloth_arr, mloth_arr)
        cloth_norm = torch.from_numpy(cloth_norm_arr.astype(np.float32).transpose(2,0,1))
        # parse
        parse1_s_arr = np.array(Image.open(osp.join(self.data_path, 'image-parse-v3', im_name.replace('.jpg', '.png'))).resize([self.fine_width,self.fine_height])).astype('uint8')
        parse1_s_arr = ParseFine(parse1_s_arr)  # [0-19] -> [0-6]
        parse7_s_arr = ParseOneHot(parse1_s_arr, 7)
        parse1_s = torch.from_numpy(parse1_s_arr.astype(np.float32)).unsqueeze(0)
        parse7_s = torch.from_numpy(parse7_s_arr.astype(np.float32))
        # image
        image_arr = np.array(Image.open(osp.join(self.data_path, 'image', im_name)).resize([self.fine_width,self.fine_height])).astype(np.float32)/255
        image = torch.from_numpy(image_arr.transpose(2,0,1))
        # hand
        densepose_arr = np.array(Image.open(osp.join(self.data_path, 'image-densepose', im_name)).convert("L").resize([self.fine_width,self.fine_height])).astype(np.float32)
        densepose = torch.from_numpy(densepose_arr / 255).unsqueeze(0)
        # hand_mask
        hand_left_mask_arr = (densepose_arr==84).astype(np.float32)
        hand_right_mask_arr = (densepose_arr==92).astype(np.float32)
        hand_left_mask = torch.from_numpy(hand_left_mask_arr).unsqueeze(0)
        hand_right_mask = torch.from_numpy(hand_right_mask_arr).unsqueeze(0)
        # skeleton
        skeleton_arr = np.array(Image.open(osp.join(self.data_path, 'openpose_img', im_name.replace(".jpg", "_rendered.png"))).convert("L").resize([self.fine_width,self.fine_height])).astype(np.float32)/255
        skeleton = torch.from_numpy(skeleton_arr.astype(np.float32)).unsqueeze(0)
        # parse_preserve
        parse_preserve_arr = (parse1_s_arr==1).astype('uint8')+(parse1_s_arr==2).astype('uint8')+(parse1_s_arr==6).astype('uint8')
        parse_preserve = torch.from_numpy(parse_preserve_arr).unsqueeze(0)

        result = {
            'c_name':               c_name,                 # list
            'im_name':              im_name,                # list
            'image':                image,                  # [b, 3, h, w] 

            'cloth_norm':           cloth_norm,
            'cloth':                cloth,                  # [b, 3, h, w]
            'mloth':                mloth,                  # [b, 1, h, w]
            'skeleton':             skeleton,               # [b, 1, h, w]
            'densepose':            densepose,              # [b, 1, h, w]
            'parse_preserve':       parse_preserve,         # [b, 1, h, w]

            "parse1_s":             parse1_s,               # [b, 3, h, w]
            "parse7_s":             parse7_s,               # [b, 7, h, w]
            "hand_left_mask":       hand_left_mask,         # [b, 1, h, w]    
            "hand_right_mask":      hand_right_mask,        # [b, 1, h, w]

        }                

        return result

    def __len__(self):
        return len(self.im_names)

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.workers, pin_memory=True)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == "__main__":
    print("Check the dataset...")

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test", choices=["train", "test"])
    parser.add_argument("--pair_mode", default="unpaired", choices=["paired", "unpaired"])  
    parser.add_argument("--fine_width", type=int, default=384)
    parser.add_argument("--fine_height", type=int, default=512)
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"
    
    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)
   
    for step, inputs in enumerate(data_loader.data_loader):
        c_name = inputs['c_name']                                  # list
        im_name = inputs['im_name']                                # list
        image = inputs['image'].cuda()                             # [b, 3, w, h]