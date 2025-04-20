import torch
import argparse
import os
from dataset_for_step1 import Dataset, DataLoader
from models.model_SCW import Network_SCW as Network1
from models.model_SLE import Network_SLE as Network2
from models.model_LTS import Network_LTS as Network3
from scripts.show import save_images, to_show, Parse_7_to_1
from scripts.refine import apply_fade_to_limb
from tqdm import tqdm

torch.manual_seed(0)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--datamode", default="test", choices=["train", "test"])
    parser.add_argument("--pair_mode", default="unpaired", choices=["paired", "unpaired"]) 
    parser.add_argument("--only_warping", action="store_true", default=False)
    parser.add_argument("--ckpt_dir", default="ckpts")
    parser.add_argument("--fine_width", type=int, default=384)
    parser.add_argument("--fine_height", type=int, default=512)
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"
    return opt

def train_network(opt, data_loader, model1, model2, model3):
    model1.eval()
    if not opt.only_warping:
        model2.eval()
        model3.eval()

    save_dir = os.path.join(opt.outdir, opt.pair_mode)
    
    dirs = ['warp_cloth', 'warp_mloth', 'parse', 'limb_w', 'limb_r', 'show'] if not opt.only_warping else ['warp_cloth', 'warp_mloth']
    dir_paths = {name: os.path.join(save_dir, name) for name in dirs}

    for path in dir_paths.values():
        os.makedirs(path, exist_ok=True)

    for step in tqdm(range(len(data_loader.data_loader))):
        inputs = data_loader.next_batch()

        # ================= step1 - warp_cloth =================
        c_name = inputs['c_name']                                  # list
        im_name = inputs['im_name']                                # list
        image = inputs['image'].cuda()                             # [b, 3, w, h]

        cloth_norm = inputs['cloth_norm'].cuda()                   # [b, 3, w, h]
        cloth = inputs['cloth'].cuda()                             # [b, 3, w, h]
        mloth = inputs['mloth'].cuda()                             # [b, 3, w, h]
        skeleton = inputs['skeleton'].cuda()                       # [b, 1, w, h]
        densepose = inputs['densepose'].cuda()                     # [b, 1, w, h]
        parse_preserve = inputs['parse_preserve'].cuda()           # [b, 1, w, h]

        warp_cloth, warp_mloth = model1(skeleton, densepose, parse_preserve, cloth, mloth, cloth_norm)

        save_images(warp_mloth, c_name, dir_paths['warp_mloth'])
        save_images(warp_cloth, c_name, dir_paths['warp_cloth'])

        if not opt.only_warping:
            # ================= step2 - parse7_t =================
            parse1_s = inputs['parse1_s'].cuda()                     # [b, 3, h, w]
            parse7_s = inputs['parse7_s'].cuda()                     # [b, 3, h, w]   
            hand_left_mask = inputs['hand_left_mask'].cuda()
            hand_right_mask = inputs['hand_right_mask'].cuda()

            # get parse_swap
            parse_swap_arr = parse7_s.clone()
            parse_swap_arr[:,3,:,:] = warp_mloth[:,0,:,:]
            parse_swap_arr[:,4,:,:] = torch.clip((skeleton * (1-warp_mloth) + hand_left_mask)[:,0,:,:], 0, 1)
            parse_swap_arr[:,5,:,:] = torch.clip((skeleton * (1-warp_mloth) + hand_right_mask)[:,0,:,:], 0, 1)

            tmp = parse_swap_arr[:,1,:,:] + parse_swap_arr[:,2,:,:] + parse_swap_arr[:,3,:,:] + parse_swap_arr[:,4,:,:] + parse_swap_arr[:,5,:,:] + parse_swap_arr[:,6,:,:]
            parse_swap_arr[:,0,:,:] =  1 - torch.clip(tmp, 0, 1)
            parse_swap = parse_swap_arr

            parse7_t = model2(skeleton, densepose, parse_swap)
        
            # ================= step3 - limb =================
            parse_limb_t = torch.clip(parse7_t[:,4,:,:] + parse7_t[:,5,:,:], 0, 1).unsqueeze(1)

            parse1_t = Parse_7_to_1(parse7_t)
            parse_limb_s = ((parse1_s==4) + (parse1_s==5)).float()
            parse_limb_t = ((parse1_t==4) + (parse1_t==5)).float()
            parse_cross_limb = torch.abs(parse_limb_s - parse_limb_t)
            parse_preserve_limb = torch.clip(parse_limb_s - parse_cross_limb, 0, 1)
            limb_input = image * parse_preserve_limb

            limb_input = apply_fade_to_limb(limb_input)

            limb_r, limb_w = model3(parse_limb_t, limb_input, skeleton, densepose)

            # save results
            save_images(Parse_7_to_1(parse7_t), im_name, dir_paths["parse"], type="parse")
            save_images(limb_w, im_name, dir_paths['limb_w'])
            save_images(limb_r, im_name, dir_paths['limb_r'])

            show = torch.cat((cloth, image, to_show(warp_mloth), warp_cloth, to_show(Parse_7_to_1(parse7_t)), to_show(limb_w), limb_r), axis=3)
            save_images(show, im_name, dir_paths['show'])


if __name__ == "__main__":
    opt = get_opt()

    print("================= load data =================")
    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)
    print("len(dataset):", len(dataset))
    print("len(data_loader):", len(data_loader.data_loader))

    print("================= load model =================")
    model1 = Network1().cuda()
    model1.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, "SCW.pth"), weights_only=True))
    if not opt.only_warping:
        model2 = Network2().cuda()
        model2.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, "SLE.pth"), weights_only=True))

        model3 = Network3().cuda()
        model3.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, "LTS.pth"), weights_only=True))
    else:
        model2 = None
        model3 = None

    print("================= start test =================")
    with torch.no_grad():
        train_network(opt, data_loader, model1, model2, model3)
    print("finished!")

