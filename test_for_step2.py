import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from dataset_for_step2 import VITONDataset
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.refine import refine_res

from torchvision.transforms import Resize

def load_model_from_config(config, ckpt_dir, verbose=False):
    print(f"Loading model from {ckpt_dir}")
    pl_sd = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    # sd = pl_sd["state_dict"]
    sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="results"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpts",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        help="unpaired mode means trying on different clothing than the original",
        default="unpaired",
        choices=["paired", "unpaired"]
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        help="path to dataroot of the dataset",
        default="data"
    )
    parser.add_argument(
        "--pre_data_dir",
        type=str,
        help="path to warped-clothing, parse, and limb",
        default="results"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="which gpu to use",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        default=True,
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/viton.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda:{}".format(opt.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt_dir}/try_on.pth")
    # model = model.to(device)

    dataset = VITONDataset(opt.dataroot, opt.H, mode='test', pair_mode=opt.pair_mode, pre_data_dir=opt.pre_data_dir)
    loader = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False, num_workers=4, pin_memory=True)

    sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)

    result_path = os.path.join(opt.outdir, opt.pair_mode, "try_on")

    os.makedirs(result_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    iterator = tqdm(loader, desc='Test Dataset', total=len(loader))
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data in iterator:
                    mask_tensor = data['inpaint_mask']
                    inpaint_image = data['inpaint_image']
                    ref_tensor = data['ref_imgs']
                    feat_tensor = data['warp_feat']
                    image_tensor = data['GT']
                    # filename = data['file_name']

                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
                    test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
                    feat_tensor = feat_tensor.to(device)
                    ref_tensor = ref_tensor.to(device)

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                        uc = uc.repeat(ref_tensor.size(0), 1, 1)
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = model.proj_out(c)

                    # z_gt = model.encode_first_stage(image_tensor.to(device))
                    # z_gt = model.get_first_stage_encoding(z_gt).detach()

                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])

                    warp_feat = model.encode_first_stage(feat_tensor)
                    warp_feat = model.get_first_stage_encoding(warp_feat).detach()

                    ts = torch.full((1,), 999, device=device, dtype=torch.long)
                    start_code = model.q_sample(warp_feat, ts)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code,
                                                     test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    x_source = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)
                    x_result = x_checked_image_torch * (1 - mask_tensor) + mask_tensor * x_source

                    resize = transforms.Resize((opt.H, int(opt.H / 256 * 192)))

                    # save images
                    for i, x_sample in enumerate(x_result):
                        filename = data['file_name'][i]
                        save_x = resize(x_sample)
                        save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                        save_x = refine_res(save_x, filename, opt.dataroot, mode='test', pair_mode=opt.pair_mode, pre_data_dir=opt.pre_data_dir)
                        img = Image.fromarray(save_x.astype(np.uint8))
                        img.save(os.path.join(result_path, filename[:-4] + ".png"), quality=95)

    print("finished!")


if __name__ == "__main__":
    main()
