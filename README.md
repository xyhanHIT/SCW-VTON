# [ACM MM 2024] Shape-Guided Clothing Warping for Virtual Try-On 
This repository is the official implementation of [SCW-VTON](https://dl.acm.org/doi/abs/10.1145/3664647.3680756)

> **Shape-Guided Clothing Warping for Virtual Try-On**<br>
> [Xiaoyu Han](https://xyhanhit.github.io), 
[Shunyuan Zheng](https://shunyuanzheng.github.io), 
[Zonglin Li](), 
[Chenyang Wang](), 
[Xin Sun](), 
[Quanling Meng]()

[[Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3680756)]&nbsp;
[[Checkpoints]()]&nbsp;

![teaser](https://github.com/xyhanHIT/SCW-VTON/blob/main/images/teaser.png)&nbsp;

<!-- ## TODO List
- [x] ~~Inference code~~
- [x] ~~Release model weights~~
- [x] ~~Training code~~ -->

## TODO List
- Inference code for clothing warping
- Inference code for try-on synthesis
- Release mode weights

## Environments
```bash
git clone https://github.com/xyhanHIT/SCW-VTON.git
cd SCW-VTON
conda create -n scw-vton python=3.8
conda activate scw-vton

# install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install opencv-python
pip install scikit-image
```

## Data preparation
### VITON
You can download the VITON-HD datasets from [VITON](https://github.com/xthan/VITON).

You also need to download the data about hands from [here](https://drive.google.com/file/d/1VbzXS6vYumRoUaVp0PRXvB_1d54aqxM6/view?usp=drive_link).

### VITON-HD
You can download the VITON-HD datasets from [VITON-HD](https://github.com/shadow2496/VITON-HD) or [GP-VTON](https://github.com/xiezhy6/GP-VTON)

## Inference
1. Download the checkpoints from [here]().

2. Get [VITON dataset](https://github.com/xthan/VITON).

3. Run the "test.py".
```bash
python test.py
```

## Results
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="images/results.png">
</div>

## Our Team's Researches
- **[[ACM MM'22] PL-VTON](https://github.com/xyhanHIT/PL-VTON)** - Progressive Limb-Aware Virtual Try-On
- **[[IEEE TMM'23] PL-VTONv2](https://github.com/aipixel/PL-VTONv2)** - Limb-Aware Virtual Try-On Network With Progressive Clothing Warping
- **[[ACM MM'24] SCW-VTON](https://github.com/xyhanHIT/SCW-VTON)** - Shape-Guided Clothing Warping for Virtual Try-On

## Acknowledgements
Our code references the implementation of [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On). Thanks for their awesome works.

## Citation
If you use our code or models, please cite with:
```bibtex
@inproceedings{han2024shape,
  title={Shape-Guided Clothing Warping for Virtual Try-On},
  author={Han, Xiaoyu and Zheng, Shunyuan and Li, Zonglin and Wang, Chenyang and Sun, Xin and Meng, Quanling},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2593--2602},
  year={2024}
}
```