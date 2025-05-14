# [ACM MM 2024] Shape-Guided Clothing Warping for Virtual Try-On 
This repository is the official implementation of [SCW-VTON](https://dl.acm.org/doi/abs/10.1145/3664647.3680756)

> **Shape-Guided Clothing Warping for Virtual Try-On**<br>
> [Xiaoyu Han](https://xyhanhit.github.io), 
[Shunyuan Zheng](https://shunyuanzheng.github.io), 
[Zonglin Li](), 
[Chenyang Wang](), 
[Xin Sun](), 
[Quanling Meng]()

[[Paper](https://arxiv.org/abs/2504.15232)]&nbsp;
[[Checkpoints](https://pan.baidu.com/s/1-ww-bGwZQpFe-eUN1Nq-vg?pwd=4hde)]&nbsp;

![teaser](https://github.com/xyhanHIT/SCW-VTON/blob/main/assets/teaser.png)&nbsp;

## Environments
1. Clone the repository
```shell
git clone https://github.com/xyhanHIT/SCW-VTON.git
cd SCW-VTON
```

2. Install Python dependencies
```shell
conda create -n scw-vton python=3.8
conda activate scw-vton
bash environment.sh
```

3. Download the pretrained [vgg](https://drive.google.com/file/d/1rvow8jStPt8t2prDcSRlnf8yzXhrYeGo/view?usp=sharing) checkpoint and put it in `models/vgg/`

## Inference

1. Download the VITON-HD datasets from [VITON-HD](https://github.com/shadow2496/VITON-HD) or [GP-VTON](https://github.com/xiezhy6/GP-VTON).

2. Download the checkpoints from [Baidu Netdisk](https://pan.baidu.com/s/1-ww-bGwZQpFe-eUN1Nq-vg?pwd=4hde) or [Google Drive](https://drive.google.com/drive/folders/1v6pYpWOQC0cHatCCECIsof_LMNymG9YI?usp=drive_link).

3. Modify the "dataroot" and "ckpts" parameters in "test.sh" according to the downloaded files above.

4. Run the "test.sh".
```bash
bash test.sh
```

5. Evaluation
```shell
python metrics.py
```

## Results
<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/results.png">
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