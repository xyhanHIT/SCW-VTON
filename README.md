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

## Environment
python 3.8

torch 2.0.1+cu118

torchvision 0.15.2+cu118

## Dataset
For the dataset, please refer to [VITON](https://github.com/xthan/VITON).

You also need to download the data about hands from [here](https://drive.google.com/file/d/1VbzXS6vYumRoUaVp0PRXvB_1d54aqxM6/view?usp=drive_link).

## Inference
1. Download the checkpoints from [here]().

2. Get [VITON dataset](https://github.com/xthan/VITON).

3. Run the "test.py".
```bash
python test.py
```

## License
The use of this code is restricted to non-commercial research and educational purposes.

## Related Works
This work builds upon our previous research:
- **[PL-VTON](https://github.com/xyhanHIT/PL-VTON)**: Progressive Limb-Aware Virtual Try-On (ACM MM'22)
- **[PL-VTONv2](https://github.com/aipixel/PL-VTONv2)**: Limb-Aware Virtual Try-On Network With Progressive Clothing Warping (IEEE TMM'23)


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