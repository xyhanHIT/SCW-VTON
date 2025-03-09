<div align="center">

# Shape-Guided Clothing Warping for Virtual Try-On

[Xiaoyu Han](https://xyhanhit.github.io), 
[Shunyuan Zheng](https://shunyuanzheng.github.io), 
[Zonglin Li](), 
[Chenyang Wang](), 
[Xin Sun](), 
[Quanling Meng]()<sup>*</sup>, 


<p>Harbin Institute of Technology

#### [[Paper]](https://dl.acm.org/doi/abs/10.1145/3664647.3680756) · [[Checkpoints]]()

</div>

![image](images/teaser.png)

## Pipeline
![image]()

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

## Sample Try-on Results
  
![image]()

![image]()

## License
The use of this code is restricted to non-commercial research and educational purposes.

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