#!/bin/bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install matplotlib
pip install tqdm

pip install omegaconf
pip install einops
pip install pytorch-lightning==1.4.2
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install transformers==4.33.2
pip install kornia

pip install lpips
pip install scikit-image
pip install torch_fidelity