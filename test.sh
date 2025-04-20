#!/bin/bash
python test_for_step1.py --outdir ./results --dataroot ./data --ckpt_dir ./ckpts --pair_mode unpaired
python test_for_step2.py --outdir ./results --dataroot ./data --ckpt_dir ./ckpts --pair_mode unpaired --plms



