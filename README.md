# EDSR Tensorflow Implementation

An implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf) written in tensorflow.

## Requirements

 - Tensorflow==1.10.1
 - scipy
 - tqdm
 - argparse

## Installation

 `pip install -r requirements.txt`

## Prepare

 - Put your imgs into ./data_dir
 - Then run "python2 cut_img.py" (it will create directory "./data_dir_crop_y" to save cut-imgs of 100*100 as label y)
 - Then run "python2 create_blur_img.py" (it will create directory "./data_dir_crop_x" to save resize-imgs of 50*50 as input x)

## Training

 - python2 train.py --dataset data_dir 

## Test
 - Put your test imgs into ./data
 - Then run "python2 test.py --image a.jpg --savedir saved_models" (it will test all imgs in ./data ,use saved_models)
