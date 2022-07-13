#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python FastSal/train_vgg.py --cached_data_file ./duts_train.p \
                                         --max_epochs 50 \
                                         --num_workers 4 \
                                         --batch_size 16 \
                                         --itersize 1 \
                                         --savedir ./results \
                                         --lr_mode poly \
                                         --lr 1e-4
