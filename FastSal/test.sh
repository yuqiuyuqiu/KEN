#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python FastSal/test.py --pretrained ./results_mod50/KEN/model_KEN_50.pth \
                                      --file_list test_set.txt \
                                      --savedir ./output/
