## Delving into Universal Lesion Segmentation: Method, Dataset, and Benchmark

This paper has been published in ECCV 2022.

This code is licensed for non-commerical research purpose only.

### Introduction

Most efforts on lesion segmentation from CT slices focus on one specific lesion type. However, universal and multi-category lesion segmentation is more important because the diagnoses of different body parts are usually correlated and carried out simultaneously. The existing universal lesion segmentation methods are weakly-supervised due to the lack of pixel-level annotation data. To bring this field into the fully-supervised era, we establish a large-scale universal lesion segmentation dataset, SegLesion. We also propose a baseline method for this task. Considering that it is easy to encode CT slices owing to the limited CT scenarios, we propose a Knowledge Embedding Module (KEM) to adapt the concept of dictionary learning for this task. Specifically, KEM first learns the knowledge encoding of CT slices and then embeds the learned knowledge encoding into the deep features of a CT slice to increase the distinguishability. With KEM incorporated, a Knowledge Embedding Network (KEN) is designed for universal lesion segmentation. To extensively compare KEN to previous segmentation methods, we build a large benchmark for SegLesion. KEN achieves state-of-the-art performance and can thus serve as a strong baseline for future research.

![SegLesion](figures/size.jpg)
![SegLesion](figures/numberPie.jpg)
![SegLesion](figures/width_height.jpg)
![SegLesion](figures/location.jpg)

![KEN](figures/frame.jpg)

### Citations

If you are using the code/model provided here in a publication, please consider citing:
   
    @inproceedings{qiu2022delving,
    title={Delving into Universal Lesion Segmentation: Method, Dataset, and Benchmark},
    author={Qiu, Yu and Jing, Xu},
    journal={Proceedings of the European conference on computer vision (ECCV)},
    year={2022}
    }

### Requirements

The code is built with the following dependencies:

- Python 3.6 or higher
- CUDA 10.0 or higher
- [PyTorch](https://pytorch.org/) 1.2 or higher

### Data Preparation
The SegLesion can be downloaded:
- [SegLesion](https://drive.google.com/file/d/19JQ919DJw8CVs0xuUsLEffAugzlpbOHN/view?usp=sharing)

*The SegLesion dataset is organized into the following tree structure:*
```
dataset
│
└───images
└───masks
└───val__masks
└───test_masks
└───train_set.txt
└───val_set.txt
└───test_set.txt
'''
```


### Testing
The pretrained model of CoANet can be downloaded:
- [KEN-VGG16](https://drive.google.com/file/d/19JQ919DJw8CVs0xuUsLEffAugzlpbOHN/view?usp=sharing)
- [KEN-ResNet50](https://drive.google.com/file/d/1XR5J0voGa8ammhh2y3e0u3b-EsjbJCmE/view?usp=sharing)

Run the following scripts to test the model:
```Shell
./FastSal/test.sh [--pretrained ./results_mod50/KEN_VGG.pth]
                [--file_list test_set.txt]
                [--savedir ./output/]
```


### Evaluate

Run the following scripts to test the model:
```
python evaluate.py
```

### Training
Run the following scripts to test the model:

```Shell
./FastSal/test.sh [--cached_data_file ./duts_train.p]
                [--max_epochs 50]
                [--num_workers 4]
                [--batch_size 16]
                [--itersize 1]
                [--savedir ./results]
                [--lr_mode poly]
                [--lr 1e-4]  
```

### Contact

For any questions, please contact me via e-mail: yqiu@mail.nankai.edu.cn.
