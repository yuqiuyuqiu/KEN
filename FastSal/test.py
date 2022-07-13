import sys
sys.path.insert(0, '.')

import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from IOUEval import iouEval
from models import Em_ResNet as net


def get_mean_set(args):
    # for DUTS training dataset
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return mean, std

@torch.no_grad()
def validateModel(args, model, image_list, label_list):
    mean, std = get_mean_set(args)
    iou_eval_val = iouEval(args.classes)
    for idx in range(len(image_list)):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        label = label.astype(dtype=np.bool)
        img = image.astype(np.float32)
        img = img[:, :, ::-1]
        img = img / 255
        img -= mean
        img /= std

        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(img, (args.inWidth, args.inHeight))

        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        img_variable = Variable(img_tensor)
        if args.gpu:
            img_variable = img_variable.cuda()

        start_time = time.time()
        img_out = model(img_variable)[0]
        torch.cuda.synchronize()
        diff_time = time.time() - start_time
        print('Segmentation for {}/{} takes {:.3f}s per image'.format(idx, len(image_list), diff_time))

        class_numpy = img_out[0].max(0)[1].data.cpu().numpy()
        iou_eval_val.add_batch(class_numpy, label)

        out_numpy = (class_numpy * 255).astype(np.uint8)
        name = image_list[idx].split('/')[-1]
        if not osp.isdir(osp.join(args.savedir, args.model_name)):
            os.mkdir(osp.join(args.savedir, args.model_name))
        cv2.imwrite(osp.join(args.savedir, args.model_name, name[:-4] + '.png'), out_numpy)

    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()
    print('Overall Acc (Val): %.4f\t mIOU (Val): %.4f' % (overall_acc, mIOU))
    return mIOU


def main(args):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, args.file_list)) as textFile:
        for line in textFile:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + 'images' + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + 'masks' + '/' + line_arr[0].strip())

    model = net.FCN_ResNetEm()

    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained)
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        new_keys.append(key)#.replace('module.', '')
        #new_keys.append(key[7:])
        new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict)
    #model.load_state_dict(state_dict)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.mkdir(args.savedir)

    mIOU = validateModel(args, model, image_list, label_list)
    print(mIOU)
    return mIOU



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./dataset", help='Data directory')
    parser.add_argument('--file_list', default="test_set.txt", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--savedir', default='./output/', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default=None, help='Pretrained model')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset')
    parser.add_argument('--model_name', default='KEN', help='Model name')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
