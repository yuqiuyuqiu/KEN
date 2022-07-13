import sys
sys.path.insert(0, '.')

import loadData as ld
import os
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
from parallel import DataParallelModel, DataParallelCriterion
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import torch.optim.lr_scheduler
from collections import OrderedDict
from torch.nn.parallel.scatter_gather import gather
import torch.nn as nn
import torch.nn.functional as F

from models import Em_ResNet as net


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_label=255):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, *inputs):
        pred, pred1, pred2, target = tuple(inputs)
        loss = self.loss(F.log_softmax(pred, 1), target)
        loss1 = self.loss(F.log_softmax(pred1, 1), target)
        loss2 = self.loss(F.log_softmax(pred2, 1), target)
        return 1.0*loss + 0.4*loss1 + 0.4*loss2


class Averagvalue(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()
    iou_eval_val = iouEval(args.classes)
    epoch_loss = []

    total_batches = len(val_loader)
    for iter, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        torch.cuda.synchronize()
        time_taken = time.time() - start_time

        if not args.onGPU or torch.cuda.device_count() <= 1:
            pred, pred1, pred2 = tuple(output)
            loss = criterion(pred, pred1, pred2, target_var)
        else:
            loss = criterion(output, target_var) # criterion->CrossEntropyLoss()
        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]

        iou_eval_val.add_batch(output.max(1)[1].data.cpu().numpy(), target_var.data.cpu().numpy())

        print('[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU

def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()
    iou_eval_train = iouEval(args.classes)
    epoch_loss = []
    counter = 0

    total_batches = len(train_loader)

    for iter, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        if not args.onGPU or torch.cuda.device_count() <= 1:
            pred, pred1, pred2 = tuple(output)
            loss = criterion(pred, pred1, pred2, target_var)
        else:
            loss = criterion(output, target_var) # criterion->CrossEntropyLoss()
        counter += 1
        loss = loss / args.itersize
        loss.backward()

        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        epoch_loss.append(loss.data.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]
        iou_eval_train.add_batch(output.max(1)[1].data.cpu().numpy(), target_var.data.cpu().numpy())

        if iter % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.max_epochs, iter, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_train.get_metric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU, lr

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = max_batches*epoch + iter
        max_iter = max_batches*args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def trainValidateSegmentation(args):
    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    # load the model
    model = net.FCN_ResNetEm()#args.classes
    if not os.path.isdir(os.path.join(args.savedir + '_mod'+ str(args.max_epochs))):
        os.mkdir(args.savedir + '_mod'+ str(args.max_epochs))
    saveDir = args.savedir + '_mod' + str(args.max_epochs) + '/' + args.model_name
    # create the directory if not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    if args.onGPU and torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
        model = DataParallelModel(model)
    if args.onGPU:
        model = model.cuda()

    state_dict = torch.load('./pretrainModel/resnet50_v2.pth')
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):#['state_dict']
        #new_keys.append(key.replace('model.', '').replace('module.', ''))
        new_keys.append('module.' + key)
        new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict)#, strict=False

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.onGPU:
        weight = weight.cuda()

    criteria = CrossEntropyLoss2d(weight=weight, ignore_index=255)
    if args.onGPU and torch.cuda.device_count() > 1 :
        criteria = DataParallelCriterion(criteria)
    if args.onGPU:
        criteria = criteria.cuda()

    data['mean'] = np.array([0.485, 0.456, 0.406], dtype=np.float32)#RGB
    data['std'] = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    print('Data statistics:')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7./224.*args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    max_batches = len(trainLoader) + len(trainLoader_scale1) + len(trainLoader_scale2) + len(trainLoader_scale3)

    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            '''
            new_keys = []
            new_values = []
            for key, value in zip(checkpoint['state_dict'].keys(), checkpoint['state_dict'].values()):
                new_keys.append(key.replace('context_path.', ''))
                new_values.append(value)
            new_dict = OrderedDict(list(zip(new_keys, new_values)))
            model.load_state_dict(new_dict)
            '''
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = os.path.join(saveDir, 'trainValLog_'+args.model_name+'.txt')
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s\t%s\t%s\t%s\tlr" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)'))

    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    maxmIOU = 0
    maxEpoch = 0
    for epoch in range(start_epoch, args.max_epochs):
        # train for one epoch
        cur_iter = 0

        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr, lr = \
                train(args, trainLoader, model, criteria, optimizer, epoch, max_batches, cur_iter)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'iouTr': mIOU_tr,
            'iouVal': mIOU_val,
            'lr': lr
        }, os.path.join(saveDir, 'checkpoint_' + args.model_name + '.pth.tar'))

        # save the model also
        model_file_name = os.path.join(saveDir, 'model_' + args.model_name + '_' + str(epoch + 1) + '.pth')
        torch.save(model.state_dict(), model_file_name)

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f\n" \
                % (epoch + 1, lossTr, lossVal, mIOU_tr, mIOU_val))
        if mIOU_val >= maxmIOU:
            maxmIOU = mIOU_val
            maxEpoch = epoch + 1
        torch.cuda.empty_cache()
    logger.flush()
    logger.close()
    return maxEpoch, maxmIOU


def main(args):
    saveDir = args.savedir+ '_mod' + str(args.max_epochs) + '/' + args.model_name

    maxEpoch, maxmIOU = trainValidateSegmentation(args)
    with open(os.path.join(saveDir, 'modelBest_' + args.model_name + '.txt'), 'a+') as log:
        log.write("\n%s\t maxEpoch: %d\t maxmIOU: %.4f" \
                % (args.model_name, maxEpoch, maxmIOU))

    print(maxEpoch, maxmIOU)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./dataset", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')  #224 336 384
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--itersize', type=int, default=1, help='iter size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--classes', type=int, default=2, help='No. of classes in the dataset')
    parser.add_argument('--cached_data_file', default='duts_train.p', help='Cached file name')
    parser.add_argument('--model_name', default='KEN_ResNet', help='Model name')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
