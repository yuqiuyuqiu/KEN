import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class VGG16Em(nn.Module):
    def __init__(self, classes=2):
        super(VGG16Em, self).__init__()
        self.conv1_1 = ConvBNReLU(3, 64)
        self.conv1_2 = ConvBNReLU(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNReLU(64, 128)
        self.conv2_2 = ConvBNReLU(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBNReLU(128, 256)
        self.conv3_2 = ConvBNReLU(256, 256)
        self.conv3_3 = ConvBNReLU(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBNReLU(256, 512)
        self.conv4_2 = ConvBNReLU(512, 512)
        self.conv4_3 = ConvBNReLU(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv5_1 = ConvBNReLU(512, 512, pad=2, dilation=2)
        self.conv5_2 = ConvBNReLU(512, 512, pad=2, dilation=2)
        self.conv5_3 = ConvBNReLU(512, 512, pad=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv6_1 = ConvBNReLU(512, 512, pad=4, dilation=4)
        self.conv6_2 = ConvBNReLU(512, 1024, ksize=1, stride=1, pad=0)

        self.conv3_low = conv1x1(256, 64, stride=1)
        self.bn3_low = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv6_high = conv1x1(1024, 256, stride=1)
        self.bn6_high = nn.BatchNorm2d(256)
        self.encoding = EmbeddingModule(256)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv_cat = ConvBNReLU(320, 256)

        self.head_sup = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, classes, kernel_size=1, stride=1))

        self.conv6_down = conv1x1(1024, 256, stride=1)
        self.conv5_down = conv1x1(512, 256, stride=1)
        self.conv4_down = conv1x1(512, 128, stride=1)

        self.head_sup6 = _FCNHead(256, classes)
        self.head_sup5 = _FCNHead(256, classes)
        self.head_sup4 = _FCNHead(128, classes)


    def forward(self, input):
        size = input.size()[2:]

        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        pool5 = self.pool5(conv5_3)

        conv6_1 = self.conv6_1(pool5)
        conv6_2 = self.conv6_2(conv6_1)

        low_conv3 = self.conv3_low(conv3_3)
        low_conv3 = self.bn3_low(low_conv3)
        low_conv3 = self.relu(low_conv3)

        high_conv6 = self.conv6_high(conv6_2)
        high_conv6 = self.bn6_high(high_conv6)
        high_conv6 = self.relu(high_conv6)
        res = high_conv6
        high_conv6 = self.encoding(high_conv6) + res
        high_conv6 = self.bn6(high_conv6)
        high_conv6 = self.relu(high_conv6)
        high_conv6 = F.upsample(high_conv6, size=low_conv3.size()[2:], mode='bilinear', align_corners=True)

        head_sup = self.conv_cat(torch.cat((high_conv6, low_conv3), dim=1))

        conv6_down = self.conv6_down(conv6_2)
        conv5_down = self.conv5_down(conv5_3)
        conv4_down = self.conv4_down(conv4_3)

        head_sup = self.head_sup(head_sup)
        head_sup6 = self.head_sup6(conv6_down)
        head_sup5 = self.head_sup5(conv5_down)
        head_sup4 = self.head_sup4(conv4_down)

        head_sup = F.interpolate(head_sup, size, mode='bilinear', align_corners=True)
        head_sup6 = F.interpolate(head_sup6, size, mode='bilinear', align_corners=True)
        head_sup5 = F.interpolate(head_sup5, size, mode='bilinear', align_corners=True)
        head_sup4 = F.interpolate(head_sup4, size, mode='bilinear', align_corners=True)

        return (head_sup, head_sup6, head_sup5, head_sup4, )


class Encoding(nn.Module):
    def __init__(self, D, K, encoding):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.reset_params()
        self.encoding = encoding

        self.fc = nn.Sequential(
            nn.Linear(D, D),
            nn.Sigmoid())

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, N, D, K = X.size(0), reduce(lambda x, y: x*y, X.shape[2:]), self.D, self.K
        if X.dim() == 3:
            # BxDxN => BxNxD
            I = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            I = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxKxD
        A = F.softmax(self.scale.view(1, 1, K, D) * (I.unsqueeze(2).expand(B, N, K, D) \
                - self.codewords.unsqueeze(0).unsqueeze(0)).pow(2), dim=2)
        if not self.encoding: # for the embedding module
            # aggregate (output: EM = BxKxD)
            EM = (A * (I.unsqueeze(2).expand(B, N, K, D) \
                    - self.codewords.unsqueeze(0).unsqueeze(0))).sum(1)
            EM = EM.mean(dim=1, keepdim=False)
            gamma = self.fc(EM)

            # aggregate (output: E = BxNxD)
            E = (A * (I.unsqueeze(2).expand(B, N, K, D) \
                    - self.codewords.unsqueeze(0).unsqueeze(0))).sum(2)
            # reshape to original size
            if X.dim() == 3:
                # BxNxD => BxDxN
                E = E.transpose(1, 2).contiguous()
                y = gamma.view(B, D, 1)
                E = F.relu_(E + E * y)
            elif X.dim() == 4:
                # Bx(HW)xD => BxDxHxW
                E = E.transpose(1, 2).view(B, D, X.shape[2], X.shape[3]).contiguous()
                y = gamma.view(B, D, 1, 1)
                E = F.relu_(E + E * y)
            else:
                raise RuntimeError('Encoding Layer unknown input dims!')
        else: # for the encoding module
            # aggregate (output: BxKxD)
            E = (A * (I.unsqueeze(2).expand(B, N, K, D) \
                    - self.codewords.unsqueeze(0).unsqueeze(0))).sum(1)
        return E


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels, ncodes=24):
        super(EmbeddingModule, self).__init__()
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes, encoding=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.conv(torch.cat((x, self.encoding(x)), dim=1))
        return output


class _FCNHead(nn.Module):
    def __init__(self, inplanes, planes):
        super(_FCNHead, self).__init__()
        inter_planes = inplanes // 4
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, inter_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_planes, planes, 1),
        )

    def forward(self, x):
        return self.block(x)

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
            dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)