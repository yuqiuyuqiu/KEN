import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

__all__ = ['FCN_ResNetEm']

def FCN_ResNetEm(pretrained=False):
    model = ResNetEm(Bottleneck, [3, 4, 6, 3])
    return model


class ResNetEm(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        super(ResNetEm, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)

        self.conv1_low = conv1x1(256, 64, stride=1)
        self.bn1_low = nn.BatchNorm2d(64)

        self.conv4_high = conv1x1(2048, 256, stride=1)
        self.bn4_high = nn.BatchNorm2d(256)
        self.encoding = EmbeddingModule(256)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv_cat = conv1x1(320, 256, stride=1)

        self.conv4_down = conv1x1(2048, 512, stride=1)
        self.conv3_down = conv1x1(1024, 256, stride=1)
        #self.conv2_down = conv1x1(512, 128, stride=1)
        #.conv1_down = conv1x1(256, 64, stride=1)

        self.head_sup4 = _FCNHead(512, num_classes)
        self.head_sup3 = _FCNHead(256, num_classes)
        #self.head_sup2 = _FCNHead(128, num_classes)
        #self.head_sup1 = _FCNHead(64, num_classes)
        self.head_sup = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Dropout(0.5),
                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        size = x.size()[2:]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        low_conv1 = self.conv1_low(x1)
        low_conv1 = self.bn1_low(low_conv1)
        low_conv1 = self.relu(low_conv1)

        high_conv4 = self.conv4_high(x4)
        high_conv4 = self.bn4_high(high_conv4)
        high_conv4 = self.relu(high_conv4)
        res = high_conv4
        high_conv4 = self.encoding(high_conv4) + res
        high_conv4 = self.bn4(high_conv4)
        high_conv4 = self.relu(high_conv4)
        high_conv4 = F.upsample(high_conv4, size=low_conv1.size()[2:], mode='bilinear', align_corners=True)

        head_sup = self.conv_cat(torch.cat((high_conv4, low_conv1), dim=1))

        conv4_down = self.conv4_down(x4)
        conv3_down = self.conv3_down(x3)
        #conv2_down = self.conv2_down(x2)
        #conv1_down = self.conv1_down(x1)

        head_sup = self.head_sup(head_sup)
        head_sup4 = self.head_sup4(conv4_down)
        head_sup3 = self.head_sup3(conv3_down)
        #head_sup2 = self.head_sup2(conv2_down)
        #head_sup1 = self.head_sup1(conv1_down)

        head_sup = F.interpolate(head_sup, size, mode='bilinear', align_corners=True)
        head_sup4 = F.interpolate(head_sup4, size, mode='bilinear', align_corners=True)
        head_sup3 = F.interpolate(head_sup3, size, mode='bilinear', align_corners=True)
        #head_sup2 = F.interpolate(head_sup2, size, mode='bilinear', align_corners=True)
        #head_sup1 = F.interpolate(head_sup1, size, mode='bilinear', align_corners=True)

        return (head_sup, head_sup4, head_sup3, )

    def forward(self, x):
        return self._forward_impl(x)


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
        # assignment weights BxNxK
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
    def __init__(self, in_channels, ncodes=26):
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                  previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
