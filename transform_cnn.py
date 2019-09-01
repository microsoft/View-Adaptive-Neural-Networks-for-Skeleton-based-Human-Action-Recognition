# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class VA(nn.Module):
    """The layer for transforming the skeleton to the observed viewpoints"""
    def __init__(self,num_classes = 60):
        super(VA, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.avepool = nn.MaxPool2d(7)
        self.fc = nn.Linear(6272, 6)
        self.classifier = models.resnet50(pretrained=True)
        self.init_weight()

    def forward(self, x1, maxmin):

        x = self.conv1(x1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.avepool(x)

        x = x.view(x.size(0), -1)
        trans = self.fc(x)

        temp1 = trans.cpu()
        x = _transform(x1, trans, maxmin)

        temp = x.cpu()
        x = self.classifier(x)
        return x, temp.data.numpy(), temp1.data.numpy()

    def init_weight(self):
        for layer in [self.conv1, self.conv2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    param.data.zero_()
        for layer in [self.bn1, self.bn2]:
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            layer.momentum = 0.99
            layer.eps = 1e-3

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)

# get transformation matrix
def _trans_rot(trans, rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = Variable(rot.data.new(rot.size()[:1] + (1,)).zero_())
    ones = Variable(rot.data.new(rot.size()[:1] + (1,)).fill_(1))

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 1)

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)


    rt1 = torch.stack((ones, zeros, zeros, trans[:,0:1]), dim = -1)
    rt2 = torch.stack((zeros, ones, zeros, trans[:,1:2]), dim = -1)
    rt3 = torch.stack((zeros, zeros, ones, trans[:,2:3]), dim = -1)
    trans = torch.cat((rt1, rt2, rt3), dim = 1)

    return trans, rot

# transform skeleton
def _transform(x, mat, maxmin):
    rot = mat[:,0:3]
    trans = mat[:,3:6]

    x = x.contiguous().view(-1, x.size()[1] , x.size()[2] * x.size()[3])

    max_val, min_val = maxmin[:,0], maxmin[:,1]
    max_val, min_val = max_val.contiguous().view(-1,1), min_val.contiguous().view(-1,1)
    max_val, min_val = max_val.repeat(1,3), min_val.repeat(1,3)
    trans, rot = _trans_rot(trans, rot)

    x1 = torch.matmul(rot,x)
    min_val1 = torch.cat((min_val, Variable(min_val.data.new(min_val.size()[0], 1).fill_(1))), dim=-1)
    min_val1 = min_val1.unsqueeze(-1)
    min_val1 = torch.matmul(trans, min_val1)

    min_val = torch.div( torch.add(torch.matmul(rot, min_val1).squeeze(-1), - min_val), torch.add(max_val, - min_val))

    min_val = min_val.mul_(255)
    x = torch.add(x1, min_val.unsqueeze(-1))

    x = x.contiguous().view(-1,3, 224,224)

    return x


