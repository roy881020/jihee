import os
import torch
import torch.nn as nn
import torchvision
import pretrainedmodels

#Hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LEARNING_RATE = 0.001

#training data load



class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dr = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dr(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, NumLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        #self.layer1 = call pretrainedmodels
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2) #have to modify inplane size
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #modify inplane
        self.layer5 = self._make_layer(block, 512, layers[3], stride=1) #modify inplane
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=True) #check function name exact
        self.layer6 = self._make_pred_layer(Classifier_Module, diliation_series=1,padding_series=1)



    def _make_layer(self,block,planes,blocks,stride=1,dilation__=1):
        donwsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation__=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes, planes, dilation__=dilation__))

        return nn.Sequential(*layers)
    def _make_pred_layer(self, block, diliation_series, padding_series):
        return block(diliation_series, padding_series)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = self.layer6(x)

        return x

class Classifier_Module(nn.Module):

    def __init__(self, diliation_series, padding_seires):
        super(Classifier_Module, self).__init__()


#for use pre-trained model(VGG16)
model_name = 'vgg16'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

def ResNet_sb():
    model = ResNet(ResBlock,)