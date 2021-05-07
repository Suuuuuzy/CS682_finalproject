# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import deeplab_network
from .backbone import resnet




class backboneModel(nn.Module):
    def __init__(self, model):
        super(backboneModel, self).__init__()

        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.fc = nn.Linear(2048, 200)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        x = self.fc(feature)
        return x

def deeplab_backbone(path='pretrain_test_tiny-imagenet-200-0002.pth',num_classes = 3,output_stride=16 ):
    model = deeplab_network.deeplabv3_resnet50(num_classes, output_stride, pretrained_backbone=False)
    state_dict = torch.load(path)['state_dict']
    model.load_state_dict(state_dict)
    model = model.backbone
    # backbone_resnet = backboneModel(model)
    print(model)

    replace_stride_with_dilation=[False, False, True]
    aspp_dilate = [6, 12, 18]
    backbone = resnet.resnet50(
    pretrained=False,
    replace_stride_with_dilation=replace_stride_with_dilation)
    backbone.load_state_dict(model.state_dict(), strict=Fasle)
    print(backbone)

    return backbone



