# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import deeplab_network



class backboneModel(nn.Module):
    def __init__(self, model):
        super(backboneModel, self).__init__()

        self.f = []
        for name, module in model.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            self.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.fc = nn.Linear(2048, 200)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        x = self.fc(feature)
        return x

def deeplab_backbone(path='pretrain_test_tiny-imagenet-200-0002.pth'):
    model = deeplab_network.deeplabv3_resnet50(num_classes=args.num_classes, output_stride=args.output_stride, pretrained_backbone=False)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model = model.backbone
    # backbone_resnet = backboneModel(model)
    print(model)
    return model



