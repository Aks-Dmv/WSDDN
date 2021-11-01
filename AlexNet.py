import numpy as np

import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
import torchvision.transforms as transforms

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LocalizerAlexNet(nn.Module):
    def __init__(self, weight_init=True, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.pre_trained_alexnet = models.alexnet(pretrained=weight_init)
        self.pre_trained_alexnet.features[-1] = Identity()
        layer_i = 0
        kernel_sizes = [3,1,1]
        output_sizes = [256,256,20]
        cnn_cntr = 0
        for layer in self.pre_trained_alexnet.classifier:
            if isinstance(layer, nn.Linear):
                self.pre_trained_alexnet.classifier[layer_i] = nn.Conv2d(256, output_sizes[cnn_cntr], kernel_size=kernel_sizes[cnn_cntr])
                nn.init.xavier_normal_(self.pre_trained_alexnet.classifier[layer_i].weight)
                nn.init.zeros_(self.pre_trained_alexnet.classifier[layer_i].bias)
                cnn_cntr+=1
            if isinstance(layer, nn.Dropout):
                self.pre_trained_alexnet.classifier[layer_i] = Identity()
            layer_i+=1


    def forward(self, x):
        #TODO: Define forward pass
        x = self.pre_trained_alexnet.features(x)
        x = self.pre_trained_alexnet.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, weight_init=True, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model
        self.model = LocalizerAlexNet(weight_init)
        # self.resize0 = transforms.Resize((64, 64))
        self.resize1 = transforms.Resize((128, 128))
        self.resize2 = transforms.Resize((256, 256))


    def forward(self, x):
        #TODO: Define fwd pass
        # x0 = self.resize0(x)
        x1 = self.resize1(x)
        x2 = self.resize2(x)
        
        
        x = self.model(x)
        # x0 = self.model(x0)
        x1 = self.model(x1)
        x2 = self.model(x2)

        return x, x1, x2


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #TODO: Initialize weights correctly based on whethet it is pretrained or not
    print("pretrained is ", pretrained)
    model = LocalizerAlexNet( pretrained, **kwargs)


    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #TODO: Ignore for now until instructed
    model = LocalizerAlexNetRobust(pretrained, **kwargs)
    

    return model