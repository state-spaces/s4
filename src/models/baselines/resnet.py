""" 2D ResNet baseline, mostly used to test Pathfinder currently. """

import torch.nn as nn
import math
import torchvision.models as models


class Resnet18CelebA(nn.Module):

    def __init__(
            self,
            d_output,
            **kwargs,
    ):
        super().__init__()
        if 'l_output' in kwargs and kwargs['l_output'] > 1:
            d_output = kwargs['l_output']

        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, d_output)

    def forward(self, x, *args, **kwargs):
        # BSC -> BCS
        x = x.transpose(1, 2)
        # BCS -> BCHW
        x = x.view(x.shape[0], 3, 178, 218)
        return self.resnet.forward(x)

class ResnetSquare(nn.Module):

    def __init__(
            self,
            d_input,
            variant='18',
    ):
        super().__init__()

        self.d_input = d_input
        self.resnet = {
            '18': models.resnet18,
            '34': models.resnet34,
            '50': models.resnet50,
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            'wrn': models.wide_resnet50_2,
        }[variant](pretrained=False)
        self.resnet.fc = nn.Identity()
        self.d_output = {
            '18': 512,
            '34': 512,
            '50': 2048,
            18: 512,
            34: 512,
            50: 2048,
            'wrn': 2048,
        }[variant]

    def forward(self, x, *args, **kwargs):
        # BSC -> BCS
        x = x.transpose(1, 2)
        # BCS -> BCHW
        n = int(x.size(-1)**.5)
        x = x.view(x.shape[0], self.d_input, n, n)
        if self.d_input == 1:
            x = x.repeat(1, 3, 1, 1)
        elif self.d_input == 3:
            pass
        else: raise NotImplementedError
        y = self.resnet.forward(x)
        y = y.unsqueeze(-2) # (B 1 C)
        return y, None
