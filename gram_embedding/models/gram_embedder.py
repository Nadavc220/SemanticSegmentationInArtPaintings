import os
import torch
from torch import nn
from functools import reduce

from gram_embedding.models.vgg_gatys import VGG, GramMatrix

STYLE_LAYERS = ['r11', 'r21', 'r31', 'r41', 'r51']


class GramEmbedder(nn.Module):
    def __init__(self):
        super(GramEmbedder, self).__init__()
        self.vgg = VGG()
        state_dict = torch.load(os.path.join('gram_embedding', 'weights', 'vgg_gatys.pth'))
        self.vgg.load_state_dict(state_dict)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.gram = GramMatrix()

    def forward(self, x):
        out = self.vgg(x, out_keys=STYLE_LAYERS)
        style_out = [self.gram(out[k]) for k in STYLE_LAYERS]
        out = [A.view(reduce(lambda a, b: a * b, A.shape)) for A in style_out]
        out = torch.cat(out)
        return out

