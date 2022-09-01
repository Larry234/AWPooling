import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

class AWPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, temperature=1, same=False):
        super(AWPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding) # convert to l, r, t, b
        self.t = temperature
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)

            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)

        else:
            padding = self.padding

        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        B, C, o_h, o_w = x.size()[:4]
        w = F.softmax(x.contiguous().view(B, C, o_h, o_w, -1) / self.t, dim=-1)
        # generate pool weights
        w = w.view(B, C, o_h, o_w, self.k[0], self.k[1])
        
        return torch.sum((w * x).view(B, C, o_h, o_w, -1), dim=-1)