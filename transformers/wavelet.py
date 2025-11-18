import torch
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql.connect.functions import hll_sketch_agg


class HaarWaveletTransform(nn.Module):
    """
    minimal differentiable wavelet transform
    """
    def __init__(self, mode):
        super().__init__()
        assert mode in ['1d', '2d']
        self.mode = mode

        # Haar low-pass and high-pass filters
        h = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)),
                          1 / torch.sqrt(torch.tensor(2.0))])

        g = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)),
                          1 / torch.sqrt(torch.tensor(2.0))])

        self.register_buffer('h', h.view(1, 1, -1))
        self.register_buffer('g', g.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == '1d':
            return self.dwt_1d(x)
        else:
            return self.dwt_2d(x)

    def dwt_1d(self, x: torch.Tensor) -> torch.Tensor:
        low = F.conv1d(x, self.h, stride=2)
        high = F.conv1d(x, self.g, stride=2)
        return low, high

    def dwt_2d(self, x: torch.Tensor) -> torch.Tensor:
        low_w = F.conv2d(x, self.h.unsqueeze(2), stride=(1, 2))
        high_w = F.conv2d(x, self.g.unsqueeze(2), stride=(1, 2))

        ll = F.conv2d(low_w, self.h.unsqueeze(3), stride=(2, 1))
        lh = F.conv2d(low_w, self.g.unsqueeze(3), stride=(2, 1))
        hl = F.conv2d(high_w, self.h.unsqueeze(3), stride=(2, 1))
        hh = F.conv2d(high_w, self.g.unsqueeze(3), stride=(2, 1))

        return ll, lh, hl, hh
