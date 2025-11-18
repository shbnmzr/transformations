import torch
import torch.nn as nn
import torch.nn.functional as F


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
                          -1 / torch.sqrt(torch.tensor(2.0))])

        self.register_buffer('h', h.view(1, 1, -1))
        self.register_buffer('g', g.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> tuple:
        if self.mode == '1d':
            return self.dwt_1d(x)
        else:
            return self.dwt_2d(x)

    def dwt_1d(self, x: torch.Tensor) -> tuple:
        # convolution with low-pass filter h
        low = F.conv1d(x, self.h, stride=2)

        # convolution with high-pass filter g
        high = F.conv1d(x, self.g, stride=2)
        return low, high

    def dwt_2d(self, x: torch.Tensor) -> tuple:
        # apply low-pass horizontally
        low_w = F.conv2d(x, self.h.unsqueeze(2), stride=(1, 2))
        # apply high-pass horizontally
        high_w = F.conv2d(x, self.g.unsqueeze(2), stride=(1, 2))

        # low-pass horizontally, low-pass vertically -> approximation
        ll = F.conv2d(low_w, self.h.unsqueeze(3), stride=(2, 1))
        # low-pass horizontally, high-pass vertically -> horizontal edges
        lh = F.conv2d(low_w, self.g.unsqueeze(3), stride=(2, 1))
        # high-pass horizontally, low-pass vertically -> vertical edges
        hl = F.conv2d(high_w, self.h.unsqueeze(3), stride=(2, 1))
        # high-pass horizontally, high-pass vertically -> diagonal details
        hh = F.conv2d(high_w, self.g.unsqueeze(3), stride=(2, 1))

        return ll, lh, hl, hh
