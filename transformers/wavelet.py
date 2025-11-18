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
                          1 / torch.sqrt(torch.tensor(2.0))])

        self.register_buffer('h', h.view(1, 1, -1))
        self.register_buffer('g', g.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == '1d':
            return self.dwt_1d(x)
        else:
            return self.dwt_2d(x)
