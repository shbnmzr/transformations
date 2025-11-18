import torch


def fourier_1d(signal: torch.Tensor) -> torch.Tensor:
    """
    computes the Fourier transform of a 1D signal
    :param signal: 1D signal, which is a Tensor of shape (...,N)
    :return: complex-valued Fourier coefficients
    """
    return torch.fft.fft(signal)


def fourier_2d(signal: torch.Tensor) -> torch.Tensor:
    """
    computes the Fourier transform of a 2D signal
    :param signal: Tensor of shape (B, C, H, W)
    :return: complex-valued Fourier coefficients
    """
    return torch.fft.fft2(signal)
