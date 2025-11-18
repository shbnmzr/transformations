import torch


def fourier_1D(signal: torch.Tensor) -> torch.Tensor:
    """
    computes the Fourier transform of a 1D signal
    :param signal: 1D signal, which is a Tensor of shape (...,N)
    :return: complex-valued Fourier coefficients
    """
    return torch.fft.fft(signal)
