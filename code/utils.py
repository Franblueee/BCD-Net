import torch
import numpy as np

def rgb2od(rgb):
    """Converts RGB image to optical density (OD_RGB) image.
    Args:
        rgb: RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    Returns:
        od: OD_RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    """
    if isinstance(rgb, np.ndarray):
        od = -np.log((rgb + 1.0) / 256.0) / np.log(256.0)
    elif isinstance(rgb, torch.Tensor):
        od = -torch.log((rgb + 1.0) / 256.0) / np.log(256.0)
    return od

def od2rgb(od):
    """Converts optical density (OD) image to RGB image.
    Args:
        od: OD image tensor / numpy array. Shape: (batch_size, 3, H, W)
    Returns:
        rgb: RGB image tensor / numpy array. Shape: (batch_size, 3, H, W)
    """
    if isinstance(od, np.ndarray):
        rgb = 256.0 * np.exp(-od * np.log(256.0)) - 1.0
    elif isinstance(od, torch.Tensor):
        rgb = 256.0 * torch.exp(-od * np.log(256.0)) - 1.0
    return rgb

def direct_deconvolution(Y, M):
    """Solves Y = MC using least squares.

    Args:
        Y: OD image tensor / numpy array. Shape: (batch_size, 3, H, W)
        M: color matrix tensor / numpy array. Shape: (batch_size, 3, 2)
    """

    squeeze_at_end = False
    if len(Y.shape) == 3:
        Y = Y.unsqueeze(0)
        M = M.unsqueeze(0)
        squeeze_at_end = True

    batch_size, _, H, W = Y.shape

    if isinstance(Y, np.ndarray):
        Y = Y.reshape(batch_size, 3, -1) # (batch_size, 3, H*W)
        M = M.reshape(batch_size, 3, 2) # (batch_size, 3, 2)
        C = np.linalg.lstsq(M, Y, rcond=None)[0] # (batch_size, 2, H*W)
        C = C.reshape(batch_size, 2, H, W) # (batch_size, 2, H, W)
    elif isinstance(Y, torch.Tensor):
        Y = Y.view(batch_size, 3, -1) # (batch_size, 3, H*W)
        M = M.view(batch_size, 3, 2) # (batch_size, 3, 2)
        C = torch.linalg.lstsq(M, Y).solution # (batch_size, 2, H*W)
        C = C.view(batch_size, 2, H, W) # (batch_size, 2, H, W)

    if squeeze_at_end:
        C = C.squeeze(0)    
    return C


def peak_signal_noise_ratio(A, B, max=255.0):
    """
    input: 
        A: tensor of shape (N, C, H, W)
        B: tensor of shape (N, C, H, W)
    return:
        psnr: tensor of shape (N, )
    """
    if max is None:
        max = torch.max(A)
    mse = torch.mean((A - B)**2, dim=(1,2,3))
    psnr = 10 * torch.log10(max**2 / mse)
    return psnr

def structural_similarity(A, B):
    """
    input: 
        A: tensor of shape (N, C, H, W)
        B: tensor of shape (N, C, H, W)
    return:
        ssim: tensor of shape (N, )
    """
    A = A.view(A.shape[0], -1) # (N, C*H*W)
    B = B.view(B.shape[0], -1) # (N, C*H*W)

    data_range = 255.0

    mu_A = torch.mean(A, dim=(1)) # (N, )
    mu_B = torch.mean(B, dim=(1)) # (N, )
    var_A = torch.var(A, dim=(1)) # (N, )
    var_B = torch.var(B, dim=(1)) # (N, )
    cov_AB = torch.mean((A - mu_A.view(-1, 1)) * (B - mu_B.view(-1, 1)), dim=(1)) # (N, )
    
    c1 = (0.01*data_range)**2
    c2 = (0.03*data_range)**2
    ssim = (2 * mu_A * mu_B + c1) * (2 * cov_AB + c2) / ((mu_A**2 + mu_B**2 + c1) * (var_A + var_B + c2))
    return ssim