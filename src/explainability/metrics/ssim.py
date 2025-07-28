
import torch
import torch.nn.functional as F

def ssim(x : torch.Tensor, 
         y : torch.Tensor, 
         kernel_size: int = 11, 
         C1: float = 0.01**2, 
         C2: float = 0.03**2, 
         C3: float = (0.03**2)/2) -> torch.Tensor:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        x (torch.Tensor): Image tensor.
        y (torch.Tensor): Image tensor to compare against x.
        kernel_size (int): Size of the gaussian kernel.
        C1 (float): Constant to stabilize division by a small denominator.
        C2 (float): Constant to stabilize division by a small denominator.
        C3 (float): Constant to stabilize division by a small denominator.

    Returns:
        torch.Tensor: structural similarity value between x and y.
    """
    x = x.float()
    y = y.float()

    _, channels, height, width = x.size()
    kernel = create_gaussian_kernel().expand(channels, 1, kernel_size, kernel_size)
    kernel /= kernel.sum()

    # ensure kernel is on the same device as the input
    kernel = kernel.to(x.device)

    # calculate local means
    mu_x = F.conv2d(x, kernel, padding=kernel_size//2, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=kernel_size//2, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_x_mu_y = mu_x * mu_y

    # calculate local variance
    sigma_x_sq = F.conv2d(x*x, kernel, padding=kernel_size//2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y*y, kernel, padding=kernel_size//2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x*y, kernel, padding=kernel_size//2, groups=channels) - mu_x_mu_y
    
    # clamp variances to avoid division by zero
    sigma_x_sq = sigma_x_sq.clamp(min=1e-12)
    sigma_y_sq = sigma_y_sq.clamp(min=1e-12)

    # structural similarity components
    luminance = (2 * mu_x_mu_y + C1) / (mu_x_sq + mu_y_sq + C1)
    contrast = (2 * torch.sqrt(sigma_x_sq) * torch.sqrt(sigma_y_sq) + C2) / (sigma_x_sq + sigma_y_sq + C2)
    structure = (sigma_xy + C3) / (torch.sqrt(sigma_x_sq) * torch.sqrt(sigma_y_sq) + C3)

    ssim_val = luminance * contrast * structure
    return ssim_val.mean()



def create_gaussian_kernel(kernel_size: int=11, 
                           sigma: float=1.5) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel using PyTorch.

    Args:
        kernel_size (int): The size of the kernel (assumed to be square).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    range_val = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(-range_val, range_val + 1), torch.arange(-range_val, range_val + 1))
    
    gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel
