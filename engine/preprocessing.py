"""
LunaMatch - Preprocessing Module
Implements Stage 1 of the Lunar Image Registration Pipeline:
- Radiometric normalization
- Noise removal (Bilateral / Gaussian)
- Local contrast normalization (LCN / Wallis filter)
- Contrast enhancement (CLAHE)
"""

import cv2
import numpy as np


def normalize_radiometry(image: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """
    Normalizes radiometric values using percentile clipping to avoid sensor saturation spikes.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Float conversion
    img_f = gray.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (lower_percentile, upper_percentile))
    
    if p_high > p_low:
        clipped = np.clip((img_f - p_low) / (p_high - p_low) * 255.0, 0, 255)
    else:
        clipped = img_f
        
    return clipped.astype(np.uint8)


def remove_noise(image: np.ndarray, method: str = "bilateral", diameter: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Removes high-frequency sensor noise while preserving sharp crater boundaries and ridge topography.
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 1.0)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance micro-craters and ejecta patterns in deep shadows and bright rims.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def local_contrast_normalization(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Wallis / Local Contrast Normalization (LCN):
    Subtracts local mean and divides by local standard deviation.
    Critical for lunar surface imaging where regional albedo variations exist.
    """
    img_f = image.astype(np.float32)
    mean = cv2.blur(img_f, (kernel_size, kernel_size))
    mean_sq = cv2.blur(img_f * img_f, (kernel_size, kernel_size))
    variance = np.maximum(mean_sq - (mean * mean), 1e-4)
    std = np.sqrt(variance)
    
    # Normalize: (I - mu) / sigma
    normalized = (img_f - mean) / (std + 10.0)
    # Re-scale to 0-255 range
    norm_rescaled = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return norm_rescaled.astype(np.uint8)


def preprocess_image(image: np.ndarray, enable_clahe: bool = True, enable_lcn: bool = True, noise_method: str = "bilateral") -> dict:
    """
    Executes full preprocessing pipeline.
    Returns the preprocessed image along with intermediate diagnostic states.
    """
    # 1. Radiometric normalization
    rad_norm = normalize_radiometry(image)
    
    # 2. Noise removal
    denoised = remove_noise(rad_norm, method=noise_method)
    
    # 3. Local contrast normalization (optional)
    if enable_lcn:
        lcn_out = local_contrast_normalization(denoised)
    else:
        lcn_out = denoised
        
    # 4. Contrast enhancement (CLAHE)
    if enable_clahe:
        enhanced = apply_clahe(lcn_out)
    else:
        enhanced = lcn_out
        
    return {
        "processed": enhanced,
        "radiometric_norm": rad_norm,
        "denoised": denoised,
        "enhanced": enhanced
    }
