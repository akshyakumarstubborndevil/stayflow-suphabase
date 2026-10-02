"""
LunaMatch - Multi-Scale Representation Module
Implements Stage 3 & 4 of the Lunar Image Registration Pipeline:
- Multi-scale Gaussian & Laplacian pyramids (1x, 0.5x, 0.25x)
- Scale-space feature generation
- Multi-resolution candidate matching
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def build_gaussian_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    """
    Builds a multi-scale Gaussian pyramid:
    Level 0: 1x (original resolution)
    Level 1: 0.5x (half resolution)
    Level 2: 0.25x (quarter resolution)
    """
    pyramid = [image]
    current = image
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(gaussian_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Builds a Laplacian pyramid from the Gaussian pyramid,
    capturing band-pass spatial frequencies of crater rims and ejecta blankets.
    """
    laplacian_pyr = []
    num_levels = len(gaussian_pyr)
    for i in range(num_levels - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


def get_multiscale_representations(image: np.ndarray, levels: int = 3) -> Dict[str, List[np.ndarray]]:
    """
    Generates multi-scale pyramid representations.
    """
    g_pyr = build_gaussian_pyramid(image, levels=levels)
    l_pyr = build_laplacian_pyramid(g_pyr)
    
    return {
        "gaussian": g_pyr,
        "laplacian": l_pyr,
        "scales": [1.0 / (2 ** i) for i in range(levels)]
    }
