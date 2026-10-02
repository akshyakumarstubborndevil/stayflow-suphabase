"""
LunaMatch - Illumination & Sun-Angle Normalizer Module
Implements Stage 2 of the Lunar Image Registration Pipeline:
- Sun-angle compensation (directional shading equalization)
- Gradient / edge-based representations (Sobel & Scharr directional derivatives)
- Structural terrain ridge and crater boundary extraction
- Structural Topography Map (illumination-invariant representation)
"""

import cv2
import numpy as np


def compute_structural_gradients(image: np.ndarray) -> np.ndarray:
    """
    Extracts gradient magnitude and structural edges.
    Terrain rims and crater boundaries remain invariant under illumination changes.
    """
    # Scharr filters provide better rotational symmetry than standard Sobel
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    # Normalize to 0-255
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return mag_norm.astype(np.uint8)


def extract_crater_boundaries(image: np.ndarray, low_thresh: int = 40, high_thresh: int = 120) -> np.ndarray:
    """
    Extracts crater boundaries and morphological ridges using Canny with hysteresis.
    """
    edges = cv2.Canny(image, low_thresh, high_thresh)
    # Dilate slightly to form connected structural rims
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges


def compensate_sun_angle(image: np.ndarray, sun_azimuth: float = None, sun_elevation: float = None) -> np.ndarray:
    """
    Compensates for solar illumination direction.
    If sun azimuth (0-360 deg) is provided, projects illumination along the incident solar vector
    and equalizes directional shadow asymmetry.
    """
    if sun_azimuth is None:
        # Default blind illumination compensation: high-pass homomorphic filtering
        img_f = image.astype(np.float32) + 1.0
        log_img = np.log(img_f)
        low_pass = cv2.GaussianBlur(log_img, (31, 31), 10.0)
        high_pass = log_img - low_pass
        comp = np.exp(high_pass)
        comp_norm = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX)
        return comp_norm.astype(np.uint8)

    # Convert azimuth to radians (0 = North, 90 = East, 180 = South, 270 = West)
    azimuth_rad = np.radians(sun_azimuth)
    dx = np.cos(azimuth_rad)
    dy = np.sin(azimuth_rad)
    
    # Compute directional derivative along sun illumination axis
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    sun_parallel = grad_x * dx + grad_y * dy
    sun_orthogonal = -grad_x * dy + grad_y * dx
    
    # Suppress illumination-dominated parallel gradients, boost terrain-defining orthogonal boundaries
    terrain_map = np.abs(sun_orthogonal) + 0.3 * np.abs(sun_parallel)
    terrain_norm = cv2.normalize(terrain_map, None, 0, 255, cv2.NORM_MINMAX)
    
    return terrain_norm.astype(np.uint8)


def generate_illumination_invariant_map(image: np.ndarray, sun_azimuth: float = None, sun_elevation: float = None, blend_weight: float = 0.65) -> dict:
    """
    Fuses gradient magnitude, crater rim structures, and sun-compensated illumination
    into a unified illumination-invariant structural representation.
    """
    # 1. Sun angle compensation
    compensated = compensate_sun_angle(image, sun_azimuth, sun_elevation)
    
    # 2. Structural gradient magnitude
    grad_mag = compute_structural_gradients(image)
    
    # 3. Crater boundaries
    crater_edges = extract_crater_boundaries(image)
    
    # 4. Fuse representations into a normalized structural feature map
    fused = (blend_weight * grad_mag.astype(np.float32) + 
             (1.0 - blend_weight) * compensated.astype(np.float32) + 
             0.2 * crater_edges.astype(np.float32))
             
    fused_norm = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return {
        "invariant_map": fused_norm,
        "gradient_magnitude": grad_mag,
        "crater_edges": crater_edges,
        "compensated": compensated
    }
