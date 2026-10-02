"""
LunaMatch - Sub-Pixel Refinement Module
Implements Stage 7 of the Lunar Image Registration Pipeline:
- Sub-pixel corner/feature localization (gradient-based)
- Local Patch Normalized Cross-Correlation (NCC) with 2D quadratic peak interpolation
- Fractional coordinate displacement estimation: (X.xx, Y.yy)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def refine_subpixel_gradient(image: np.ndarray, points: np.ndarray, win_size: int = 5) -> np.ndarray:
    """
    Refines points with gradient-based sub-pixel precision using cornerSubPix.
    """
    if len(points) == 0:
        return points
    
    pts_f = np.float32(points).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(image, pts_f, (win_size, win_size), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def quadratic_peak_2d(patch: np.ndarray) -> Tuple[float, float]:
    """
    Fits a 2D quadratic surface to a 3x3 neighborhood around the peak
    and computes the fractional sub-pixel offset (dx, dy).
    patch: 3x3 matrix centered at peak
    """
    # Coordinates: [-1, 0, 1]
    # Taylor expansion: f(x, y) = f0 + fx*x + fy*y + 0.5*fxx*x^2 + fxy*x*y + 0.5*fyy*y^2
    # Extrema is at inv(Hessian) * -grad
    if patch.shape != (3, 3):
        return 0.0, 0.0
    
    fx = (patch[1, 2] - patch[1, 0]) / 2.0
    fy = (patch[2, 1] - patch[0, 1]) / 2.0
    
    fxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
    fyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
    fxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0
    
    det = fxx * fyy - fxy * fxy
    if abs(det) < 1e-6:
        return 0.0, 0.0
        
    dx = -(fyy * fx - fxy * fy) / det
    dy = -(fxx * fy - fxy * fx) / det
    
    # Clip displacement to sub-pixel range [-1.0, 1.0]
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    return dx, dy


def refine_matches_subpixel(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    matches: List[Dict[str, Any]],
    patch_radius: int = 7
) -> List[Dict[str, Any]]:
    """
    Executes sub-pixel refinement on all validated inlier correspondences.
    Returns matches with fractional coordinate precision (e.g. 124.37, 85.72).
    """
    if len(matches) == 0:
        return []

    h_src, w_src = img_src.shape[:2]
    h_ref, w_ref = img_ref.shape[:2]
    
    refined_matches = []
    
    for m in matches:
        match_refined = dict(m)
        sx, sy = m["src_pt"]
        rx, ry = m["ref_pt"]
        
        # Check boundary margins for patch extraction
        if (sx < patch_radius or sx >= w_src - patch_radius or
            sy < patch_radius or sy >= h_src - patch_radius or
            rx < patch_radius + 2 or rx >= w_ref - patch_radius - 2 or
            ry < patch_radius + 2 or ry >= h_ref - patch_radius - 2):
            # If too close to edge, preserve initial coordinates rounded to 2 decimal places
            match_refined["src_pt"] = [round(sx, 2), round(sy, 2)]
            match_refined["ref_pt"] = [round(rx, 2), round(ry, 2)]
            match_refined["subpixel_offset"] = [0.0, 0.0]
            match_refined["is_subpixel_refined"] = False
            refined_matches.append(match_refined)
            continue

        # Extract source patch
        isx, isy = int(round(sx)), int(round(sy))
        irx, iry = int(round(rx)), int(round(ry))
        
        patch_src = img_src[isy - patch_radius:isy + patch_radius + 1,
                            isx - patch_radius:isx + patch_radius + 1]
                            
        # Extract slightly larger reference search window
        search_radius = patch_radius + 2
        window_ref = img_ref[iry - search_radius:iry + search_radius + 1,
                             irx - search_radius:irx + search_radius + 1]
                             
        if patch_src.shape[0] != 2 * patch_radius + 1 or patch_src.shape[1] != 2 * patch_radius + 1:
            match_refined["src_pt"] = [round(sx, 2), round(sy, 2)]
            match_refined["ref_pt"] = [round(rx, 2), round(ry, 2)]
            match_refined["subpixel_offset"] = [0.0, 0.0]
            match_refined["is_subpixel_refined"] = False
            refined_matches.append(match_refined)
            continue

        # Compute Normalized Cross-Correlation (NCC)
        res = cv2.matchTemplate(window_ref, patch_src, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Peak coordinates in search window
        peak_x, peak_y = max_loc
        
        # Sub-pixel quadratic interpolation around peak if not on boundary of res
        if 0 < peak_x < res.shape[1] - 1 and 0 < peak_y < res.shape[0] - 1:
            sub_res = res[peak_y - 1:peak_y + 2, peak_x - 1:peak_x + 2]
            dx, dy = quadratic_peak_2d(sub_res)
        else:
            dx, dy = 0.0, 0.0
            
        # Shift relative to window center
        center_x = (window_ref.shape[1] - patch_src.shape[1]) / 2.0
        center_y = (window_ref.shape[0] - patch_src.shape[0]) / 2.0
        
        delta_x = (peak_x + dx) - center_x
        delta_y = (peak_y + dy) - center_y
        
        # Validate that displacement is within reasonable sub-pixel bounds (< 1.5 pixels)
        if np.sqrt(delta_x**2 + delta_y**2) < 2.0:
            refined_rx = float(round(rx + delta_x, 2))
            refined_ry = float(round(ry + delta_y, 2))
            match_refined["ref_pt"] = [refined_rx, refined_ry]
            match_refined["subpixel_offset"] = [round(float(delta_x), 3), round(float(delta_y), 3)]
            match_refined["is_subpixel_refined"] = True
        else:
            match_refined["src_pt"] = [round(sx, 2), round(sy, 2)]
            match_refined["ref_pt"] = [round(rx, 2), round(ry, 2)]
            match_refined["subpixel_offset"] = [0.0, 0.0]
            match_refined["is_subpixel_refined"] = False

        match_refined["src_pt"] = [round(sx, 2), round(sy, 2)]
        refined_matches.append(match_refined)
        
    return refined_matches
