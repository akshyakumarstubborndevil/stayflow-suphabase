"""
LunaMatch - Registration Warping & Visual Output Module
Implements Stage 9 of the Lunar Image Registration Pipeline:
- Perspective image warping onto reference coordinate frame
- 50/50 Alpha blend overlay
- Checkerboard verification overlay (GIS standard)
- Spectral / Absolute difference residual map
- Side-by-side correspondence map rendering
- Uniform spatial grid visualizer
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def warp_source(image_src: np.ndarray, matrix: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """
    Warps source image into reference coordinate system using estimated homography matrix.
    ref_shape: (height, width)
    """
    h_ref, w_ref = ref_shape[:2]
    warped = cv2.warpPerspective(
        image_src,
        matrix,
        (w_ref, h_ref),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped


def create_alpha_blend(img_warped: np.ndarray, img_ref: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Creates an alpha blended image between warped source and reference.
    """
    w_rgb = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR) if len(img_warped.shape) == 2 else img_warped
    r_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if len(img_ref.shape) == 2 else img_ref
    
    # Mask out background where warped source is black (no data)
    mask = (img_warped > 5).astype(np.float32)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        
    blended = (w_rgb.astype(np.float32) * alpha * mask + 
               r_rgb.astype(np.float32) * (1.0 - alpha * mask))
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_checkerboard(img_warped: np.ndarray, img_ref: np.ndarray, square_size: int = 64) -> np.ndarray:
    """
    Creates a checkerboard alternating composite image.
    Essential for inspecting continuous crater rim transitions.
    """
    h, w = img_ref.shape[:2]
    w_rgb = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR) if len(img_warped.shape) == 2 else img_warped
    r_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if len(img_ref.shape) == 2 else img_ref
    
    checker = np.zeros_like(r_rgb)
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            y_end = min(y + square_size, h)
            x_end = min(x + square_size, w)
            if ((y // square_size) + (x // square_size)) % 2 == 0:
                checker[y:y_end, x:x_end] = w_rgb[y:y_end, x:x_end]
            else:
                checker[y:y_end, x:x_end] = r_rgb[y:y_end, x:x_end]
    return checker


def create_difference_heatmap(img_warped: np.ndarray, img_ref: np.ndarray) -> np.ndarray:
    """
    Generates an absolute difference heatmap to visualize registration residuals.
    """
    # Ensure grayscale
    g_warp = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY) if len(img_warped.shape) == 3 else img_warped
    g_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
    
    diff = cv2.absdiff(g_warp, g_ref)
    
    # Mask out regions outside valid source coverage
    valid_mask = (g_warp > 0).astype(np.uint8)
    diff = cv2.bitwise_and(diff, diff, mask=valid_mask)
    
    # Apply color map
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    # Set invalid background to dark gray
    diff_colored[valid_mask == 0] = [20, 20, 25]
    return diff_colored


def draw_correspondences(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    matches: List[Dict[str, Any]],
    max_draw: int = 120
) -> np.ndarray:
    """
    Renders side-by-side correspondence map with color-coded confidence links.
    Green = High confidence (> 0.8)
    Yellow = Moderate confidence (0.5 - 0.8)
    Red = Low confidence (< 0.5) or Outlier
    """
    h_src, w_src = img_src.shape[:2]
    h_ref, w_ref = img_ref.shape[:2]
    
    canvas_h = max(h_src, h_ref)
    canvas_w = w_src + w_ref + 20  # 20px divider
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    s_rgb = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if len(img_src.shape) == 2 else img_src
    r_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if len(img_ref.shape) == 2 else img_ref
    
    canvas[:h_src, :w_src] = s_rgb
    # Draw middle separator
    canvas[:, w_src:w_src + 20] = [25, 30, 45]
    canvas[:h_ref, w_src + 20:canvas_w] = r_rgb
    
    draw_matches = matches[:max_draw]
    
    for m in draw_matches:
        sx, sy = int(round(m["src_pt"][0])), int(round(m["src_pt"][1]))
        rx, ry = int(round(m["ref_pt"][0])) + w_src + 20, int(round(m["ref_pt"][1]))
        
        conf = m.get("confidence", 0.5)
        is_inlier = m.get("is_inlier", True)
        
        if not is_inlier:
            color = (0, 0, 220)  # Red for outliers
        elif conf >= 0.80:
            color = (50, 230, 100) # Bright Green
        elif conf >= 0.60:
            color = (0, 220, 255) # Cyan / Yellow-Green
        else:
            color = (0, 150, 255) # Orange
            
        # Draw keypoint circle on source
        cv2.circle(canvas, (sx, sy), 4, color, -1)
        cv2.circle(canvas, (sx, sy), 6, (0, 0, 0), 1)
        
        # Draw keypoint circle on reference
        cv2.circle(canvas, (rx, ry), 4, color, -1)
        cv2.circle(canvas, (rx, ry), 6, (0, 0, 0), 1)
        
        # Draw line connecting corresponding points
        cv2.line(canvas, (sx, sy), (rx, ry), color, 1, cv2.LINE_AA)
        
    return canvas


def draw_grid_density(
    img_src: np.ndarray,
    grid_counts: List[List[int]],
    grid_cols: int,
    grid_rows: int
) -> np.ndarray:
    """
    Renders the uniform spatial grid distribution on the source image.
    """
    h, w = img_src.shape[:2]
    canvas = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if len(img_src.shape) == 2 else img_src.copy()
    
    cell_w = w / float(grid_cols)
    cell_h = h / float(grid_rows)
    
    max_count = max([max(row) for row in grid_counts]) if grid_counts else 1
    max_count = max(1, max_count)
    
    overlay = canvas.copy()
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            
            count = grid_counts[r][c]
            # Color intensity based on count
            norm_density = count / float(max_count)
            color = (int(30 + 180 * norm_density), int(160 * (1 - norm_density)), int(230 * norm_density))
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 100, 140), 1)
            
            # Text count
            cv2.putText(canvas, str(count), (x1 + int(cell_w / 2) - 6, y1 + int(cell_h / 2) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        
    # Alpha blend density overlay
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    return canvas
