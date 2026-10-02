"""
LunaMatch - Geometric Verification Module
Implements Stage 6 of the Lunar Image Registration Pipeline:
- RANSAC geometric model estimation (Projective Homography & Affine)
- Outlier rejection (eliminating repetitive crater and false positive matches)
- Geometric residual calculation
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def verify_geometry(
    matches: List[Dict[str, Any]],
    model_type: str = "homography",
    ransac_thresh: float = 3.0,
    max_iters: int = 5000,
    confidence_prob: float = 0.995
) -> Dict[str, Any]:
    """
    Performs robust geometric verification with RANSAC to eliminate outliers.
    Models supported:
    - 'homography': 3x3 Projective transformation (handles viewpoint perspective differences)
    - 'affine': 2x3 Affine transformation (rotation, scale, shear, translation)
    """
    if len(matches) < 4:
        return {
            "success": False,
            "error_msg": f"Insufficient matches for geometric verification (found {len(matches)}, minimum required is 4)",
            "matrix": None,
            "inliers": [],
            "outliers": matches,
            "inlier_ratio": 0.0,
            "mean_error": 0.0
        }

    src_pts = np.float32([m["src_pt"] for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([m["ref_pt"] for m in matches]).reshape(-1, 1, 2)

    if model_type.lower() == "affine":
        matrix_affine, inlier_mask = cv2.estimateAffine2D(
            src_pts,
            ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence_prob
        )
        if matrix_affine is not None:
            # Convert 2x3 to 3x3 for uniform representation
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, :] = matrix_affine
        else:
            matrix = None
        mask = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.zeros(len(matches), dtype=bool)
    else:
        # Homography (default)
        matrix, inlier_mask = cv2.findHomography(
            src_pts,
            ref_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            maxIters=max_iters,
            confidence=confidence_prob
        )
        mask = (inlier_mask.ravel() == 1) if inlier_mask is not None else np.zeros(len(matches), dtype=bool)

    if matrix is None or np.sum(mask) < 4:
        return {
            "success": False,
            "error_msg": "RANSAC failed to estimate a consistent geometric transformation.",
            "matrix": None,
            "inliers": [],
            "outliers": matches,
            "inlier_ratio": 0.0,
            "mean_error": 0.0
        }

    # Compute reprojection residuals for all points
    # Transform src_pts using matrix
    src_pts_homogeneous = np.concatenate([src_pts.reshape(-1, 2), np.ones((len(matches), 1))], axis=1) # (N, 3)
    projected = (matrix @ src_pts_homogeneous.T).T # (N, 3)
    
    # Avoid divide-by-zero
    w = projected[:, 2:3]
    w[w == 0] = 1e-8
    projected_pts = projected[:, :2] / w # (N, 2)
    
    actual_ref = ref_pts.reshape(-1, 2)
    reprojection_errors = np.linalg.norm(actual_ref - projected_pts, axis=1)

    inliers = []
    outliers = []
    
    for i, m in enumerate(matches):
        match_copy = dict(m)
        match_copy["reprojection_error"] = float(reprojection_errors[i])
        match_copy["is_inlier"] = bool(mask[i])
        match_copy["projected_ref_pt"] = [float(projected_pts[i, 0]), float(projected_pts[i, 1])]
        
        if mask[i]:
            inliers.append(match_copy)
        else:
            outliers.append(match_copy)

    inlier_errors = reprojection_errors[mask]
    mean_inlier_error = float(np.mean(inlier_errors)) if len(inlier_errors) > 0 else 0.0
    inlier_ratio = float(len(inliers) / len(matches)) if len(matches) > 0 else 0.0

    return {
        "success": True,
        "matrix": matrix.tolist(),
        "model_type": model_type,
        "inliers": inliers,
        "outliers": outliers,
        "inlier_ratio": inlier_ratio,
        "mean_error": mean_inlier_error,
        "total_candidates": len(matches),
        "inlier_count": len(inliers)
    }
