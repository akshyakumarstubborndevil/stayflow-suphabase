"""
LunaMatch - Quantitative Evaluation & Metrics Module
Implements Stage 10 of the Lunar Image Registration Pipeline:
- RMSE (Root Mean Square Error): sqrt(1/N * sum((x_i - x^_i)^2 + (y_i - y^_i)^2))
- Inlier Count & Inlier Ratio
- Mean Matching Error (pixels)
- Spatial Coverage (%)
- Sub-pixel Localization Accuracy (%)
- Uniformity Score
- Formatted Quality Report Generator
"""

import numpy as np
from typing import List, Dict, Any


def compute_metrics(
    total_candidates: int,
    inliers: List[Dict[str, Any]],
    coverage_pct: float,
    uniformity_score: float,
    matrix: List[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous quantitative registration metrics.
    """
    inlier_count = len(inliers)
    inlier_ratio = round((inlier_count / float(total_candidates) * 100.0), 2) if total_candidates > 0 else 0.0
    
    if inlier_count == 0:
        return {
            "total_candidates": total_candidates,
            "inlier_count": 0,
            "inlier_ratio_pct": 0.0,
            "rmse": 0.0,
            "mean_matching_error": 0.0,
            "median_error": 0.0,
            "max_error": 0.0,
            "coverage_pct": 0.0,
            "uniformity_score": 0.0,
            "subpixel_accuracy_pct": 0.0,
            "quality_grade": "Failed (No Inliers)"
        }

    # Extract reprojection errors
    errors = np.array([m.get("reprojection_error", 0.0) for m in inliers], dtype=np.float64)
    
    # RMSE: sqrt(1/N * sum(e_i^2))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    max_err = float(np.max(errors))
    
    # Sub-pixel accuracy rate: percentage of points with error <= 0.5 pixels
    subpixel_count = np.sum(errors <= 0.5)
    subpixel_accuracy_pct = round((subpixel_count / float(inlier_count)) * 100.0, 2)
    
    # Assess registration quality tier
    if rmse < 0.5 and inlier_ratio >= 65.0:
        quality_grade = "Sub-Pixel Optimal (Research Grade)"
    elif rmse < 1.0 and inlier_ratio >= 50.0:
        quality_grade = "Excellent (Sub-Pixel Precision)"
    elif rmse < 2.0:
        quality_grade = "Good (Valid Geometric Match)"
    else:
        quality_grade = "Moderate (Coarse Alignment)"

    return {
        "total_candidates": total_candidates,
        "inlier_count": inlier_count,
        "inlier_ratio_pct": inlier_ratio,
        "rmse": round(rmse, 4),
        "mean_matching_error": round(mean_err, 4),
        "median_error": round(median_err, 4),
        "max_error": round(max_err, 4),
        "coverage_pct": round(coverage_pct, 2),
        "uniformity_score": round(uniformity_score, 3),
        "subpixel_accuracy_pct": subpixel_accuracy_pct,
        "quality_grade": quality_grade
    }


def format_quality_report(metrics: Dict[str, Any], sensor_type: str = "OHRC") -> str:
    """
    Generates a standardized text report formatted as specified in the problem statement.
    """
    report = f"""=======================================================
   LUNAMATCH - CHANDRAYAAN-2 REGISTRATION QUALITY REPORT
   Sensor Modality: {sensor_type}
=======================================================
Total candidate matches : {metrics['total_candidates']}
RANSAC inliers          : {metrics['inlier_count']}
Inlier ratio             : {metrics['inlier_ratio_pct']}%
RMSE                     : {metrics['rmse']} pixels
Mean matching error      : {metrics['mean_matching_error']} pixels
Coverage                 : {metrics['coverage_pct']}%
Sub-pixel accuracy (<0.5): {metrics['subpixel_accuracy_pct']}%
Spatial Uniformity Score : {metrics['uniformity_score']} / 1.000
Registration Quality     : {metrics['quality_grade']}
======================================================="""
    return report
