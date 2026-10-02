"""
LunaMatch - Master Image Registration Pipeline Orchestrator
Coordinates the 10 stages:
1. Image Preprocessing
2. Illumination / Sun-Angle Compensation
3. Multi-Scale Representation
4. Sensor-Invariant Feature Extraction (RootSIFT)
5. Feature Correspondence & Cross-Check
6. Geometric Verification (RANSAC)
7. Sub-Pixel Refinement
8. Uniform Spatial Distribution Sampling
9. Registration Output Generation (Warping, Blending, Checkerboard, Difference)
10. Quantitative Evaluation & Quality Report
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, Any

from engine.preprocessing import preprocess_image
from engine.illumination import generate_illumination_invariant_map
from engine.multi_scale import get_multiscale_representations
from engine.feature_matching import extract_features, match_features
from engine.geometric import verify_geometry
from engine.subpixel import refine_matches_subpixel
from engine.uniform_sampler import sample_uniformly
from engine.evaluator import compute_metrics, format_quality_report
from engine.registration import (
    warp_source,
    create_alpha_blend,
    create_checkerboard,
    create_difference_heatmap,
    draw_correspondences,
    draw_grid_density
)


class LunaMatchPipeline:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        img_src_path: str,
        img_ref_path: str,
        sensor_type: str = "OHRC",
        sun_azimuth_src: float = None,
        sun_azimuth_ref: float = None,
        geometric_model: str = "homography",
        ransac_thresh: float = 2.5,
        grid_cols: int = 8,
        grid_rows: int = 8,
        max_pts_per_cell: int = 4,
        enable_subpixel: bool = True,
        enable_clahe: bool = True,
        enable_lcn: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete LunaMatch registration pipeline.
        """
        t_start = time.time()
        stage_logs = []

        # Read images
        img_src_raw = cv2.imread(img_src_path)
        img_ref_raw = cv2.imread(img_ref_path)

        if img_src_raw is None or img_ref_raw is None:
            raise ValueError(f"Could not load images. Source: {img_src_path}, Ref: {img_ref_path}")

        # Grayscale conversions
        gray_src = cv2.cvtColor(img_src_raw, cv2.COLOR_BGR2GRAY) if len(img_src_raw.shape) == 3 else img_src_raw
        gray_ref = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2GRAY) if len(img_ref_raw.shape) == 3 else img_ref_raw

        h_src, w_src = gray_src.shape[:2]
        h_ref, w_ref = gray_ref.shape[:2]

        stage_logs.append({
            "stage": 1,
            "name": "Image Preprocessing",
            "detail": f"Source dimensions: {w_src}x{h_src} ({sensor_type}), Reference: {w_ref}x{h_ref}. Denoising & radiometric normalization."
        })

        # Stage 1: Preprocessing
        prep_src = preprocess_image(gray_src, enable_clahe=enable_clahe, enable_lcn=enable_lcn)
        prep_ref = preprocess_image(gray_ref, enable_clahe=enable_clahe, enable_lcn=enable_lcn)

        # Stage 2: Illumination / Sun-Angle Normalizer
        stage_logs.append({
            "stage": 2,
            "name": "Illumination & Sun-Angle Compensation",
            "detail": f"Constructing gradient magnitude and topographic ridge invariant representations (Src Az: {sun_azimuth_src}°, Ref Az: {sun_azimuth_ref}°)."
        })
        illum_src = generate_illumination_invariant_map(prep_src["processed"], sun_azimuth=sun_azimuth_src)
        illum_ref = generate_illumination_invariant_map(prep_ref["processed"], sun_azimuth=sun_azimuth_ref)

        # Stage 3: Multi-Scale Pyramid Representation
        stage_logs.append({
            "stage": 3,
            "name": "Multi-Scale Pyramid Representation",
            "detail": "Building 3-level Gaussian & Laplacian multi-resolution image pyramids (1x, 0.5x, 0.25x)."
        })
        pyr_src = get_multiscale_representations(prep_src["processed"], levels=3)
        pyr_ref = get_multiscale_representations(prep_ref["processed"], levels=3)

        # Stage 4: Sensor-Invariant Feature Extraction (RootSIFT)
        stage_logs.append({
            "stage": 4,
            "name": "Sensor-Invariant Feature Extraction",
            "detail": "Detecting scale-space extrema & mapping descriptors to RootSIFT Hellinger kernel space."
        })
        kp_src, desc_src = extract_features(prep_src["processed"], illum_src["invariant_map"], max_keypoints=4000)
        kp_ref, desc_ref = extract_features(prep_ref["processed"], illum_ref["invariant_map"], max_keypoints=4000)

        stage_logs.append({
            "stage": 5,
            "name": "Feature Correspondence & Mutual Verification",
            "detail": f"Extracted {len(kp_src)} source and {len(kp_ref)} reference keypoints. Applying Lowe's ratio test and bilateral cross-check."
        })

        # Stage 5: Feature Correspondence
        candidate_matches = match_features(kp_src, desc_src, kp_ref, desc_ref, ratio_thresh=0.78, mutual_check=True)

        if len(candidate_matches) < 4:
            # Fallback to looser ratio test if strictly cross-checked matches are low
            candidate_matches = match_features(kp_src, desc_src, kp_ref, desc_ref, ratio_thresh=0.85, mutual_check=False)

        # Stage 6: Geometric Verification (RANSAC)
        stage_logs.append({
            "stage": 6,
            "name": "Geometric Verification (RANSAC)",
            "detail": f"Estimating {geometric_model.upper()} transformation across {len(candidate_matches)} candidate correspondences with reprojection threshold {ransac_thresh} px."
        })
        geom_result = verify_geometry(
            candidate_matches,
            model_type=geometric_model,
            ransac_thresh=ransac_thresh,
            max_iters=5000
        )

        inliers = geom_result["inliers"]
        outliers = geom_result["outliers"]
        H_matrix = np.array(geom_result["matrix"]) if geom_result["matrix"] is not None else np.eye(3)

        # Stage 7: Sub-Pixel Refinement
        if enable_subpixel and len(inliers) >= 4:
            stage_logs.append({
                "stage": 7,
                "name": "Sub-Pixel Refinement",
                "detail": f"Refining {len(inliers)} inlier correspondences to fractional pixel precision using local patch NCC quadratic peak estimation and least-squares model fine-tuning."
            })
            inliers = refine_matches_subpixel(gray_src, gray_ref, inliers, patch_radius=6)
            
            # Re-estimate transformation using refined sub-pixel coordinates for optimal sub-pixel accuracy
            src_ref_pts = np.float32([m["src_pt"] for m in inliers]).reshape(-1, 1, 2)
            dst_ref_pts = np.float32([m["ref_pt"] for m in inliers]).reshape(-1, 1, 2)
            
            if geometric_model.lower() == "affine":
                A_fine, _ = cv2.estimateAffine2D(src_ref_pts, dst_ref_pts)
                if A_fine is not None:
                    H_matrix = np.eye(3, dtype=np.float64)
                    H_matrix[:2, :] = A_fine
            else:
                H_fine, _ = cv2.findHomography(src_ref_pts, dst_ref_pts, 0)
                if H_fine is not None:
                    H_matrix = H_fine
                    
            # Recalculate precise sub-pixel reprojection residuals
            src_h = np.concatenate([src_ref_pts.reshape(-1, 2), np.ones((len(inliers), 1))], axis=1)
            proj = (H_matrix @ src_h.T).T
            w_div = proj[:, 2:3]
            w_div[w_div == 0] = 1e-8
            proj_pts = proj[:, :2] / w_div
            refined_errs = np.linalg.norm(dst_ref_pts.reshape(-1, 2) - proj_pts, axis=1)
            
            for idx, m in enumerate(inliers):
                m["reprojection_error"] = float(round(refined_errs[idx], 3))
                m["projected_ref_pt"] = [float(proj_pts[idx, 0]), float(proj_pts[idx, 1])]
        else:
            stage_logs.append({
                "stage": 7,
                "name": "Sub-Pixel Refinement",
                "detail": "Sub-pixel refinement skipped or insufficient inliers."
            })

        # Stage 8: Uniform Distribution Sampling
        stage_logs.append({
            "stage": 8,
            "name": "Uniform Spatial Distribution",
            "detail": f"Binning into {grid_cols}x{grid_rows} grid to eliminate crater clustering and maximize lunar surface coverage."
        })
        uniform_result = sample_uniformly(
            inliers,
            img_width=w_src,
            img_height=h_src,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            max_pts_per_cell=max_pts_per_cell,
            min_distance=10.0
        )
        final_matches = uniform_result["sampled_matches"]

        # Stage 9: Registration & Visual Outputs
        stage_logs.append({
            "stage": 9,
            "name": "Registration & Visual Outputs",
            "detail": "Generating warped registered image, 50/50 blend, checkerboard overlay, and residual difference map."
        })

        ts = int(time.time() * 1000)
        # Warped source
        warped_src = warp_source(img_src_raw, H_matrix, (h_ref, w_ref))
        warped_path = os.path.join(self.output_dir, f"warped_{ts}.png")
        cv2.imwrite(warped_path, warped_src)

        # Alpha blend
        blend_img = create_alpha_blend(warped_src, img_ref_raw, alpha=0.5)
        blend_path = os.path.join(self.output_dir, f"blend_{ts}.png")
        cv2.imwrite(blend_path, blend_img)

        # Checkerboard
        checker_img = create_checkerboard(warped_src, img_ref_raw, square_size=48)
        checker_path = os.path.join(self.output_dir, f"checker_{ts}.png")
        cv2.imwrite(checker_path, checker_img)

        # Difference heatmap
        diff_img = create_difference_heatmap(warped_src, img_ref_raw)
        diff_path = os.path.join(self.output_dir, f"diff_{ts}.png")
        cv2.imwrite(diff_path, diff_img)

        # Correspondence map
        # Visualize top uniform matches (or all inliers if fewer)
        corr_viz = draw_correspondences(img_src_raw, img_ref_raw, final_matches if len(final_matches) > 0 else inliers)
        corr_path = os.path.join(self.output_dir, f"correspondence_{ts}.png")
        cv2.imwrite(corr_path, corr_viz)

        # Grid density map
        grid_viz = draw_grid_density(img_src_raw, uniform_result["grid_counts"], grid_cols, grid_rows)
        grid_path = os.path.join(self.output_dir, f"grid_{ts}.png")
        cv2.imwrite(grid_path, grid_viz)

        # Stage 10: Evaluation Metrics & Quality Report
        stage_logs.append({
            "stage": 10,
            "name": "Evaluation Metrics & Quality Report",
            "detail": "Computing RMSE, inlier ratio, spatial coverage, and sub-pixel accuracy scores."
        })

        eval_pts = final_matches if len(final_matches) > 0 else inliers
        metrics = compute_metrics(
            total_candidates=len(candidate_matches),
            inliers=eval_pts,
            coverage_pct=uniform_result["coverage_pct"],
            uniformity_score=uniform_result["uniformity_score"],
            matrix=H_matrix.tolist()
        )
        quality_report_text = format_quality_report(metrics, sensor_type=sensor_type)

        elapsed = round(time.time() - t_start, 3)

        return {
            "success": geom_result["success"],
            "elapsed_seconds": elapsed,
            "sensor_type": sensor_type,
            "metrics": metrics,
            "quality_report_text": quality_report_text,
            "stage_logs": stage_logs,
            "matrix": H_matrix.tolist(),
            "matches_summary": {
                "total_candidates": len(candidate_matches),
                "ransac_inliers": len(inliers),
                "uniform_selected": len(final_matches),
                "coverage_pct": uniform_result["coverage_pct"],
                "uniformity_score": uniform_result["uniformity_score"]
            },
            "output_files": {
                "warped": f"/outputs/{os.path.basename(warped_path)}",
                "blend": f"/outputs/{os.path.basename(blend_path)}",
                "checker": f"/outputs/{os.path.basename(checker_path)}",
                "diff": f"/outputs/{os.path.basename(diff_path)}",
                "correspondence": f"/outputs/{os.path.basename(corr_path)}",
                "grid": f"/outputs/{os.path.basename(grid_path)}"
            },
            "match_points": [
                {
                    "id": i + 1,
                    "source_x": round(m["src_pt"][0], 2),
                    "source_y": round(m["src_pt"][1], 2),
                    "reference_x": round(m["ref_pt"][0], 2),
                    "reference_y": round(m["ref_pt"][1], 2),
                    "confidence": round(m.get("confidence", 0.0), 3),
                    "reprojection_error": round(m.get("reprojection_error", 0.0), 3),
                    "is_subpixel": m.get("is_subpixel_refined", False)
                }
                for i, m in enumerate(final_matches if len(final_matches) > 0 else inliers)
            ]
        }
