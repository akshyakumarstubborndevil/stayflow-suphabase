"""
LunaMatch - Sensor-Invariant Feature Extraction & Correspondence Module
Implements Stage 4 & 5 of the Lunar Image Registration Pipeline:
- Sensor-invariant feature extraction (RootSIFT + structural descriptors)
- Multi-scale keypoint detection
- Dual-image matching with Lowe's ratio test
- Mutual nearest neighbor (cross-check) verification
- Correspondence confidence score computation
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def rootsift(descriptors: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Transforms SIFT descriptors into RootSIFT (Hellinger kernel).
    L1-normalize, take square root, then L2-normalize.
    Significantly improves matching performance across multimodal sensor domains (OHRC vs TMC vs IIRS).
    """
    if descriptors is None or len(descriptors) == 0:
        return descriptors
    
    # L1 normalize
    l1_norm = descriptors / (np.sum(np.abs(descriptors), axis=1, keepdims=True) + eps)
    # Square root
    root_desc = np.sqrt(l1_norm)
    # L2 normalize
    l2_norm = root_desc / (np.linalg.norm(root_desc, axis=1, keepdims=True) + eps)
    return l2_norm.astype(np.float32)


def extract_features(image: np.ndarray, structural_map: np.ndarray = None, max_keypoints: int = 5000) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extracts sensor-invariant scale-space keypoints and descriptors.
    Uses SIFT on both radiometric-enhanced image and structural gradient maps.
    """
    sift = cv2.SIFT_create(
        nfeatures=max_keypoints,
        nOctaveLayers=4,
        contrastThreshold=0.02,  # Lower threshold for low-contrast lunar craters
        edgeThreshold=12.0,
        sigma=1.4
    )
    
    # Primary keypoint extraction on preprocessed image
    kp1, desc1 = sift.detectAndCompute(image, None)
    
    if structural_map is not None:
        kp2, desc2 = sift.detectAndCompute(structural_map, None)
        if desc1 is not None and desc2 is not None and len(desc2) > 0:
            # Combine keypoints and descriptors, filtering duplicates
            all_kp = list(kp1) + list(kp2)
            all_desc = np.vstack([desc1, desc2])
        elif desc1 is not None:
            all_kp = list(kp1)
            all_desc = desc1
        elif desc2 is not None:
            all_kp = list(kp2)
            all_desc = desc2
        else:
            all_kp, all_desc = [], None
    else:
        all_kp, all_desc = list(kp1) if kp1 else [], desc1

    if all_desc is not None and len(all_desc) > 0:
        # Convert to RootSIFT
        all_desc = rootsift(all_desc)

    return all_kp, all_desc


def match_features(
    kp_src: List[cv2.KeyPoint],
    desc_src: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    desc_ref: np.ndarray,
    ratio_thresh: float = 0.78,
    mutual_check: bool = True
) -> List[Dict[str, Any]]:
    """
    Matches features between source and reference images.
    Applies Lowe's ratio test and optional mutual cross-check.
    Computes a normalized confidence score for each correspondence.
    """
    if desc_src is None or desc_ref is None or len(desc_src) < 4 or len(desc_ref) < 4:
        return []

    # Flann-based or BFMatcher with L2 norm
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # 1. Forward 2-NN matching (Source -> Reference)
    knn_matches = matcher.knnMatch(desc_src, desc_ref, k=2)
    
    forward_good = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                # Lowe's ratio passes
                forward_good.append((m, n))

    # 2. Reverse 2-NN matching (Reference -> Source) for mutual cross-check
    if mutual_check:
        knn_reverse = matcher.knnMatch(desc_ref, desc_src, k=2)
        reverse_best = {}
        for r_pair in knn_reverse:
            if len(r_pair) >= 1:
                # r_pair[0].queryIdx is ref idx, r_pair[0].trainIdx is src idx
                reverse_best[r_pair[0].queryIdx] = r_pair[0].trainIdx
    else:
        reverse_best = None

    candidate_matches = []
    
    for m, n in forward_good:
        src_idx = m.queryIdx
        ref_idx = m.trainIdx
        
        # Check mutual consistency
        if mutual_check and reverse_best is not None:
            if reverse_best.get(ref_idx) != src_idx:
                continue  # Not mutually consistent, drop

        # Compute confidence score:
        # Based on Lowe's ratio margin and absolute distance
        ratio_margin = max(0.0, 1.0 - (m.distance / (n.distance + 1e-6)))
        # Distance weight: lower distance gives higher score
        dist_factor = np.exp(-m.distance / 1.5)
        confidence = float(np.clip(0.6 * ratio_margin + 0.4 * dist_factor, 0.05, 0.99))
        
        pt_src = kp_src[src_idx].pt
        pt_ref = kp_ref[ref_idx].pt
        
        candidate_matches.append({
            "src_idx": int(src_idx),
            "ref_idx": int(ref_idx),
            "src_pt": [float(pt_src[0]), float(pt_src[1])],
            "ref_pt": [float(pt_ref[0]), float(pt_ref[1])],
            "confidence": confidence,
            "distance": float(m.distance),
            "ratio": float(m.distance / (n.distance + 1e-6)),
            "src_size": float(kp_src[src_idx].size),
            "ref_size": float(kp_ref[ref_idx].size)
        })

    # Sort descending by confidence
    candidate_matches.sort(key=lambda x: x["confidence"], reverse=True)
    return candidate_matches
