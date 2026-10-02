"""
LunaMatch - Uniform Distribution & Spatial Sampling Module
Implements Stage 8 of the Lunar Image Registration Pipeline:
- Image grid partitioning (e.g. 8x8 or 10x10)
- Spatial binning & density balancing
- Intra-cell Non-Maximum Suppression (NMS)
- Prevents crater clustering and guarantees uniform lunar surface coverage
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def sample_uniformly(
    matches: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    grid_cols: int = 8,
    grid_rows: int = 8,
    max_pts_per_cell: int = 4,
    min_distance: float = 12.0
) -> Dict[str, Any]:
    """
    Distributes match points uniformly across the image by binning into a grid
    and retaining the highest-confidence points while enforcing minimum distance suppression.
    """
    if len(matches) == 0:
        return {
            "sampled_matches": [],
            "coverage_pct": 0.0,
            "uniformity_score": 0.0,
            "occupied_cells": 0,
            "total_cells": grid_cols * grid_rows,
            "grid_counts": [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        }

    cell_w = img_width / float(grid_cols)
    cell_h = img_height / float(grid_rows)
    
    # Initialize grid buckets
    grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
    
    # Assign each match to grid cell based on source image coordinate
    for m in matches:
        x, y = m["src_pt"]
        col = int(np.clip(x // cell_w, 0, grid_cols - 1))
        row = int(np.clip(y // cell_h, 0, grid_rows - 1))
        grid[row][col].append(m)

    selected_matches = []
    grid_counts = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
    occupied_cells = 0

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell_items = grid[r][c]
            if not cell_items:
                continue
                
            occupied_cells += 1
            # Sort by confidence descending
            cell_items.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            
            chosen_in_cell = []
            for item in cell_items:
                if len(chosen_in_cell) >= max_pts_per_cell:
                    break
                    
                px, py = item["src_pt"]
                
                # Check spatial distance with already chosen points in cell
                too_close = False
                for chosen in chosen_in_cell:
                    cx, cy = chosen["src_pt"]
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist < min_distance:
                        too_close = True
                        break
                        
                if not too_close:
                    item_with_grid = dict(item)
                    item_with_grid["grid_cell"] = [r, c]
                    chosen_in_cell.append(item_with_grid)
                    
            selected_matches.extend(chosen_in_cell)
            grid_counts[r][c] = len(chosen_in_cell)

    total_cells = grid_cols * grid_rows
    coverage_pct = round((occupied_cells / float(total_cells)) * 100.0, 2)
    
    # Compute Uniformity Score:
    # Normalized Shannon Entropy of cell distribution: H = -sum(p * log(p)) / log(total_cells)
    counts_flat = [count for row in grid_counts for count in row]
    total_selected = len(selected_matches)
    if total_selected > 0:
        probs = [c / float(total_selected) for c in counts_flat if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(total_cells)
        uniformity_score = round(float(entropy / max_entropy), 3) if max_entropy > 0 else 1.0
    else:
        uniformity_score = 0.0

    return {
        "sampled_matches": selected_matches,
        "coverage_pct": coverage_pct,
        "uniformity_score": uniformity_score,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "grid_counts": grid_counts,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "cell_w": cell_w,
        "cell_h": cell_h
    }
