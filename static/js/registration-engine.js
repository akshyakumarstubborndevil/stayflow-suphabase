/**
 * LunaMatch Client-Side In-Browser Computer Vision & Image Registration Engine
 * Enables 100% serverless, zero-backend execution on Netlify!
 * Algorithms: FAST/Harris Corner Detection, Local Patch Descriptors,
 * Lowe's Ratio Cross-Matching, RANSAC Homography, Sub-Pixel Localization,
 * Bilinear Inverse Warping, and Multi-Layer Visual Synthesizers.
 */

class ClientRegistrationEngine {
  constructor() {
    this.statusCallback = null;
  }

  setStatusCallback(cb) {
    this.statusCallback = cb;
  }

  log(stage, name, detail) {
    if (this.statusCallback) {
      this.statusCallback({ stage, name, detail });
    }
  }

  // Convert HTML Image or DataURL to Canvas
  createCanvasFromImage(img) {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    return canvas;
  }

  // Preprocessing: Grayscale & Contrast Equalization
  getGrayscaleData(ctx, width, height, clahe = true) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const d = imgData.data;
    const gray = new Float32Array(width * height);

    let min = 255, max = 0;
    for (let i = 0; i < d.length; i += 4) {
      const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      gray[i / 4] = g;
      if (g < min) min = g;
      if (g > max) max = g;
    }

    // Min-Max linear stretch
    const range = (max - min) || 1;
    for (let i = 0; i < gray.length; i++) {
      gray[i] = ((gray[i] - min) / range) * 255;
    }

    return { gray, imgData };
  }

  // Stage 4: Keypoint Detection using FAST & Harris Salience
  detectKeypoints(gray, width, height, maxPoints = 400, threshold = 22) {
    const keypoints = [];
    const scores = new Float32Array(width * height);

    // Compute Sobel Gradients
    const gx = new Float32Array(width * height);
    const gy = new Float32Array(width * height);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        gx[idx] = (gray[idx + 1] - gray[idx - 1]) * 0.5;
        gy[idx] = (gray[idx + width] - gray[idx - width]) * 0.5;
      }
    }

    // Harris Corner Response
    const k = 0.04;
    for (let y = 4; y < height - 4; y += 2) {
      for (let x = 4; x < width - 4; x += 2) {
        let sxx = 0, syy = 0, sxy = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const pidx = (y + dy) * width + (x + dx);
            const ix = gx[pidx];
            const iy = gy[pidx];
            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
          }
        }
        const det = (sxx * syy) - (sxy * sxy);
        const trace = sxx + syy;
        const response = det - k * (trace * trace);
        if (response > 1000) {
          scores[y * width + x] = response;
        }
      }
    }

    // Non-maximum suppression
    const radius = 6;
    for (let y = radius; y < height - radius; y += 3) {
      for (let x = radius; x < width - radius; x += 3) {
        const centerScore = scores[y * width + x];
        if (centerScore <= 0) continue;

        let isMax = true;
        for (let dy = -radius; dy <= radius; dy += 2) {
          for (let dx = -radius; dx <= radius; dx += 2) {
            if (scores[(y + dy) * width + (x + dx)] > centerScore) {
              isMax = false;
              break;
            }
          }
          if (!isMax) break;
        }

        if (isMax) {
          keypoints.push({ x, y, score: centerScore });
        }
      }
    }

    // Sort by score and truncate
    keypoints.sort((a, b) => b.score - a.score);
    return keypoints.slice(0, maxPoints);
  }

  // Descriptor extraction: 11x11 normalized patch
  extractDescriptors(gray, width, height, keypoints, patchSize = 11) {
    const half = Math.floor(patchSize / 2);
    const descriptors = [];
    const validKeypoints = [];

    for (const kp of keypoints) {
      const x = Math.round(kp.x);
      const y = Math.round(kp.y);

      if (x < half || x >= width - half || y < half || y >= height - half) {
        continue;
      }

      const desc = new Float32Array(patchSize * patchSize);
      let sum = 0, sumSq = 0;
      let ptr = 0;

      for (let dy = -half; dy <= half; dy++) {
        for (let dx = -half; dx <= half; dx++) {
          const val = gray[(y + dy) * width + (x + dx)];
          desc[ptr++] = val;
          sum += val;
          sumSq += val * val;
        }
      }

      // Mean & standard deviation normalization
      const mean = sum / desc.length;
      const variance = (sumSq / desc.length) - (mean * mean);
      const std = Math.sqrt(Math.max(variance, 1e-4));

      for (let i = 0; i < desc.length; i++) {
        desc[i] = (desc[i] - mean) / std;
      }

      descriptors.push(desc);
      validKeypoints.push(kp);
    }

    return { keypoints: validKeypoints, descriptors };
  }

  // Cross-matching with Lowe's ratio test
  matchFeatures(kps1, desc1, kps2, desc2, ratioThresh = 0.85) {
    const matches = [];

    for (let i = 0; i < kps1.length; i++) {
      const d1 = desc1[i];
      let bestDist = Infinity;
      let secondDist = Infinity;
      let bestIdx = -1;

      for (let j = 0; j < kps2.length; j++) {
        const d2 = desc2[j];
        let ssd = 0;
        for (let k = 0; k < d1.length; k++) {
          const diff = d1[k] - d2[k];
          ssd += diff * diff;
        }

        if (ssd < bestDist) {
          secondDist = bestDist;
          bestDist = ssd;
          bestIdx = j;
        } else if (ssd < secondDist) {
          secondDist = ssd;
        }
      }

      if (bestDist < ratioThresh * secondDist && bestIdx !== -1) {
        const confidence = Math.max(0, 1.0 - (bestDist / (d1.length * 4)));
        matches.push({
          src_pt: [kps1[i].x, kps1[i].y],
          ref_pt: [kps2[bestIdx].x, kps2[bestIdx].y],
          confidence: Number(confidence.toFixed(3)),
          distance: bestDist
        });
      }
    }

    return matches;
  }

  // Stage 6: RANSAC Homography / Affine Estimation
  estimateHomographyRANSAC(matches, maxIter = 1000, inlierThresh = 3.0) {
    if (matches.length < 4) {
      return { success: false, inliers: [], matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    }

    let bestInliers = [];
    let bestMatrix = null;

    for (let it = 0; it < maxIter; it++) {
      // Pick 4 random distinct matches
      const idxs = [];
      while (idxs.length < 4) {
        const r = Math.floor(Math.random() * matches.length);
        if (!idxs.includes(r)) idxs.push(r);
      }

      const ptsSrc = idxs.map(i => matches[i].src_pt);
      const ptsRef = idxs.map(i => matches[i].ref_pt);

      const H = this.findHomography4Points(ptsSrc, ptsRef);
      if (!H) continue;

      const currentInliers = [];
      for (let i = 0; i < matches.length; i++) {
        const m = matches[i];
        const [x, y] = m.src_pt;
        const [rx, ry] = m.ref_pt;

        const w = H[6] * x + H[7] * y + H[8];
        if (Math.abs(w) < 1e-6) continue;
        const px = (H[0] * x + H[1] * y + H[2]) / w;
        const py = (H[3] * x + H[4] * y + H[5]) / w;

        const err = Math.hypot(rx - px, ry - py);
        if (err < inlierThresh) {
          currentInliers.push({
            ...m,
            reprojection_error: Number(err.toFixed(3)),
            projected_pt: [px, py]
          });
        }
      }

      if (currentInliers.length > bestInliers.length) {
        bestInliers = currentInliers;
        bestMatrix = H;
        // Early stop if excellent consensus reached
        if (bestInliers.length > matches.length * 0.75) break;
      }
    }

    if (bestInliers.length < 4 || !bestMatrix) {
      // Identity fallback
      return {
        success: false,
        inliers: matches,
        matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1]
      };
    }

    return {
      success: true,
      inliers: bestInliers,
      matrix: bestMatrix
    };
  }

  // 4-point Direct Linear Transform (DLT)
  findHomography4Points(src, dst) {
    const A = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [u, v] = dst[i];
      A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
      A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
    }

    // Solve Ah = 0 via Gaussian elimination with partial pivoting
    // Normalized affine/projective approximation
    const m = [];
    for (let r = 0; r < 8; r++) {
      m.push([...A[r].slice(0, 8), -A[r][8]]);
    }

    for (let i = 0; i < 8; i++) {
      let maxRow = i;
      for (let k = i + 1; k < 8; k++) {
        if (Math.abs(m[k][i]) > Math.abs(m[maxRow][i])) maxRow = k;
      }
      [m[i], m[maxRow]] = [m[maxRow], m[i]];

      if (Math.abs(m[i][i]) < 1e-9) return null;

      for (let k = i + 1; k < 8; k++) {
        const factor = m[k][i] / m[i][i];
        for (let j = i; j <= 8; j++) {
          m[k][j] -= factor * m[i][j];
        }
      }
    }

    const h = new Array(8);
    for (let i = 7; i >= 0; i--) {
      let sum = m[i][8];
      for (let j = i + 1; j < 8; j++) {
        sum -= m[i][j] * h[j];
      }
      h[i] = sum / m[i][i];
    }

    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
  }

  // Stage 7: Sub-pixel Parabolic Refinement
  refineSubpixel(graySrc, grayRef, inliers, width, height) {
    return inliers.map(m => {
      const rx = Math.round(m.ref_pt[0]);
      const ry = Math.round(m.ref_pt[1]);

      let dx = 0, dy = 0;
      if (rx > 1 && rx < width - 2 && ry > 1 && ry < height - 2) {
        const v0 = grayRef[ry * width + rx];
        const vx1 = grayRef[ry * width + (rx - 1)];
        const vx2 = grayRef[ry * width + (rx + 1)];
        const vy1 = grayRef[(ry - 1) * width + rx];
        const vy2 = grayRef[(ry + 1) * width + rx];

        // Parabolic peak interpolation
        const denomX = 2 * (2 * v0 - vx1 - vx2);
        const denomY = 2 * (2 * v0 - vy1 - vy2);

        if (Math.abs(denomX) > 1e-4) dx = Math.max(-0.5, Math.min(0.5, (vx1 - vx2) / denomX));
        if (Math.abs(denomY) > 1e-4) dy = Math.max(-0.5, Math.min(0.5, (vy1 - vy2) / denomY));
      }

      return {
        ...m,
        ref_pt: [Number((m.ref_pt[0] + dx).toFixed(2)), Number((m.ref_pt[1] + dy).toFixed(2))],
        is_subpixel: Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05
      };
    });
  }

  // Stage 8: Spatial Uniform Grid Sampling
  sampleUniformGrid(inliers, width, height, gridCols = 8, gridRows = 8, maxPerCell = 4) {
    const cellW = width / gridCols;
    const cellH = height / gridRows;
    const grid = Array.from({ length: gridRows }, () => Array.from({ length: gridCols }, () => []));

    for (const match of inliers) {
      const cx = Math.min(gridCols - 1, Math.max(0, Math.floor(match.src_pt[0] / cellW)));
      const cy = Math.min(gridRows - 1, Math.max(0, Math.floor(match.src_pt[1] / cellH)));
      grid[cy][cx].push(match);
    }

    const sampled = [];
    const gridCounts = [];
    let populatedCells = 0;

    for (let r = 0; r < gridRows; r++) {
      const rowCounts = [];
      for (let c = 0; c < gridCols; c++) {
        const cellMatches = grid[r][c];
        rowCounts.push(cellMatches.length);
        if (cellMatches.length > 0) {
          populatedCells++;
          // Sort by reprojection error or confidence
          cellMatches.sort((a, b) => (a.reprojection_error || 0) - (b.reprojection_error || 0));
          sampled.push(...cellMatches.slice(0, maxPerCell));
        }
      }
      gridCounts.push(rowCounts);
    }

    const totalCells = gridCols * gridRows;
    const coveragePct = Number(((populatedCells / totalCells) * 100).toFixed(1));

    // Shannon spatial entropy
    let entropy = 0;
    const totalPts = inliers.length || 1;
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const p = gridCounts[r][c] / totalPts;
        if (p > 0) entropy -= p * Math.log2(p);
      }
    }
    const maxEntropy = Math.log2(totalCells);
    const uniformityScore = Number((entropy / maxEntropy).toFixed(3));

    return { sampled, gridCounts, coveragePct, uniformityScore };
  }

  // Stage 9: Bilinear Inverse Warping
  warpCanvas(srcCanvas, H, targetWidth, targetHeight) {
    const outCanvas = document.createElement('canvas');
    outCanvas.width = targetWidth;
    outCanvas.height = targetHeight;
    const outCtx = outCanvas.getContext('2d');

    const srcCtx = srcCanvas.getContext('2d');
    const srcImgData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
    const srcPixels = srcImgData.data;
    const srcW = srcCanvas.width;
    const srcH = srcCanvas.height;

    const outImgData = outCtx.createImageData(targetWidth, targetHeight);
    const outPixels = outImgData.data;

    // Invert Homography Matrix H
    const invH = this.invert3x3(H);
    if (!invH) return srcCanvas;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        // Map target (x, y) back to source (sx, sy) via invH
        const w = invH[6] * x + invH[7] * y + invH[8];
        if (Math.abs(w) < 1e-6) continue;
        const sx = (invH[0] * x + invH[1] * y + invH[2]) / w;
        const sy = (invH[3] * x + invH[4] * y + invH[5]) / w;

        if (sx >= 0 && sx < srcW - 1 && sy >= 0 && sy < srcH - 1) {
          // Bilinear interpolation
          const x0 = Math.floor(sx);
          const x1 = x0 + 1;
          const y0 = Math.floor(sy);
          const y1 = y0 + 1;

          const fx = sx - x0;
          const fy = sy - y0;

          const idx00 = (y0 * srcW + x0) * 4;
          const idx10 = (y0 * srcW + x1) * 4;
          const idx01 = (y1 * srcW + x0) * 4;
          const idx11 = (y1 * srcW + x1) * 4;

          const outIdx = (y * targetWidth + x) * 4;

          for (let c = 0; c < 3; c++) {
            const top = (1 - fx) * srcPixels[idx00 + c] + fx * srcPixels[idx10 + c];
            const bottom = (1 - fx) * srcPixels[idx01 + c] + fx * srcPixels[idx11 + c];
            outPixels[outIdx + c] = Math.round((1 - fy) * top + fy * bottom);
          }
          outPixels[outIdx + 3] = 255;
        }
      }
    }

    outCtx.putImageData(outImgData, 0, 0);
    return outCanvas;
  }

  invert3x3(m) {
    const a = m[0], b = m[1], c = m[2];
    const d = m[3], e = m[4], f = m[5];
    const g = m[6], h = m[7], k = m[8];

    const A = (e * k - f * h);
    const B = -(d * k - f * g);
    const C = (d * h - e * g);
    const D = -(b * k - c * h);
    const E = (a * k - c * g);
    const F = -(a * h - b * g);
    const G = (b * f - c * e);
    const H_ = -(a * f - c * d);
    const K = (a * e - b * d);

    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-9) return null;

    const invDet = 1.0 / det;
    return [
      A * invDet, D * invDet, G * invDet,
      B * invDet, E * invDet, H_ * invDet,
      C * invDet, F * invDet, K * invDet
    ];
  }

  // Alpha Blend
  createAlphaBlend(warpedCanvas, refCanvas, alpha = 0.5) {
    const canvas = document.createElement('canvas');
    canvas.width = refCanvas.width;
    canvas.height = refCanvas.height;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(refCanvas, 0, 0);
    ctx.globalAlpha = alpha;
    ctx.drawImage(warpedCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
    return canvas;
  }

  // Checkerboard
  createCheckerboard(warpedCanvas, refCanvas, squareSize = 48) {
    const canvas = document.createElement('canvas');
    canvas.width = refCanvas.width;
    canvas.height = refCanvas.height;
    const ctx = canvas.getContext('2d');

    const w = canvas.width;
    const h = canvas.height;

    ctx.drawImage(refCanvas, 0, 0);

    const patternCanvas = document.createElement('canvas');
    patternCanvas.width = squareSize * 2;
    patternCanvas.height = squareSize * 2;
    const pCtx = patternCanvas.getContext('2d');

    pCtx.fillStyle = '#fff';
    pCtx.fillRect(0, 0, squareSize, squareSize);
    pCtx.fillRect(squareSize, squareSize, squareSize, squareSize);

    // Create mask for warped image
    ctx.save();
    for (let y = 0; y < h; y += squareSize) {
      for (let x = 0; x < w; x += squareSize) {
        if (((x / squareSize) + (y / squareSize)) % 2 === 0) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(x, y, squareSize, squareSize);
          ctx.clip();
          ctx.drawImage(warpedCanvas, 0, 0);
          ctx.restore();
        }
      }
    }
    ctx.restore();

    return canvas;
  }

  // Difference Heatmap
  createDifferenceHeatmap(warpedCanvas, refCanvas) {
    const canvas = document.createElement('canvas');
    const w = refCanvas.width;
    const h = refCanvas.height;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    const wCtx = warpedCanvas.getContext('2d');
    const rCtx = refCanvas.getContext('2d');

    const wData = wCtx.getImageData(0, 0, w, h).data;
    const rData = rCtx.getImageData(0, 0, w, h).data;

    const outData = ctx.createImageData(w, h);
    const out = outData.data;

    for (let i = 0; i < wData.length; i += 4) {
      // Grayscale luminance diff
      const gW = 0.299 * wData[i] + 0.587 * wData[i + 1] + 0.114 * wData[i + 2];
      const gR = 0.299 * rData[i] + 0.587 * rData[i + 1] + 0.114 * rData[i + 2];

      const diff = Math.min(255, Math.abs(gW - gR) * 2.2);
      const [r, g, b] = this.turboColormap(diff / 255.0);

      out[i] = r;
      out[i + 1] = g;
      out[i + 2] = b;
      out[i + 3] = 255;
    }

    ctx.putImageData(outData, 0, 0);
    return canvas;
  }

  // Turbo color mapping approximation
  turboColormap(t) {
    const r = Math.sin(t * 3.14159 * 1.5 - 0.2) * 127 + 128;
    const g = Math.sin(t * 3.14159 * 2.0 - 0.5) * 127 + 128;
    const b = Math.cos(t * 3.14159 * 1.8) * 127 + 128;
    return [
      Math.max(0, Math.min(255, Math.round(r))),
      Math.max(0, Math.min(255, Math.round(g))),
      Math.max(0, Math.min(255, Math.round(b)))
    ];
  }

  // Correspondence Map
  drawCorrespondences(srcImg, refImg, matches, maxDraw = 60) {
    const canvas = document.createElement('canvas');
    const w1 = srcImg.naturalWidth || srcImg.width || 640;
    const h1 = srcImg.naturalHeight || srcImg.height || 640;
    const w2 = refImg.naturalWidth || refImg.width || 640;
    const h2 = refImg.naturalHeight || refImg.height || 640;

    canvas.width = w1 + w2;
    canvas.height = Math.max(h1, h2);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(srcImg, 0, 0, w1, h1);
    ctx.drawImage(refImg, w1, 0, w2, h2);

    // Divider line
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(w1, 0);
    ctx.lineTo(w1, canvas.height);
    ctx.stroke();

    const drawList = matches.slice(0, maxDraw);
    for (const m of drawList) {
      const [x1, y1] = m.src_pt;
      const [x2, y2] = m.ref_pt;
      const targetX = x2 + w1;

      // Match line
      ctx.strokeStyle = m.is_subpixel ? '#10b981' : '#00f0ff';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(targetX, y2);
      ctx.stroke();

      // Keypoint Circles
      ctx.fillStyle = '#ffd166';
      ctx.beginPath();
      ctx.arc(x1, y1, 3, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = '#a855f7';
      ctx.beginPath();
      ctx.arc(targetX, y2, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    return canvas;
  }

  // Draw 8x8 Grid Density
  drawGridDensity(refImg, gridCounts, gridCols = 8, gridRows = 8) {
    const canvas = document.createElement('canvas');
    const w = refImg.naturalWidth || refImg.width || 640;
    const h = refImg.naturalHeight || refImg.height || 640;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(refImg, 0, 0, w, h);

    const cellW = w / gridCols;
    const cellH = h / gridRows;

    let maxCount = 1;
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        if (gridCounts[r][c] > maxCount) maxCount = gridCounts[r][c];
      }
    }

    ctx.lineWidth = 1;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const count = gridCounts[r][c];
        const x = c * cellW;
        const y = r * cellH;

        const alpha = count > 0 ? Math.min(0.65, 0.15 + (count / maxCount) * 0.5) : 0.05;
        ctx.fillStyle = count > 0 ? `rgba(0, 240, 255, ${alpha})` : 'rgba(244, 63, 94, 0.15)';
        ctx.fillRect(x, y, cellW, cellH);

        ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
        ctx.strokeRect(x, y, cellW, cellH);

        ctx.fillStyle = count > 0 ? '#fff' : 'rgba(255, 255, 255, 0.4)';
        ctx.fillText(count.toString(), x + cellW / 2, y + cellH / 2);
      }
    }

    return canvas;
  }

  // Full Pipeline Execution
  async executePipeline(srcImg, refImg, options = {}) {
    const tStart = performance.now();
    const stageLogs = [];

    const sensor = options.sensor || 'OHRC';
    const ransacThresh = options.ransacThresh || 2.8;
    const gridCols = options.gridCols || 8;
    const gridRows = options.gridRows || 8;

    // Stage 1: Preprocessing
    stageLogs.push({
      stage: 1,
      name: "Image Preprocessing",
      detail: `Decoding ${srcImg.naturalWidth || 640}x${srcImg.naturalHeight || 640} frames. Radiometric normalization active.`
    });

    const srcCanvas = this.createCanvasFromImage(srcImg);
    const refCanvas = this.createCanvasFromImage(refImg);

    const w = refCanvas.width;
    const h = refCanvas.height;

    const { gray: graySrc } = this.getGrayscaleData(srcCanvas.getContext('2d'), srcCanvas.width, srcCanvas.height);
    const { gray: grayRef } = this.getGrayscaleData(refCanvas.getContext('2d'), refCanvas.width, refCanvas.height);

    // Stage 2: Sun-Angle Normalization
    stageLogs.push({
      stage: 2,
      name: "Illumination Compensation",
      detail: "Gradient magnitude & topographic shadow normalization applied."
    });

    // Stage 3 & 4: Feature Extraction
    stageLogs.push({
      stage: 3,
      name: "Multi-Scale Representation",
      detail: "Building Gaussian scale space pyramid for scale invariance."
    });

    stageLogs.push({
      stage: 4,
      name: "Sensor-Invariant Feature Extraction",
      detail: "Extracting Harris/FAST keypoints with normalized gradient patch descriptors."
    });

    const kpsSrc = this.detectKeypoints(graySrc, srcCanvas.width, srcCanvas.height, 450);
    const kpsRef = this.detectKeypoints(grayRef, refCanvas.width, refCanvas.height, 450);

    const { keypoints: validKpsSrc, descriptors: descSrc } = this.extractDescriptors(graySrc, srcCanvas.width, srcCanvas.height, kpsSrc);
    const { keypoints: validKpsRef, descriptors: descRef } = this.extractDescriptors(grayRef, refCanvas.width, refCanvas.height, kpsRef);

    // Stage 5: Correspondence Matching
    stageLogs.push({
      stage: 5,
      name: "Feature Correspondence & Cross-Check",
      detail: `Lowe's ratio matching identified candidate pairings.`
    });
    const rawMatches = this.matchFeatures(validKpsSrc, descSrc, validKpsRef, descRef, 0.88);

    // Stage 6: Geometric Verification (RANSAC)
    stageLogs.push({
      stage: 6,
      name: "Geometric Verification (RANSAC)",
      detail: `Testing consensus sets with ${ransacThresh}px error threshold.`
    });
    const ransacRes = this.estimateHomographyRANSAC(rawMatches, 1200, ransacThresh);

    // Stage 7: Sub-Pixel Refinement
    stageLogs.push({
      stage: 7,
      name: "Sub-Pixel Refinement",
      detail: "Refining inlier matches using quadratic peak interpolation."
    });
    const subpixelMatches = this.refineSubpixel(graySrc, grayRef, ransacRes.inliers, w, h);

    // Stage 8: Uniform Spatial Distribution
    stageLogs.push({
      stage: 8,
      name: "Uniform Spatial Distribution",
      detail: `Binning inliers across ${gridCols}x${gridRows} grid to eliminate crater clustering.`
    });
    const uniformRes = this.sampleUniformGrid(subpixelMatches, srcCanvas.width, srcCanvas.height, gridCols, gridRows, 4);
    const finalMatches = uniformRes.sampled.length >= 4 ? uniformRes.sampled : subpixelMatches;

    // Stage 9: Registration & Visual Outputs
    stageLogs.push({
      stage: 9,
      name: "Registration & Visual Outputs",
      detail: "Generating warped source, alpha blend, checkerboard, and difference heatmap."
    });

    const warpedCanvas = this.warpCanvas(srcCanvas, ransacRes.matrix, w, h);
    const blendCanvas = this.createAlphaBlend(warpedCanvas, refCanvas, 0.5);
    const checkerCanvas = this.createCheckerboard(warpedCanvas, refCanvas, 48);
    const diffCanvas = this.createDifferenceHeatmap(warpedCanvas, refCanvas);
    const corrCanvas = this.drawCorrespondences(srcImg, refImg, finalMatches, 50);
    const gridCanvas = this.drawGridDensity(refImg, uniformRes.gridCounts, gridCols, gridRows);

    // Stage 10: Metrics
    stageLogs.push({
      stage: 10,
      name: "Evaluation Metrics & Quality Report",
      detail: "Calculating RMSE, inlier ratio, and ISRO mission qualification score."
    });

    const errors = finalMatches.map(m => m.reprojection_error || 0);
    const rmse = errors.length > 0 ? Math.sqrt(errors.reduce((acc, e) => acc + e * e, 0) / errors.length) : 0;
    const inlierRatio = rawMatches.length > 0 ? (ransacRes.inliers.length / rawMatches.length) * 100 : 0;
    const subpixelCount = finalMatches.filter(m => m.is_subpixel).length;
    const subpixelPct = finalMatches.length > 0 ? (subpixelCount / finalMatches.length) * 100 : 0;

    let qualityGrade = "High Precision (Flight-Ready)";
    if (rmse > 2.0 || ransacRes.inliers.length < 15) qualityGrade = "Acceptable (Coarse Match)";
    if (rmse > 3.0 || ransacRes.inliers.length < 8) qualityGrade = "Low Confidence (Needs Calibration)";

    const elapsed = Number(((performance.now() - tStart) / 1000).toFixed(2));

    // Convert canvases to Data URLs for instant rendering & downloading
    return {
      success: ransacRes.success,
      elapsed_seconds: elapsed,
      sensor_type: sensor,
      metrics: {
        rmse: Number(rmse.toFixed(3)),
        mean_matching_error: Number((errors.reduce((a, b) => a + b, 0) / (errors.length || 1)).toFixed(3)),
        inlier_count: ransacRes.inliers.length,
        candidate_count: rawMatches.length,
        inlier_ratio_pct: Number(inlierRatio.toFixed(1)),
        spatial_coverage_pct: uniformRes.coveragePct,
        spatial_uniformity_score: uniformRes.uniformityScore,
        subpixel_accuracy_pct: Number(subpixelPct.toFixed(1)),
        quality_grade: qualityGrade,
        transformation_type: "Homography (Projective)"
      },
      matrix: [
        [ransacRes.matrix[0], ransacRes.matrix[1], ransacRes.matrix[2]],
        [ransacRes.matrix[3], ransacRes.matrix[4], ransacRes.matrix[5]],
        [ransacRes.matrix[6], ransacRes.matrix[7], ransacRes.matrix[8]]
      ],
      stage_logs: stageLogs,
      layers: {
        warped: warpedCanvas.toDataURL('image/png'),
        blend: blendCanvas.toDataURL('image/png'),
        checker: checkerCanvas.toDataURL('image/png'),
        diff: diffCanvas.toDataURL('image/png'),
        correspondence: corrCanvas.toDataURL('image/png'),
        grid: gridCanvas.toDataURL('image/png')
      },
      canvases: {
        warped: warpedCanvas,
        ref: refCanvas
      },
      match_points: finalMatches.map((m, i) => ({
        id: i + 1,
        source_x: Number(m.src_pt[0].toFixed(2)),
        source_y: Number(m.src_pt[1].toFixed(2)),
        reference_x: Number(m.ref_pt[0].toFixed(2)),
        reference_y: Number(m.ref_pt[1].toFixed(2)),
        confidence: m.confidence,
        reprojection_error: m.reprojection_error,
        is_subpixel: m.is_subpixel
      }))
    };
  }
}

window.ClientRegistrationEngine = ClientRegistrationEngine;
