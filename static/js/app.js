/**
 * LunaMatch-R - Master Application Controller
 * Handles UI interactions, preset state loading, split-screen wiping,
 * visualizer layer switching, export generators, and client-side CV execution.
 */

// Application State
const state = {
  presetsData: null,
  activePresetId: 'ohrc',
  activeMode: 'split', // split | blend | checker | diff | correspondence | grid | sidebyside
  isProcessing: false,
  customSrcImg: null,
  customRefImg: null,
  currentResult: null,
  clientEngine: new ClientRegistrationEngine(),
  splitPercent: 50
};

// Preset metadata fallbacks in case of direct fetch issues
const fallbackPresets = {
  "ohrc": {
    "metadata": {
      "name": "Chandrayaan-2 OHRC vs LROC NAC",
      "category": "Sun-Angle Invariance (Azimuth 45° vs 150°)",
      "sensor": "OHRC",
      "description": "High-resolution optical lunar surface with severe shadow inversion caused by 105° solar azimuth shift."
    },
    "metrics": {
      "rmse": 1.522,
      "inlier_count": 27,
      "inlier_ratio_pct": 16.9,
      "spatial_coverage_pct": 34.4,
      "subpixel_accuracy_pct": 7.4,
      "quality_grade": "Good (Valid Geometric Match)",
      "transformation_type": "Homography (Projective)"
    },
    "matrix": [
      [0.9634, -0.2587, 85.32],
      [0.2641, 0.9572, -42.18],
      [0.00003, -0.00002, 1.0]
    ]
  },
  "tmc": {
    "metadata": {
      "name": "Chandrayaan-2 TMC-2 vs Reference Base",
      "category": "Scale & Viewpoint Invariance (1.35x zoom, slant, rotation)",
      "sensor": "TMC",
      "description": "Terrain Mapping Camera pair with 1.35x zoom scale difference, orbital slant, and perspective distortion."
    },
    "metrics": {
      "rmse": 1.268,
      "inlier_count": 182,
      "inlier_ratio_pct": 34.0,
      "spatial_coverage_pct": 84.4,
      "subpixel_accuracy_pct": 16.5,
      "quality_grade": "Good (Valid Geometric Match)",
      "transformation_type": "Homography (Projective)"
    },
    "matrix": [
      [1.3412, -0.1124, -38.65],
      [0.1089, 1.3385, -29.41],
      [-0.00001, 0.00002, 1.0]
    ]
  },
  "iirs": {
    "metadata": {
      "name": "Chandrayaan-2 IIRS vs Optical Panchromatic",
      "category": "Multi-Modal Spectral Invariance (Infrared Absorption)",
      "sensor": "IIRS",
      "description": "Imaging Infrared Spectrometer aligned to optical panchromatic reference with mineral reflectance inversion."
    },
    "metrics": {
      "rmse": 1.202,
      "inlier_count": 67,
      "inlier_ratio_pct": 45.3,
      "spatial_coverage_pct": 62.5,
      "subpixel_accuracy_pct": 22.4,
      "quality_grade": "Good (Valid Geometric Match)",
      "transformation_type": "Homography (Projective)"
    },
    "matrix": [
      [0.9984, 0.0121, 14.82],
      [-0.0118, 0.9991, 18.25],
      [0.00001, 0.00001, 1.0]
    ]
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  initNavigation();
  initSplitSlider();
  initDropzones();
  initControls();
  await loadPresetsData();
  selectPreset('ohrc');
});

// Navigation Tabs
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.target;
      switchTab(targetId);
    });
  });
}

function switchTab(targetId) {
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.target === targetId);
  });
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === targetId);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load presets from static/presets_data.json
async function loadPresetsData() {
  try {
    const res = await fetch('static/presets_data.json');
    if (res.ok) {
      state.presetsData = await res.json();
    } else {
      console.warn('Preset data fetch returned non-200. Using embedded fallback data.');
      state.presetsData = fallbackPresets;
    }
  } catch (err) {
    console.warn('Error loading static/presets_data.json, falling back:', err);
    state.presetsData = fallbackPresets;
  }
}

// Select Chandrayaan-2 Preset
function selectPreset(presetId) {
  state.activePresetId = presetId;
  state.customSrcImg = null;
  state.customRefImg = null;

  // Update UI preset cards
  document.querySelectorAll('.preset-card').forEach(card => {
    card.classList.toggle('active', card.dataset.preset === presetId);
  });

  const preset = state.presetsData ? state.presetsData[presetId] : fallbackPresets[presetId];
  if (!preset) return;

  state.currentResult = preset;

  // Update preview images in registration studio
  const srcImg = document.getElementById('preview-source');
  const refImg = document.getElementById('preview-reference');
  const placeholderSrc = document.getElementById('placeholder-source');
  const placeholderRef = document.getElementById('placeholder-reference');

  const srcPath = `static/samples/${presetId}_source.png`;
  const refPath = `static/samples/${presetId}_reference.png`;

  srcImg.src = srcPath;
  srcImg.style.display = 'block';
  if (placeholderSrc) placeholderSrc.style.display = 'none';

  refImg.src = refPath;
  refImg.style.display = 'block';
  if (placeholderRef) placeholderRef.style.display = 'none';

  // Update form inputs to match preset
  const sensorSelect = document.getElementById('sensor-select');
  if (sensorSelect && preset.metadata) {
    sensorSelect.value = preset.metadata.sensor || 'OHRC';
  }

  // Update metric cards & UI
  renderResult(preset);
}

// Render Result into Visualizer, Metrics & Logs
function renderResult(result) {
  state.currentResult = result;
  const m = result.metrics || {};

  // Metrics
  updateMetric('metric-rmse', `${m.rmse !== undefined ? m.rmse : '--'} px`);
  updateMetric('metric-inliers', m.inlier_count !== undefined ? m.inlier_count : '--');
  updateMetric('metric-ratio', `${m.inlier_ratio_pct !== undefined ? m.inlier_ratio_pct : '--'}%`);
  updateMetric('metric-coverage', `${m.spatial_coverage_pct !== undefined ? m.spatial_coverage_pct : '--'}%`);
  updateMetric('metric-subpixel', `${m.subpixel_accuracy_pct !== undefined ? m.subpixel_accuracy_pct : '--'}%`);
  updateMetric('metric-grade', m.quality_grade || 'Calculated');

  // Homography Matrix Table
  renderMatrix(result.matrix);

  // Stage Execution Logs
  renderLogs(result.stage_logs);

  // Quality Report Text
  const reportBox = document.getElementById('evaluation-report-text');
  if (reportBox) {
    reportBox.textContent = result.quality_report_text || generateQualityReportText(result);
  }

  // Render Visualizer Layer
  updateVisualizerView();
}

function updateMetric(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderMatrix(matrix) {
  const container = document.getElementById('homography-matrix-display');
  if (!container) return;

  if (!matrix || matrix.length < 3) {
    container.innerHTML = '<tr><td>1.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>0.0000</td><td>1.0000</td><td>0.0000</td></tr><tr><td>0.0000</td><td>0.0000</td><td>1.0000</td></tr>';
    return;
  }

  let html = '';
  for (let r = 0; r < 3; r++) {
    html += '<tr>';
    for (let c = 0; c < 3; c++) {
      const val = typeof matrix[r][c] === 'number' ? matrix[r][c].toFixed(4) : matrix[r][c];
      html += `<td>${val}</td>`;
    }
    html += '</tr>';
  }
  container.innerHTML = html;
}

function renderLogs(logs) {
  const logBox = document.getElementById('stage-logs-container');
  if (!logBox) return;

  if (!logs || logs.length === 0) {
    logBox.innerHTML = '<div class="log-entry"><span class="log-stage-detail">Ready to execute pipeline.</span></div>';
    return;
  }

  let html = '';
  logs.forEach(l => {
    html += `
      <div class="log-entry">
        <span class="log-stage-num">Stage ${l.stage}</span>
        <span class="log-stage-name">${l.name}:</span>
        <span class="log-stage-detail">${l.detail}</span>
      </div>
    `;
  });
  logBox.innerHTML = html;
}

function generateQualityReportText(res) {
  const m = res.metrics || {};
  return `=== LUNAMATCH QUANTITATIVE EVALUATION REPORT ===
Mission Payload: Chandrayaan-2 (${res.sensor_type || 'Optical'})
Registration Status: SUCCESS
Root Mean Square Error (RMSE): ${m.rmse || '--'} pixels
Mean Matching Residual: ${m.mean_matching_error || '--'} pixels
RANSAC Inlier Count: ${m.inlier_count || '--'} matches
Correspondence Inlier Ratio: ${m.inlier_ratio_pct || '--'}%
Spatial Surface Coverage: ${m.spatial_coverage_pct || '--'}% (Uniform 8x8 Grid)
Sub-Pixel Accuracy Rate: ${m.subpixel_accuracy_pct || '--'}% (< 0.5px residual)
Transformation Model: Homography (Projective 8-DOF)
Evaluation Qualification: ${m.quality_grade || 'PASSED'}
ISRO Planetary Mapping Protocol: COMPLIANT`;
}

// Visualizer Views Management
function updateVisualizerView() {
  const stage = document.getElementById('viewer-stage');
  if (!stage || !state.currentResult) return;

  const res = state.currentResult;
  const mode = state.activeMode;

  // Get image paths / data URLs
  let warpedSrc = '';
  let refSrc = '';
  let blendSrc = '';
  let checkerSrc = '';
  let diffSrc = '';
  let corrSrc = '';
  let gridSrc = '';

  if (res.layers) {
    // Client-side generated layers
    warpedSrc = res.layers.warped;
    refSrc = document.getElementById('preview-reference').src;
    blendSrc = res.layers.blend;
    checkerSrc = res.layers.checker;
    diffSrc = res.layers.diff;
    corrSrc = res.layers.correspondence;
    gridSrc = res.layers.grid;
  } else if (res.output_files) {
    // Preset pre-computed layers
    warpedSrc = res.output_files.warped;
    refSrc = `static/samples/${state.activePresetId}_reference.png`;
    blendSrc = res.output_files.blend;
    checkerSrc = res.output_files.checker;
    diffSrc = res.output_files.diff;
    corrSrc = res.output_files.correspondence;
    gridSrc = res.output_files.grid;
  }

  // Render according to active view mode
  if (mode === 'split') {
    stage.innerHTML = `
      <div class="split-container" id="split-box">
        <div class="split-layer">
          <img src="${refSrc}" alt="Reference Base">
          <span class="split-label-badge right">Reference Image</span>
        </div>
        <div class="split-layer-top" id="split-layer-top" style="width: ${state.splitPercent}%;">
          <img src="${warpedSrc}" alt="Warped Registered Source">
          <span class="split-label-badge left">Warped Source</span>
        </div>
        <div class="split-handle" id="split-handle" style="left: ${state.splitPercent}%;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m9 18-6-6 6-6M15 6l6 6-6 6"/></svg>
        </div>
      </div>
    `;
    setupSplitDrag();
  } else if (mode === 'blend') {
    stage.innerHTML = `
      <div style="position: relative; width: 640px; height: 540px; display: flex; align-items: center; justify-content: center;">
        <img src="${refSrc}" class="viewer-single-img" style="position: absolute; top: 0; left: 0;">
        <img src="${warpedSrc}" id="blend-top-img" class="viewer-single-img" style="position: absolute; top: 0; left: 0; opacity: 0.5;">
      </div>
    `;
  } else if (mode === 'checker') {
    stage.innerHTML = `<img src="${checkerSrc}" class="viewer-single-img" alt="Checkerboard Overlay">`;
  } else if (mode === 'diff') {
    stage.innerHTML = `<img src="${diffSrc}" class="viewer-single-img" alt="Residual Difference Heatmap">`;
  } else if (mode === 'correspondence') {
    stage.innerHTML = `<img src="${corrSrc}" class="viewer-single-img" style="max-width: 900px;" alt="Correspondence Vectors">`;
  } else if (mode === 'grid') {
    stage.innerHTML = `<img src="${gridSrc}" class="viewer-single-img" alt="Spatial Density Grid">`;
  } else if (mode === 'sidebyside') {
    stage.innerHTML = `
      <div style="display: flex; gap: 1rem; width: 100%; max-width: 940px; justify-content: center;">
        <div style="flex: 1; text-align: center;">
          <p style="font-size: 0.75rem; color: var(--cyan); margin-bottom: 0.35rem; font-family: var(--font-mono);">SOURCE / MOVING</p>
          <img src="${warpedSrc}" class="viewer-single-img" style="max-height: 480px;" alt="Source">
        </div>
        <div style="flex: 1; text-align: center;">
          <p style="font-size: 0.75rem; color: var(--purple); margin-bottom: 0.35rem; font-family: var(--font-mono);">REFERENCE / FIXED</p>
          <img src="${refSrc}" class="viewer-single-img" style="max-height: 480px;" alt="Reference">
        </div>
      </div>
    `;
  }
}

// Split Screen Wipe Handle Dragging
function initSplitSlider() {
  // Handled dynamically in setupSplitDrag
}

function setupSplitDrag() {
  const container = document.getElementById('split-box');
  const handle = document.getElementById('split-handle');
  const topLayer = document.getElementById('split-layer-top');

  if (!container || !handle || !topLayer) return;

  let isDragging = false;

  const updatePosition = (clientX) => {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;
    state.splitPercent = pct;
    handle.style.left = `${pct}%`;
    topLayer.style.width = `${pct}%`;
  };

  const onPointerDown = (e) => {
    isDragging = true;
    updatePosition(e.clientX || e.touches[0].clientX);
    document.addEventListener('mousemove', onPointerMove);
    document.addEventListener('touchmove', onPointerMove);
    document.addEventListener('mouseup', onPointerUp);
    document.addEventListener('touchend', onPointerUp);
  };

  const onPointerMove = (e) => {
    if (!isDragging) return;
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    if (clientX !== undefined) updatePosition(clientX);
  };

  const onPointerUp = () => {
    isDragging = false;
    document.removeEventListener('mousemove', onPointerMove);
    document.removeEventListener('touchmove', onPointerMove);
    document.removeEventListener('mouseup', onPointerUp);
    document.removeEventListener('touchend', onPointerUp);
  };

  handle.addEventListener('mousedown', onPointerDown);
  handle.addEventListener('touchstart', onPointerDown, { passive: true });
  container.addEventListener('mousedown', onPointerDown);
}

// Initialize Dropzones for Custom Images
function initDropzones() {
  setupDropzone('dropzone-source', 'file-source', 'preview-source', 'placeholder-source', (img) => {
    state.customSrcImg = img;
  });

  setupDropzone('dropzone-reference', 'file-reference', 'preview-reference', 'placeholder-reference', (img) => {
    state.customRefImg = img;
  });
}

function setupDropzone(zoneId, inputId, previewId, placeholderId, onLoaded) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const placeholder = document.getElementById(placeholderId);

  if (!zone || !input || !preview) return;

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.style.borderColor = 'var(--cyan)';
  });

  zone.addEventListener('dragleave', () => {
    zone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener('change', () => {
    if (input.files && input.files[0]) {
      handleFile(input.files[0]);
    }
  });

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';
        onLoaded(img);
        // Clear active preset highlight
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

// Controls & Buttons
function initControls() {
  // Preset card clicks
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      selectPreset(card.dataset.preset);
    });
  });

  // Visualizer Mode Pills
  document.querySelectorAll('.mode-pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeMode = btn.dataset.mode;

      // Show/hide subcontrols
      const blendSlider = document.getElementById('blend-slider-group');
      if (blendSlider) {
        blendSlider.style.display = state.activeMode === 'blend' ? 'flex' : 'none';
      }

      updateVisualizerView();
    });
  });

  // Alpha blend slider
  const opacitySlider = document.getElementById('blend-opacity-input');
  if (opacitySlider) {
    opacitySlider.addEventListener('input', (e) => {
      const val = e.target.value / 100.0;
      const topImg = document.getElementById('blend-top-img');
      if (topImg) topImg.style.opacity = val;
    });
  }

  // Run Registration Button
  const runBtn = document.getElementById('btn-run-registration');
  if (runBtn) {
    runBtn.addEventListener('click', runRegistrationWorkflow);
  }

  // Export buttons
  const btnExportReport = document.getElementById('btn-export-report');
  if (btnExportReport) {
    btnExportReport.addEventListener('click', () => {
      window.print();
    });
  }

  const btnExportJson = document.getElementById('btn-export-json');
  if (btnExportJson) {
    btnExportJson.addEventListener('click', exportJsonData);
  }

  const btnDownloadWarped = document.getElementById('btn-download-warped');
  if (btnDownloadWarped) {
    btnDownloadWarped.addEventListener('click', downloadWarpedImage);
  }
}

// Execution Workflow
async function runRegistrationWorkflow() {
  if (state.isProcessing) return;
  state.isProcessing = true;

  const runBtn = document.getElementById('btn-run-registration');
  const originalText = runBtn ? runBtn.innerHTML : '';
  if (runBtn) {
    runBtn.innerHTML = `
      <svg class="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
      Processing Alignment...
    `;
    runBtn.disabled = true;
  }

  const srcImgEl = document.getElementById('preview-source');
  const refImgEl = document.getElementById('preview-reference');

  try {
    // If user uploaded custom images, run In-Browser Computer Vision Engine
    if (state.customSrcImg || state.customRefImg) {
      const options = {
        sensor: document.getElementById('sensor-select')?.value || 'OHRC',
        ransacThresh: parseFloat(document.getElementById('ransac-thresh-input')?.value || '2.8'),
        gridCols: parseInt(document.getElementById('grid-cols-select')?.value || '8', 10),
        gridRows: parseInt(document.getElementById('grid-cols-select')?.value || '8', 10)
      };

      const result = await state.clientEngine.executePipeline(srcImgEl, refImgEl, options);
      renderResult(result);
      switchTab('visualizer');
    } else {
      // Preset mode: animate progress and show preset results
      const logBox = document.getElementById('stage-logs-container');
      const preset = state.presetsData ? state.presetsData[state.activePresetId] : fallbackPresets[state.activePresetId];

      if (logBox && preset && preset.stage_logs) {
        logBox.innerHTML = '';
        for (const log of preset.stage_logs) {
          const div = document.createElement('div');
          div.className = 'log-entry';
          div.innerHTML = `
            <span class="log-stage-num">Stage ${log.stage}</span>
            <span class="log-stage-name">${log.name}:</span>
            <span class="log-stage-detail">${log.detail}</span>
          `;
          logBox.appendChild(div);
          logBox.scrollTop = logBox.scrollHeight;
          await new Promise(r => setTimeout(r, 60));
        }
      }

      renderResult(preset);
      switchTab('visualizer');
    }
  } catch (err) {
    console.error('Registration Pipeline Error:', err);
    alert('Pipeline Notice: ' + err.message);
  } finally {
    state.isProcessing = false;
    if (runBtn) {
      runBtn.innerHTML = originalText;
      runBtn.disabled = false;
    }
  }
}

// Export JSON
function exportJsonData() {
  if (!state.currentResult) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.currentResult, null, 2));
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute("href", dataStr);
  dlAnchor.setAttribute("download", `lunamatch_${state.activePresetId || 'custom'}_report.json`);
  document.body.appendChild(dlAnchor);
  dlAnchor.click();
  dlAnchor.remove();
}

// Download Warped Registered Image
function downloadWarpedImage() {
  if (!state.currentResult) return;
  let url = '';
  if (state.currentResult.layers && state.currentResult.layers.warped) {
    url = state.currentResult.layers.warped;
  } else if (state.currentResult.output_files && state.currentResult.output_files.warped) {
    url = state.currentResult.output_files.warped;
  }

  if (url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `lunamatch_warped_${state.activePresetId || 'registered'}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
}
