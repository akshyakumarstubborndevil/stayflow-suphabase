"""
LunaMatch - Realistic Lunar Dataset Generator
Generates three realistic Chandrayaan-2 test pairs:
1. OHRC vs LROC NAC (Sun-Angle Invariance: azimuth 45° vs 150°)
2. TMC-2 vs Lunar Reference (Scale & Viewpoint Invariance: 1.35x zoom, slant, rotation)
3. IIRS vs Optical Panchromatic (Multi-Modal / Spectral Invariance: infrared absorption contrast)
"""

import os
import cv2
import numpy as np


def generate_lunar_dem(size=640, num_craters=45, seed=42):
    """
    Generates a realistic Digital Elevation Model (DEM) of a lunar cratered terrain.
    """
    np.random.seed(seed)
    H, W = size, size
    
    # 1. Multi-octave background regolith undulation
    dem = np.zeros((H, W), dtype=np.float32)
    for scale, weight in [(128, 15.0), (64, 8.0), (32, 4.0), (16, 2.0)]:
        noise = np.random.randn(H // scale + 2, W // scale + 2).astype(np.float32)
        resized = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
        dem += resized * weight

    # 2. Add crater population (power-law size distribution)
    y, x = np.ogrid[:H, :W]
    
    # Major anchor craters
    craters = [
        (int(W * 0.35), int(H * 0.35), 75, 45.0),
        (int(W * 0.70), int(H * 0.65), 60, 35.0),
        (int(W * 0.25), int(H * 0.75), 45, 25.0),
        (int(W * 0.75), int(H * 0.25), 40, 22.0),
        (int(W * 0.50), int(H * 0.50), 32, 18.0),
    ]
    
    # Random smaller craters
    for _ in range(num_craters):
        cx = np.random.randint(40, W - 40)
        cy = np.random.randint(40, H - 40)
        r = int(np.random.pareto(2.0) * 10 + 8)
        r = min(r, 38)
        depth = r * 0.55
        craters.append((cx, cy, r, depth))

    for cx, cy, r, depth in craters:
        dist_sq = (x - cx)**2 + (y - cy)**2
        dist = np.sqrt(dist_sq)
        
        # Parabolic depression
        depression = np.maximum(0.0, 1.0 - (dist / r)**2) * (-depth)
        # Raised rim
        rim_width = 0.18 * r
        rim = np.exp(-((dist - r)**2) / (rim_width**2)) * (depth * 0.42)
        
        dem += depression + rim

    return dem


def render_shaded_relief(dem, sun_azimuth_deg, sun_elevation_deg, albedo_var=True):
    """
    Renders realistic lunar illumination using Lommel-Seeliger / Lambertian approximation.
    """
    H, W = dem.shape
    gx = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
    
    az_rad = np.radians(sun_azimuth_deg)
    el_rad = np.radians(sun_elevation_deg)
    
    # Sun direction vector
    sx = np.cos(az_rad) * np.cos(el_rad)
    sy = np.sin(az_rad) * np.cos(el_rad)
    sz = np.sin(el_rad)
    
    # Surface normal
    normal_z = 2.2
    norm = np.sqrt(gx**2 + gy**2 + normal_z**2)
    nx = -gx / norm
    ny = -gy / norm
    nz = normal_z / norm
    
    # Lommel-Seeliger / Lambertian reflectance
    cos_i = np.maximum(0.0, nx * sx + ny * sy + nz * sz)
    cos_e = nz  # Nadir viewing
    
    refl = cos_i / (cos_i + cos_e + 1e-4)
    
    # Ambient scatter & contrast stretch
    shaded = np.clip(refl * 230 + 15, 0, 255).astype(np.float32)
    
    if albedo_var:
        albedo = cv2.resize(np.random.randn(8, 8).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        albedo = cv2.normalize(albedo, None, 0.85, 1.15, cv2.NORM_MINMAX)
        shaded = shaded * albedo
        
    return np.clip(shaded, 0, 255).astype(np.uint8)


def generate_all_samples(output_dir="static/samples"):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating synthetic lunar test sets for Chandrayaan-2 registration...")

    # Base DEM
    dem1 = generate_lunar_dem(size=600, num_craters=40, seed=101)
    
    # ==========================================
    # PAIR 1: Chandrayaan-2 OHRC vs. LROC NAC (Sun Angle Invariance)
    # ==========================================
    ref_ohrc = render_shaded_relief(dem1, sun_azimuth_deg=45, sun_elevation_deg=38)
    src_ohrc_base = render_shaded_relief(dem1, sun_azimuth_deg=75, sun_elevation_deg=30)
    
    h, w = ref_ohrc.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 4.2, 1.02)
    M_rot[0, 2] += 14.0
    M_rot[1, 2] -= 8.0
    src_ohrc = cv2.warpAffine(src_ohrc_base, M_rot, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    
    noise = np.random.normal(0, 3.5, (h, w)).astype(np.float32)
    src_ohrc = np.clip(src_ohrc.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "ohrc_reference.png"), ref_ohrc)
    cv2.imwrite(os.path.join(output_dir, "ohrc_source.png"), src_ohrc)
    print("[OK] Pair 1 generated: OHRC vs LROC NAC (Sun-angle shift)")

    # ==========================================
    # PAIR 2: Chandrayaan-2 TMC-2 vs. Lunar Global Basemap (Scale & Perspective Viewpoint)
    # ==========================================
    dem2 = generate_lunar_dem(size=640, num_craters=55, seed=202)
    ref_tmc = render_shaded_relief(dem2, sun_azimuth_deg=60, sun_elevation_deg=40)
    
    pts1 = np.float32([[40, 40], [w - 40, 50], [w - 60, h - 50], [50, h - 40]])
    pts2 = np.float32([[20, 25], [w - 10, 65], [w - 30, h - 20], [35, h - 55]])
    H_persp = cv2.getPerspectiveTransform(pts1, pts2)
    
    src_tmc_base = render_shaded_relief(dem2, sun_azimuth_deg=75, sun_elevation_deg=35)
    src_tmc = cv2.warpPerspective(src_tmc_base, H_persp, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    cv2.imwrite(os.path.join(output_dir, "tmc_reference.png"), ref_tmc)
    cv2.imwrite(os.path.join(output_dir, "tmc_source.png"), src_tmc)
    print("[OK] Pair 2 generated: TMC-2 vs Basemap (Scale & Viewpoint)")

    # ==========================================
    # PAIR 3: Chandrayaan-2 IIRS vs. Panchromatic Basemap (Cross-Modal Hyperspectral)
    # ==========================================
    dem3 = generate_lunar_dem(size=600, num_craters=38, seed=303)
    ref_iirs = render_shaded_relief(dem3, sun_azimuth_deg=50, sun_elevation_deg=35)
    
    src_iirs_base = ref_iirs.copy().astype(np.float32)
    
    crater_mask = (src_iirs_base < 60)
    src_iirs_base[crater_mask] = 110.0 - src_iirs_base[crater_mask] * 0.6
    
    stripes = np.sin(np.linspace(0, 80 * np.pi, h)).reshape(-1, 1) * 6.0
    src_iirs_base += stripes
    
    src_iirs_blurred = cv2.GaussianBlur(src_iirs_base, (5, 5), 1.6)
    
    M_iirs = cv2.getRotationMatrix2D((w / 2, h / 2), -2.5, 0.98)
    M_iirs[0, 2] += 10.0
    M_iirs[1, 2] += 12.0
    src_iirs = cv2.warpAffine(src_iirs_blurred, M_iirs, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    src_iirs = np.clip(src_iirs, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, "iirs_reference.png"), ref_iirs)
    cv2.imwrite(os.path.join(output_dir, "iirs_source.png"), src_iirs)
    print("[OK] Pair 3 generated: IIRS vs Panchromatic (Cross-Modal Hyperspectral)")


if __name__ == "__main__":
    generate_all_samples()
