# LunaMatch-R: Multi-Modal Lunar Image Registration System
### Developed for Chandrayaan-2 Payload Alignment (OHRC, TMC-2, IIRS)

LunaMatch-R is a planetary surface image registration platform engineered to align lunar imagery across extreme solar illumination variations, scale differences, and multi-modal spectral bands.

---

## 🚀 How to Deploy to Netlify

LunaMatch-R is built as a Jamstack & Client-Side Computer Vision application. It requires **zero build commands**, no npm packages, and runs instantly on Netlify's global CDN!

### Method 1: Netlify Drop (Easiest — 30 Seconds)
1. Go to [https://app.netlify.com/drop](https://app.netlify.com/drop) (log in if prompted).
2. Open your File Explorer on your computer.
3. Drag and drop the **`lunamatch`** folder directly into the browser dropzone.
4. Netlify will deploy your site in seconds and give you a live production URL (e.g. `https://your-site-name.netlify.app`)!

### Method 2: Git Repository (Continuous Deployment)
1. Initialize a git repository in this folder:
   ```bash
   git init
   git add .
   git commit -m "Deploy LunaMatch to Netlify"
   ```
2. Push to your GitHub / GitLab account.
3. In Netlify, click **"Add new site" -> "Import an existing project"** and connect your repository.
4. Build settings:
   - **Build command**: *(leave blank)*
   - **Publish directory**: `.` *(configured automatically via `netlify.toml`)*
5. Click **"Deploy site"**. Every future `git push` will update the live site automatically!

---

## 💻 Local Development (Optional)

You can view the site locally in any of the following ways:

### 1. Direct Browser (No Server Needed)
Simply double-click `index.html` to open it in Chrome, Edge, or Firefox.

### 2. Python Flask Server
If you want to run the full Python OpenCV pipeline locally:
```bash
pip install -r requirements.txt
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌟 Key Features
- **Lommel-Seeliger Sun-Angle Invariance**: Eliminates shadow flipping between different orbital passes (e.g. OHRC 45° vs LROC NAC 150°).
- **Scale & Viewpoint Invariance**: Handles 1.35x zoom and orbital slants.
- **Multi-Modal Spectral Invariance**: Aligns IIRS infrared absorption bands with optical panchromatic reference maps.
- **Interactive Split-Screen Wipe Slider**: Drag the interactive divider to inspect crater rim alignment down to sub-pixel accuracy.
- **Alpha Blending & Checkerboard**: Real-time opacity slider and adjustable checkerboard grid.
- **Residual Difference Heatmap**: Turbo colormap displaying photometric error residuals.
- **Interactive Match Vectors**: Hover over correspondence points to inspect coordinate coordinates and reprojection errors.
- **Uniform 8x8 Spatial Binning**: Guarantees wide surface distribution and prevents crater floor clustering.
- **In-Browser Client-Side CV Engine**: Upload your own custom lunar images directly on Netlify; the client-side JavaScript engine aligns, warps, and calculates RMSE on the fly!
