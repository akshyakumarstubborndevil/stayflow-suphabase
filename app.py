"""
LunaMatch-R - Python Flask API & Local Development Server
Provides REST API endpoints for the LunaMatch image registration engine.
Can be run locally with 'python app.py' or deployed to Render / Railway / Hugging Face.
"""

import os
import time
import json
import base64
from flask import Flask, request, jsonify, send_from_directory
from engine.pipeline import LunaMatchPipeline

app = Flask(__name__, static_folder="static", template_folder=".")

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static/preset_outputs", exist_ok=True)

pipeline = LunaMatchPipeline(output_dir="static/preset_outputs")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/outputs/<path:path>")
def serve_outputs(path):
    return send_from_directory("outputs", path)


@app.route("/api/presets", methods=["GET"])
def get_presets():
    presets_file = "static/presets_data.json"
    if os.path.exists(presets_file):
        with open(presets_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Presets file not found"}), 404


@app.route("/api/register", methods=["POST"])
def register_images():
    try:
        data = request.form
        sensor_type = data.get("sensor_type", "OHRC")
        geometric_model = data.get("geometric_model", "homography")
        ransac_thresh = float(data.get("ransac_thresh", 2.5))
        grid_cols = int(data.get("grid_cols", 8))
        grid_rows = int(data.get("grid_rows", 8))

        ts = int(time.time() * 1000)
        src_path = None
        ref_path = None

        if "source" in request.files and "reference" in request.files:
            file_src = request.files["source"]
            file_ref = request.files["reference"]
            src_path = os.path.join("uploads", f"src_{ts}_{file_src.filename}")
            ref_path = os.path.join("uploads", f"ref_{ts}_{file_ref.filename}")
            file_src.save(src_path)
            file_ref.save(ref_path)
        elif "preset_id" in data:
            pid = data["preset_id"]
            src_path = f"static/samples/{pid}_source.png"
            ref_path = f"static/samples/{pid}_reference.png"

        if not src_path or not ref_path or not os.path.exists(src_path) or not os.path.exists(ref_path):
            return jsonify({"error": "Valid source and reference images are required"}), 400

        result = pipeline.run(
            img_src_path=src_path,
            img_ref_path=ref_path,
            sensor_type=sensor_type,
            geometric_model=geometric_model,
            ransac_thresh=ransac_thresh,
            grid_cols=grid_cols,
            grid_rows=grid_rows
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting LunaMatch-R server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
