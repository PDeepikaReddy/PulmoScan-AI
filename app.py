"""
Flask web app for Tuberculosis chest X-ray screening.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import uuid

from flask import (
    Flask, jsonify, render_template, request, send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

import config
import model_utils
import remedies

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def _allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=model_utils.model_exists(),
        classes=config.CLASS_NAMES,
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": model_utils.model_exists()})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG/JPG."}), 400

    if not model_utils.model_exists():
        return jsonify({
            "error": (
                "The model has not been trained yet. Run "
                "`python download_data.py` then `python train.py` to create "
                "models/tb_model.h5, and restart the app."
            )
        }), 503

    # Save upload with a unique name.
    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    uid = uuid.uuid4().hex
    fname = f"{uid}{ext}"
    fpath = os.path.join(config.UPLOAD_DIR, fname)
    file.save(fpath)

    try:
        result = model_utils.predict_image(fpath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Grad-CAM explainability overlay (optional).
    gradcam_url = None
    cam_name = f"{uid}_cam.jpg"
    cam_path = os.path.join(config.UPLOAD_DIR, cam_name)
    if model_utils.generate_gradcam(fpath, cam_path):
        gradcam_url = url_for("uploaded_file", filename=cam_name)

    guidance = remedies.get_guidance(result["label"])

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "tb_probability": result["tb_probability"],
        "normal_probability": result["normal_probability"],
        "threshold": result["threshold"],
        "image_url": url_for("uploaded_file", filename=fname),
        "gradcam_url": gradcam_url,
        "guidance": guidance,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(config.UPLOAD_DIR, filename)


if __name__ == "__main__":
    print("=" * 60)
    print(" Tuberculosis Detection — Flask app")
    print(f" Model trained: {model_utils.model_exists()}")
    if not model_utils.model_exists():
        print(" >> No model yet. Run download_data.py then train.py.")
        print("    The UI will still load so you can preview it.")
    print(" Open: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
