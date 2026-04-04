import os
import uuid
import warnings
import numpy as np
import cv2
import base64
import io
import json
import time

warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# ── Model globals ──────────────────────────────────────────────────────────
model = None
model_status = {"loaded": False, "loading": False, "error": None, "path": None}

# Default model file — placed in the same folder as app.py
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "best_resunet_depth_model.h5")

IMG_HEIGHT = 256
IMG_WIDTH  = 256
MAX_DEPTH  = 10.0   # metres (display scale)

COLORMAPS = {
    "plasma":  cv2.COLORMAP_PLASMA,
    "magma":   cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo":   cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet":     cv2.COLORMAP_JET,
}

# ── Helpers ────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ndarray_to_b64(arr: np.ndarray, quality: int = 92) -> str:
    """Encode a uint8 BGR/RGB image array to a base64 JPEG string."""
    success, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf).decode('utf-8')


def apply_colormap(depth_norm: np.ndarray, cmap_name: str = "plasma") -> np.ndarray:
    """depth_norm: float32 [0,1] H×W → uint8 BGR H×W×3 coloured map."""
    d8 = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
    cmap_id = COLORMAPS.get(cmap_name, cv2.COLORMAP_PLASMA)
    return cv2.applyColorMap(d8, cmap_id)


def compute_metrics(depth: np.ndarray) -> dict:
    """Return simple statistics about a predicted depth map."""
    d = depth.flatten()
    eps = 1e-6
    return {
        "min":    float(np.min(d) * MAX_DEPTH),
        "max":    float(np.max(d) * MAX_DEPTH),
        "mean":   float(np.mean(d) * MAX_DEPTH),
        "std":    float(np.std(d) * MAX_DEPTH),
        "near_ratio": float(np.mean(d < 0.25) * 100),
        "mid_ratio":  float(np.mean((d >= 0.25) & (d < 0.75)) * 100),
        "far_ratio":  float(np.mean(d >= 0.75) * 100),
    }


def preprocess_image(image_path: str) -> tuple:
    """Load & preprocess an image → (original_bgr, model_input)."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    inp = cv2.resize(bgr, (IMG_WIDTH, IMG_HEIGHT))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    return bgr, inp


def run_inference(image_path: str, cmap_name: str = "plasma") -> dict:
    """Run model inference and return dict of base64-encoded images + metrics."""
    global model

    bgr_orig, inp = preprocess_image(image_path)

    # ── Run model (or fallback demo) ───────────────────────────────────────
    t0 = time.time()
    if model is not None:
        depth_pred = model.predict(inp, verbose=0).squeeze()  # [0,1] float
    else:
        # Demo mode: synthetic gradient depth for UI preview
        h, w = IMG_HEIGHT, IMG_WIDTH
        y, x  = np.mgrid[0:h, 0:w] / max(h, w)
        depth_pred = (0.3 + 0.5 * y + 0.2 * np.sin(x * 6)).astype(np.float32)
        depth_pred = np.clip(depth_pred, 0, 1)
    elapsed = time.time() - t0

    # ── Visualise ──────────────────────────────────────────────────────────
    orig_resized = cv2.resize(bgr_orig, (IMG_WIDTH, IMG_HEIGHT))
    depth_colored = apply_colormap(depth_pred, cmap_name)

    # side-by-side composite (extra visual)
    composite = np.concatenate([orig_resized, depth_colored], axis=1)

    # depth greyscale (for download-like preview)
    depth_grey = (np.clip(depth_pred, 0, 1) * 255).astype(np.uint8)
    depth_grey_bgr = cv2.cvtColor(depth_grey, cv2.COLOR_GRAY2BGR)

    metrics = compute_metrics(depth_pred)
    metrics["inference_ms"] = round(elapsed * 1000, 1)
    metrics["demo_mode"] = model is None

    return {
        "original":      ndarray_to_b64(orig_resized),
        "depth_colored": ndarray_to_b64(depth_colored),
        "depth_grey":    ndarray_to_b64(depth_grey_bgr),
        "composite":     ndarray_to_b64(composite),
        "metrics":       metrics,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/model/status')
def model_status_route():
    return jsonify(model_status)


def _build_custom_objects():
    """Return the dict of custom Keras objects used in the notebook."""
    import tensorflow as tf

    def depth_loss_mae(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def depth_loss_rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def depth_accuracy_delta1(y_true, y_pred):
        eps = 1e-6
        thresh = tf.maximum((y_true + eps) / (y_pred + eps),
                            (y_pred + eps) / (y_true + eps))
        return tf.reduce_mean(tf.cast(thresh < 1.25, tf.float32))

    def depth_accuracy_delta2(y_true, y_pred):
        eps = 1e-6
        thresh = tf.maximum((y_true + eps) / (y_pred + eps),
                            (y_pred + eps) / (y_true + eps))
        return tf.reduce_mean(tf.cast(thresh < 1.5625, tf.float32))

    def depth_accuracy_delta3(y_true, y_pred):
        eps = 1e-6
        thresh = tf.maximum((y_true + eps) / (y_pred + eps),
                            (y_pred + eps) / (y_true + eps))
        return tf.reduce_mean(tf.cast(thresh < 1.953125, tf.float32))

    return {
        'depth_loss_mae':        depth_loss_mae,
        'depth_loss_rmse':       depth_loss_rmse,
        'depth_accuracy_delta1': depth_accuracy_delta1,
        'depth_accuracy_delta2': depth_accuracy_delta2,
        'depth_accuracy_delta3': depth_accuracy_delta3,
    }


def _load_model_from_path(model_path: str):
    """Load a .h5 or .keras model and return it. Raises on failure."""
    from tensorflow import keras
    ext = os.path.splitext(model_path)[1].lower()
    custom_objects = _build_custom_objects()

    if ext == '.h5':
        # Legacy HDF5 — compile=False avoids re-compiling with missing objects
        loaded = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
    else:
        # Native Keras format (.keras)
        loaded = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
        )
    return loaded


@app.route('/api/model/load', methods=['POST'])
def load_model():
    global model, model_status
    data = request.get_json(force=True)
    model_path = data.get('path', '').strip()

    if not model_path:
        return jsonify({"error": "No model path provided"}), 400

    if not os.path.exists(model_path):
        return jsonify({"error": f"File not found: {model_path}"}), 404

    model_status = {"loaded": False, "loading": True, "error": None, "path": model_path}
    try:
        model = _load_model_from_path(model_path)
        model_status = {"loaded": True, "loading": False, "error": None, "path": model_path}
        return jsonify({"success": True, "message": "Model loaded successfully"})
    except Exception as e:
        model_status = {"loaded": False, "loading": False, "error": str(e), "path": model_path}
        return jsonify({"error": str(e)}), 500


def _auto_load_default_model():
    """Called once at startup — silently loads DEFAULT_MODEL_PATH if it exists."""
    global model, model_status
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"[startup] Default model not found at: {DEFAULT_MODEL_PATH}")
        return
    print(f"[startup] Auto-loading model: {DEFAULT_MODEL_PATH}")
    try:
        model = _load_model_from_path(DEFAULT_MODEL_PATH)
        model_status = {"loaded": True, "loading": False, "error": None,
                        "path": DEFAULT_MODEL_PATH}
        print("[startup] Model loaded successfully ✓")
    except Exception as e:
        model_status = {"loaded": False, "loading": False, "error": str(e),
                        "path": DEFAULT_MODEL_PATH}
        print(f"[startup] Failed to load model: {e}")


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    cmap = request.form.get('colormap', 'plasma')

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename  = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        result = run_inference(save_path, cmap_name=cmap)
        # Clean up upload after inference
        os.remove(save_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/colormaps')
def list_colormaps():
    return jsonify(list(COLORMAPS.keys()))


if __name__ == '__main__':
    print("=" * 60)
    print("  Depth Estimation — Res-UNet Flask UI")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    _auto_load_default_model()
    app.run(debug=True, host='0.0.0.0', port=5000)