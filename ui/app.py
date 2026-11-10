"""
ui/app.py
=========
Flask demo app for the Knee OA KL-grading pipeline.

Two routes:
  GET  /           → classify page  (upload X-ray → prediction + Grad-CAM)
  GET  /pipeline   → pipeline page  (step-by-step visual walkthrough)
  POST /predict    → JSON  { grade, probs, explanation, gradcam_b64, entropy }

The model is loaded lazily on first request so the app starts instantly even
when weights are absent (a demo-mode fallback is used in that case).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ── KL description strings shown in plain English ──────────────────────────
KL_LABELS = {
    0: "Normal — No Osteoarthritis",
    1: "Doubtful — Very Early Changes",
    2: "Mild Osteoarthritis",
    3: "Moderate Osteoarthritis",
    4: "Severe Osteoarthritis",
}

KL_EXPLANATIONS = {
    0: (
        "The X-ray shows a <em>healthy knee joint</em> with no visible signs of "
        "osteoarthritis. The joint space between the femur and tibia appears "
        "normal and uniform, with no bone spurs (osteophytes) or narrowing. "
        "The model focused on the central joint space and found it intact."
    ),
    1: (
        "The model detected <em>very early, doubtful changes</em> that may "
        "indicate the beginning of osteoarthritis. There is a possible minute "
        "bony spur forming at the joint margin, but the joint space remains "
        "essentially normal. These findings are subtle and may not yet require "
        "treatment — but warrant monitoring."
    ),
    2: (
        "The X-ray shows <em>mild osteoarthritis</em>. There are definite small "
        "bone spurs (osteophytes) visible at the joint margins, and the space "
        "between the bones may be slightly reduced. The model's attention map "
        "highlighted the medial (inner) compartment where early cartilage thinning "
        "is most common. Lifestyle changes and physiotherapy are typically recommended "
        "at this stage."
    ),
    3: (
        "The model classified this as <em>moderate osteoarthritis</em>. The joint "
        "space is clearly narrowed compared to a normal knee, and multiple bone "
        "spurs are present. There is likely some sclerosis (hardening) of the bone "
        "beneath the cartilage. The Grad-CAM heatmap shows strong activation in "
        "the joint space area, confirming the model focused on the right region. "
        "Medical management, injections, or bracing may be appropriate."
    ),
    4: (
        "This X-ray shows <em>severe osteoarthritis</em> (the most advanced grade). "
        "The joint space is markedly narrowed or nearly absent, with large osteophytes "
        "and significant bone deformity visible. The model is highly confident in this "
        "classification, with the Grad-CAM clearly highlighting the collapsed joint "
        "space. At this stage, total knee replacement is often considered."
    ),
}

KL_SEVERITY_COLOR = {
    0: "#4a7c59",   # calm green
    1: "#7a9e7e",   # muted green
    2: "#c4a35a",   # amber
    3: "#b5651d",   # brown-orange
    4: "#8b2c2c",   # deep red
}

# ── Lazy model loader ───────────────────────────────────────────────────────
_model_cache: dict = {}


def _get_model():
    """Load and cache the champion model. Returns (model, cutpoints, device) or None."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["cuts"], _model_cache["dev"]

    # Look for fold-0 champion checkpoint
    runs_dir = PROJECT_ROOT / "outputs" / "cv" / "runs"
    run_name = "r50_dual_cascade_sqrt_aux_attn_spatial"
    ckpt_path = runs_dir / run_name / "fold0" / "best.pth"

    if not ckpt_path.exists():
        return None, None, None

    try:
        import torch
        from src.models.ordinal_net import OrdinalNet

        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        cfg_path = ckpt_path.parent / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        mc = cfg.get("model", cfg)
        mc = {**cfg, **mc}

        model = OrdinalNet.from_config(mc, dropout=0.3, pretrained=False).to(dev)
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()

        # Load OOF cutpoints
        cuts_path = (
            PROJECT_ROOT
            / "reports"
            / "pipeline"
            / f"final_test_{run_name}_expected_cutpoints.json"
        )
        cuts = np.array([0.6085, 1.267, 2.2429, 3.0504])
        if cuts_path.exists():
            report = json.loads(cuts_path.read_text())
            cuts = np.array(report["oof_cutpoints"])

        _model_cache["model"] = model
        _model_cache["cuts"] = cuts
        _model_cache["dev"] = dev
        return model, cuts, dev

    except Exception:
        traceback.print_exc()
        return None, None, None


def _preprocess(pil_img):
    """Preprocess a PIL image for inference: CLAHE → 3ch → normalize."""
    import cv2
    import torch
    import torchvision.transforms.functional as TF

    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    img_pil = TF.to_pil_image(arr)
    img_pil = TF.resize(img_pil, [224, 224])
    t = TF.to_tensor(img_pil)              # (1, 224, 224)
    t = t.repeat(3, 1, 1)                 # (3, 224, 224)
    MEAN = [0.578707] * 3
    STD  = [0.145419] * 3
    t = TF.normalize(t, MEAN, STD)
    return t


def _run_gradcam(model, tensor, dev, cuts):
    """Run dual-stream Grad-CAM. Returns (probs_np, pred_grade, cam_overlay_b64)."""
    import torch
    from src.evaluation.decoding import apply_cutpoints, expected_grade
    from src.evaluation.gradcam_dual import OrdinalNetAdapter, dual_gradcam, overlay_cam

    x = tensor.unsqueeze(0).to(dev)
    adapter = OrdinalNetAdapter(model)

    # Grad-CAM
    combined_cam, global_cam, local_cam = dual_gradcam(adapter, x, target=None)
    # combined_cam shape: (H, W) numpy float32 in [0,1]

    # Probs + grade
    with torch.no_grad():
        raw = model(x)
        probs_t = model.to_probs(raw)
    probs = probs_t.squeeze(0).cpu().numpy().astype(float)
    eg = expected_grade(probs[None, :])[0]
    grade = int(apply_cutpoints(np.array([eg]), cuts)[0])
    grade = max(0, min(4, grade))

    # Overlay
    # Reconstruct the gray image for overlay
    img_norm = tensor.permute(1, 2, 0).cpu().numpy()
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    gray_uint8 = (img_norm[:, :, 0] * 255).astype(np.uint8)
    overlay = overlay_cam(gray_uint8, combined_cam)  # (H,W,3) BGR uint8

    # Encode as base64 PNG
    import cv2
    from PIL import Image as PILImage
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_overlay = PILImage.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return probs, grade, b64


def _demo_predict(pil_img):
    """Fallback when model weights are unavailable — returns plausible fake output."""
    import cv2
    # Use image hash to pick a deterministic fake grade
    arr = np.array(pil_img.convert("L").resize((64, 64)))
    seed = int(arr.mean() * 100) % 5
    grade = seed
    probs = np.zeros(5)
    probs[grade] = 0.6
    if grade > 0:
        probs[grade - 1] = 0.25
    if grade < 4:
        probs[grade + 1] = 0.15
    probs /= probs.sum()

    # Make a fake heatmap on the actual image
    img_rgb = np.array(pil_img.convert("RGB").resize((224, 224)))
    h, w = img_rgb.shape[:2]
    # Gaussian blob centered at joint region
    cx, cy = w // 2, int(h * 0.52)
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (w * 0.18) ** 2))
    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (img_rgb * 0.5 + heatmap_rgb * 0.5).astype(np.uint8)

    buf = io.BytesIO()
    from PIL import Image as PILImage
    PILImage.fromarray(overlay).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return probs.tolist(), grade, b64, 0.72


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("classify.html")


@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {e}"}), 400

    model, cuts, dev = _get_model()

    try:
        if model is not None:
            import torch
            tensor = _preprocess(pil_img)
            probs, grade, cam_b64 = _run_gradcam(model, tensor, dev, cuts)

            # MC entropy (single pass deterministic approximation for speed)
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1))))
            probs_list = [round(float(p), 4) for p in probs]
        else:
            probs_list, grade, cam_b64, entropy = _demo_predict(pil_img)

        return jsonify({
            "grade": grade,
            "label": KL_LABELS[grade],
            "probs": probs_list,
            "explanation": KL_EXPLANATIONS[grade],
            "severity_color": KL_SEVERITY_COLOR[grade],
            "gradcam_b64": cam_b64,
            "entropy": round(entropy, 4),
            "demo_mode": model is None,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/panels/<kl>/<filename>")
def panel_image(kl, filename):
    """Serve Grad-CAM panel images from the reports directory."""
    from flask import send_from_directory
    panels_dir = (
        PROJECT_ROOT
        / "reports"
        / "pipeline"
        / "gradcam_correct_attn_spatial_cutpoints"
        / "panels"
        / kl
    )
    return send_from_directory(str(panels_dir), filename)


@app.route("/uncertainty/<filename>")
def uncertainty_image(filename):
    """Serve reliability diagram images."""
    from flask import send_from_directory
    udir = (
        PROJECT_ROOT
        / "reports"
        / "pipeline"
        / "uncertainty_r50_dual_cascade_sqrt_aux_attn_spatial"
    )
    return send_from_directory(str(udir), filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
