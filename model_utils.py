"""
Inference helpers shared by the Flask app and the CLI.

Provides:
  - load_tb_model()          : cached model loader
  - predict_image(path)      : returns label, confidence, probabilities
  - generate_gradcam(...)    : saves a Grad-CAM heat-map overlay for explainability
"""
import os

import numpy as np

import config

_MODEL = None


def model_exists():
    return os.path.exists(config.MODEL_PATH)


def load_tb_model():
    """Load (and cache) the trained Keras model. Returns None if not trained."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not model_exists():
        return None
    import tensorflow as tf  # imported lazily so the app starts fast
    _MODEL = tf.keras.models.load_model(config.MODEL_PATH)
    return _MODEL


def _input_size():
    """Image side length the loaded model expects (falls back to config)."""
    model = load_tb_model()
    try:
        h = model.input_shape[1]
        if h:
            return int(h)
    except Exception:
        pass
    return config.IMG_SIZE


def _load_array(image_path):
    """
    Load an image as (raw_uint8_array, batch), sized to the model's input.

    Preprocessing (x/127.5 - 1) is baked into the model as a Rescaling layer,
    so we feed raw [0, 255] pixels here. `arr` (raw) is kept for Grad-CAM
    overlay rendering.
    """
    import tensorflow as tf

    size = _input_size()
    img = tf.keras.utils.load_img(image_path, target_size=(size, size))
    arr = tf.keras.utils.img_to_array(img)
    batch = np.expand_dims(arr, axis=0)
    return arr, batch


def predict_image(image_path):
    """
    Run inference on one image.

    Returns a dict:
        {
          "label": "Tuberculosis" | "Normal",
          "confidence": 0-100 float (confidence in the predicted label),
          "tb_probability": 0-100 float,
          "normal_probability": 0-100 float,
        }
    Raises RuntimeError if no model is available.
    """
    model = load_tb_model()
    if model is None:
        raise RuntimeError(
            "No trained model found. Run `python train.py` to create "
            "models/tb_model.h5 first."
        )

    _, batch = _load_array(image_path)
    tb_prob = float(model.predict(batch, verbose=0)[0][0])  # sigmoid -> P(TB)
    normal_prob = 1.0 - tb_prob

    thr = config.DECISION_THRESHOLD
    is_tb = tb_prob >= thr
    label = config.CLASS_NAMES[1] if is_tb else config.CLASS_NAMES[0]

    # Calibrated confidence: 50% at the decision boundary, scaling to 100% at
    # the extremes. Keeps the headline confidence intuitive despite thr != 0.5.
    if is_tb:
        confidence = 0.5 + 0.5 * (tb_prob - thr) / max(1e-6, 1.0 - thr)
    else:
        confidence = 0.5 + 0.5 * (thr - tb_prob) / max(1e-6, thr)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "tb_probability": round(tb_prob * 100, 2),
        "normal_probability": round(normal_prob * 100, 2),
        "threshold": round(thr * 100, 2),
    }


def generate_gradcam(image_path, output_path, last_conv_layer_name="Conv_1"):
    """
    Produce a Grad-CAM heat-map overlay highlighting the regions that drove
    the prediction. Saves the overlay to output_path. Returns True on success.
    Failure is non-fatal (returns False) so the app keeps working.
    """
    try:
        import cv2
        import tensorflow as tf

        model = load_tb_model()
        if model is None:
            return False

        arr, batch = _load_array(image_path)

        # The MobileNetV2 base is nested as a sub-model. Find the sub-model that
        # actually contains the target conv layer (skip the augmentation block).
        base = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    layer.get_layer(last_conv_layer_name)
                    base = layer
                    break
                except (ValueError, KeyError):
                    continue
        if base is None:
            return False

        # Layers of the outer model that come AFTER the base form the classifier
        # head (pooling, dropout, dense...). We replay them to get the score.
        base_idx = model.layers.index(base)
        head_layers = model.layers[base_idx + 1:]

        # Grad model maps the base's input to (target feature map, base output).
        grad_model = tf.keras.models.Model(
            base.input,
            [base.get_layer(last_conv_layer_name).output, base.output],
        )

        with tf.GradientTape() as tape:
            # Same preprocessing the model's Rescaling layer applies internally.
            rescaled = tf.cast(batch, tf.float32) / 127.5 - 1.0
            conv_out, base_out = grad_model(rescaled)
            x = base_out
            for layer in head_layers:
                x = layer(x, training=False)
            loss = x[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        size = _input_size()
        heatmap = cv2.resize(heatmap, (size, size))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(output_path, overlay)
        return True
    except Exception as e:
        print(f"(Grad-CAM skipped: {e})")
        return False
