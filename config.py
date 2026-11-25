"""
Central configuration for the Tuberculosis Detection project.
Edit values here instead of hard-coding them across scripts.
"""
import os

# ---- Paths ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")            # raw dataset lives here
TRAIN_DIR = os.path.join(DATA_DIR, "TB_Chest_Radiography_Database")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "tb_model.h5")  # trained model output
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# ---- Kaggle dataset ------------------------------------------------------
# https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
KAGGLE_DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"

# ---- Image / model hyper-parameters --------------------------------------
IMG_SIZE = 160          # MobileNetV2 supported size; 160 trains fast on CPU
BATCH_SIZE = 32
EPOCHS_HEAD = 8         # epochs training the classifier head (early-stopped)
EPOCHS_FINETUNE = 4     # epochs fine-tuning top conv layers (early-stopped)
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
SEED = 42

# Class order is fixed so the app and the model always agree.
# Index 0 -> Normal, Index 1 -> Tuberculosis
CLASS_NAMES = ["Normal", "Tuberculosis"]

# Decision threshold on P(Tuberculosis). The sigmoid is well-calibrated but the
# classes are imbalanced, so the optimal operating point is below 0.5. Tuned on
# a held-out sample (sensitivity 0.95 / specificity 0.98 @ 0.20). Lower it to
# catch more TB (higher sensitivity), raise it to reduce false alarms.
DECISION_THRESHOLD = 0.20

# Ensure runtime folders exist
for _d in (MODELS_DIR, UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)
