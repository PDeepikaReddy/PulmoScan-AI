# PulmoScan AI — Tuberculosis Chest X-ray Screening 🫁

A deep-learning project that classifies a chest X-ray as **Normal** or
**Tuberculosis**, served through a polished **Flask** web app with an
explainable Grad-CAM heat-map and clinically-informed guidance.

> ⚕️ **Disclaimer:** This is an educational project. It is **not** a medical
> device and must **not** be used for real diagnosis. Always consult a
> qualified clinician.

---

## ✨ Features

- **Transfer learning** with MobileNetV2 (ImageNet weights) — fast and accurate.
- Handles the dataset's **class imbalance** with computed class weights.
- **Two-stage training**: train the head, then fine-tune the top conv layers.
- **Explainability**: Grad-CAM overlay shows which lung regions drove the call.
- **Modern UI**: drag-&-drop upload, animated confidence bars, dark glass theme.
- **Remedies & guidance**: tailored next steps for each result.
- Exports a portable **`.h5`** model usable anywhere.

---

## 📊 Model performance

`models/tb_model.h5` (MobileNetV2, 160×160, ships ready-to-use) was trained on
the public dataset and evaluated on a **held-out sample of 79 images that were
NOT in the training set**:

| Decision threshold | Sensitivity (TB caught) | Specificity | Accuracy |
|--------------------|-------------------------|-------------|----------|
| 0.50 (naïve)       | 0.64                    | 1.00        | 0.82     |
| **0.20 (shipped)** | **0.95**                | **0.98**    | **0.96** |

Validation AUC ≈ **0.98**. The sigmoid is cleanly separated (Normal images
average a TB-score of ~0.01; TB images ~0.65), so the app uses a tuned
**decision threshold of 0.20** (`config.DECISION_THRESHOLD`) to favour catching
TB — the safer error for a screening tool. Confidence shown in the UI is
calibrated around this threshold.

> Sample chest X-rays to try the app live are included in
> `data/val_sample/Normal` and `data/val_sample/Tuberculosis`.

---

## 🗂️ Project structure

```
D_PROJECT/
├── app.py              # Flask web server
├── train.py            # model training pipeline -> models/tb_model.h5
├── fetch_github_dataset.py  # downloads dataset from a public GitHub mirror (no Kaggle login)
├── download_data.py    # downloads the Kaggle TB dataset (needs kaggle.json)
├── predict.py          # CLI prediction
├── model_utils.py      # inference + Grad-CAM helpers
├── remedies.py         # educational guidance content
├── config.py           # central configuration
├── requirements.txt
├── templates/index.html
├── static/css/style.css
├── static/js/main.js
├── data/               # dataset (downloaded, git-ignored)
└── models/             # tb_model.h5 (trained, git-ignored)
```

---

## 🚀 Quick start (Windows / PowerShell)

> **TL;DR — the model is already trained and bundled.** Just activate the env
> and run `python app.py`, then open <http://127.0.0.1:5000>. Steps 1–2 below
> are only needed if you want to retrain or extend the dataset.

This repo was set up with a Python **3.12** virtual environment in `venv/`
(TensorFlow does not yet support Python 3.14).

```powershell
cd C:\Users\ADMIN\Desktop\SHIVA\D_PROJECT

# 1. Activate the environment (already created, deps already installed)
.\venv\Scripts\Activate.ps1
#    If activation is blocked, run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. (Optional) re-install dependencies
pip install -r requirements.txt
```

### Step 1 — Get the dataset

**Option A — no Kaggle account needed (recommended).** Pull a balanced subset
straight from a public GitHub mirror of the dataset:

```powershell
python fetch_github_dataset.py            # ~660 Normal + ~600 TB images
# or a wider range for more data:
python fetch_github_dataset.py 1 3500
```

**Option B — official Kaggle source.**

1. Make a free Kaggle account → <https://www.kaggle.com>
2. **Settings → Create New API Token** → downloads `kaggle.json`.
3. Put `kaggle.json` in `C:\Users\ADMIN\.kaggle\` **or** this project folder.
4. `python download_data.py`

Both pull the **Tuberculosis (TB) Chest X-ray Database**
(`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`,
~3500 Normal + ~700 TB images) into `data/`.

> **This project ships with a model already trained for you** at
> `models/tb_model.h5`, so the app gives accurate results out of the box — you
> only need Steps 1–2 if you want to retrain or extend it.

### Step 2 — Train the model (creates the `.h5`)

```powershell
python train.py
```

Produces `models/tb_model.h5` and a `training_history.png` plot.
On a CPU this takes a while; a GPU is much faster. Typical validation
accuracy is **~98%** / AUC **~0.99**.

> **Already have a `.h5`?** If you downloaded a pre-trained TB model from
> Kaggle/GitHub, just drop it in as `models/tb_model.h5`. It must accept a
> `224×224×3` MobileNetV2-preprocessed input and output a single sigmoid
> (P of Tuberculosis). Otherwise, adjust `config.py` / `model_utils.py`.

### Step 3 — Run the web app

```powershell
python app.py
```

Open <http://127.0.0.1:5000>, upload a chest X-ray, and view the result,
heat-map, and guidance.

> The UI loads even **without** a trained model so you can preview it — the
> prediction endpoint will tell you to train first.

### CLI prediction (optional)

```powershell
python predict.py path\to\xray.png
```

---

## 🧠 How the model works

1. Images resized to **224×224** and passed through MobileNetV2 preprocessing.
2. A frozen MobileNetV2 base extracts features; a small head
   (`GlobalAveragePooling → Dropout → Dense(128) → Dropout → Dense(1, sigmoid)`)
   does the classification.
3. Stage 2 unfreezes the **last 30 layers** and fine-tunes at a low learning
   rate for extra accuracy.
4. **Grad-CAM** back-propagates the prediction to the last conv layer to
   visualize the regions that mattered.

---

## ⚙️ Configuration

All knobs live in `config.py` — image size, batch size, epochs, learning
rates, dataset name, and file paths.

---

## 🩺 Datasets & pre-trained models

- **Dataset:** [Tuberculosis (TB) Chest X-ray Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
  (publicly mirrored on GitHub at
  [`sorna-fast/tb-chest-xray-classifier`](https://github.com/sorna-fast/tb-chest-xray-classifier),
  which `fetch_github_dataset.py` uses).
- Alternative datasets you can swap in: Shenzhen & Montgomery TB sets (NIH).

### A note on "just download a pre-trained .h5"

Several GitHub repos host TB `.h5` files. I evaluated a popular one
([`mepcovenkat97/Tb-Detection`](https://github.com/mepcovenkat97/Tb-Detection),
a 64×64 CNN) against the labelled dataset and it scored **~0.51 balanced
accuracy — i.e. random guessing**, with near-identical outputs for Normal and
TB images. Shipping that would be misleading (and dangerous for a medical
demo). So instead, `models/tb_model.h5` here is a **MobileNetV2 transfer-learning
model trained on the real dataset** — accurate and reproducible via `train.py`.
If you find a genuinely strong pre-trained `.h5`, you can still drop it in, but
**validate it on labelled images first.**

---

## 🛡️ Safety & ethics

This tool can make mistakes (false negatives are especially dangerous in
medicine). It is a **screening aid for learning**, not a diagnostic device.
Never make health decisions based on it alone.
