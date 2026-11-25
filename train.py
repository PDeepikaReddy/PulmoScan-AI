"""
Train a Tuberculosis vs. Normal chest X-ray classifier using transfer
learning (MobileNetV2) and export it to models/tb_model.h5.

Usage:
    python train.py

The script:
  1. Loads images from data/TB_Chest_Radiography_Database/{Normal,Tuberculosis}
  2. Builds an 80/20 train/validation split.
  3. Trains a MobileNetV2-based classifier head, then fine-tunes the top layers.
  4. Handles class imbalance (the dataset has ~5x more Normal than TB).
  5. Saves the best model to models/tb_model.h5 and a training plot.
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)


def build_datasets():
    if not os.path.isdir(config.TRAIN_DIR):
        raise FileNotFoundError(
            f"Dataset not found at {config.TRAIN_DIR}.\n"
            "Run `python download_data.py` first."
        )

    common = dict(
        directory=config.TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,   # forces Normal=0, Tuberculosis=1
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.2, subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_split=0.2, subset="validation", **common
    )

    # Compute class weights from the training split to fight imbalance.
    counts = np.zeros(2)
    for _, y in train_ds.unbatch().as_numpy_iterator():
        counts[int(y)] += 1
    total = counts.sum()
    class_weight = {i: total / (2.0 * c) if c else 1.0 for i, c in enumerate(counts)}
    print(f"Class counts (train): Normal={int(counts[0])}, TB={int(counts[1])}")
    print(f"Class weights: {class_weight}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_weight


def build_model():
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base = MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = data_augmentation(inputs)
    # MobileNetV2 preprocessing == x/127.5 - 1, implemented as a Rescaling layer
    # so the model serializes/reloads cleanly from the legacy .h5 format.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model, base


def plot_history(histories):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        acc, val_acc, loss, val_loss = [], [], [], []
        for h in histories:
            acc += h.history.get("accuracy", [])
            val_acc += h.history.get("val_accuracy", [])
            loss += h.history.get("loss", [])
            val_loss += h.history.get("val_loss", [])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(acc, label="train"); ax1.plot(val_acc, label="val")
        ax1.set_title("Accuracy"); ax1.legend()
        ax2.plot(loss, label="train"); ax2.plot(val_loss, label="val")
        ax2.set_title("Loss"); ax2.legend()
        out = os.path.join(config.MODELS_DIR, "training_history.png")
        fig.tight_layout(); fig.savefig(out)
        print(f"Saved training plot -> {out}")
    except Exception as e:  # plotting is optional
        print(f"(Skipping plot: {e})")


def main():
    train_ds, val_ds, class_weight = build_datasets()
    model, base = build_model()
    model.summary()

    ckpt = ModelCheckpoint(
        config.MODEL_PATH, monitor="val_auc", mode="max",
        save_best_only=True, verbose=1,
    )
    early = EarlyStopping(
        monitor="val_auc", mode="max", patience=4,
        restore_best_weights=True, verbose=1,
    )
    reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    callbacks = [ckpt, early, reduce]

    print("\n=== Stage 1: training classifier head ===")
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=config.EPOCHS_HEAD,
        class_weight=class_weight, callbacks=callbacks,
    )

    print("\n=== Stage 2: fine-tuning top layers ===")
    base.trainable = True
    for layer in base.layers[:-30]:   # unfreeze only the last 30 layers
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(config.FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINETUNE,
        class_weight=class_weight, callbacks=callbacks,
    )

    # Ensure the final best model is on disk (checkpoint already saves best).
    if not os.path.exists(config.MODEL_PATH):
        model.save(config.MODEL_PATH)

    print("\n=== Final validation metrics ===")
    results = model.evaluate(val_ds, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:10s}: {val:.4f}")

    plot_history([h1, h2])
    print(f"\nModel saved -> {config.MODEL_PATH}")
    print("Next step:  python app.py")


if __name__ == "__main__":
    main()
