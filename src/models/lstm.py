import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score,
                             precision_score, recall_score)


stationary_features = [
    'Return', 'Return_5d', 'Return_20d', 'Return_Smooth',
    'RSI_14', 'MACD_Hist', 'Drawdown',
    'VIX', 'VIX_Change', 'VIX_Change_5d',
    'GDP_YoY', 'Core_Inflation_YoY', 'M2_YoY', 'Unemployment',
    'Risk_Adj_Return_20d', 'Relative_Volume',
    'MACD_Hist_Accel'
]

cv_path      = "notebooks/data/cv.csv"
lstm_path    = "src/trained/lstm_classification.keras"
results_path = "src/trained/results"
window_size  = 20
n_features   = len(stationary_features)
n_splits     = 5

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


def load_data():
    data = pd.read_csv(cv_path).dropna(subset=["target"] + stationary_features)

    X = data[stationary_features]
    y = data["target"]

    total_rows = len(data)
    val_end    = int(total_rows * 0.85)

    X_cv   = X.iloc[:val_end]
    y_cv   = y.iloc[:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    scaler        = StandardScaler()
    scaler.fit(X_cv)
    X_test_scaled = scaler.transform(X_test)

    return X_cv.values, y_cv.values, X_test_scaled, y_test.values


def create_windows(X: np.ndarray, y: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    n_samples  = X.shape[0] - window_size + 1
    n_features = X.shape[1]

    X_windows = np.zeros((n_samples, window_size, n_features))
    y_windows = np.zeros(n_samples)

    for i in range(n_samples):
        X_windows[i] = X[i:i + window_size]
        y_windows[i] = y[i + window_size - 1]

    return X_windows, y_windows


def build_model(window_size: int, n_features: int,
                lstm_units_1: int = 128,
                lstm_units_2: int = 64,
                dense_units:  int = 32,
                dropout_rate: float = 0.3,
                recurrent_dropout: float = 0.2,
                l2_reg: float = 1e-4,
                learning_rate: float = 1e-3,
) -> tf.keras.Model:

    inputs = Input(shape=(window_size, n_features), name="Features")

    # First LSTM layer — return sequences so second LSTM can attend to all timesteps
    x = LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        recurrent_dropout=recurrent_dropout,
        name="LSTM_1"
    )(inputs)
    x = BatchNormalization(name="BN_1")(x)
    x = Dropout(dropout_rate, name="Dropout_1")(x)

    # Second LSTM layer
    x = LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        recurrent_dropout=recurrent_dropout,
        name="LSTM_2"
    )(x)
    x = BatchNormalization(name="BN_2")(x)
    x = Dropout(dropout_rate, name="Dropout_2")(x)

    # Dense head
    x = Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="Dense_1"
    )(x)
    x = Dropout(dropout_rate / 2, name="Dropout_3")(x)

    outputs = Dense(1, activation="sigmoid", name="Output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_fold_curves(fold_histories: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold, h in enumerate(fold_histories):
        axes[0].plot(h["loss"],         label=f"Fold {fold+1} train")
        axes[0].plot(h["val_loss"],     label=f"Fold {fold+1} val", linestyle="--")
        axes[1].plot(h["accuracy"],     label=f"Fold {fold+1} train")
        axes[1].plot(h["val_accuracy"], label=f"Fold {fold+1} val", linestyle="--")

    axes[0].set_title("Loss per Fold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)

    axes[1].set_title("Accuracy per Fold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "training_curves.png"), dpi=150)
    plt.show()
    plt.close()


def train_crossval(X_cv: np.ndarray, y_cv: np.ndarray) -> None:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = {
        "accuracy":  [],
        "roc_auc":   [],
        "precision": [],
        "recall":    [],
    }

    fold_histories = []
    best_val_loss  = float("inf")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        scaler_fold  = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_cv[train_idx])
        X_val_fold   = scaler_fold.transform(X_cv[val_idx])
        y_train_fold = y_cv[train_idx]
        y_val_fold   = y_cv[val_idx]

        X_train_w, y_train_w = create_windows(X_train_fold, y_train_fold, window_size)
        X_val_w,   y_val_w   = create_windows(X_val_fold,   y_val_fold,   window_size)

        model = build_model(window_size=window_size, n_features=n_features)

        os.makedirs(os.path.dirname(lstm_path), exist_ok=True)

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, mode='min'
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        )

        fold_path    = lstm_path.replace(".keras", f"_fold{fold + 1}.keras")
        checkpointer = ModelCheckpoint(
            filepath=fold_path, verbose=0,
            monitor='val_loss', mode='min',
            save_best_only=True
        )

        history = model.fit(
            X_train_w, y_train_w,
            batch_size=32,
            epochs=100,
            validation_data=(X_val_w, y_val_w),
            callbacks=[early_stopping, reduce_lr, checkpointer],
            verbose=1,
        )

        fold_histories.append(history.history)

        y_pred_prob = model.predict(X_val_w, verbose=0).flatten()
        y_pred      = (y_pred_prob > 0.5).astype(int)

        fold_metrics["accuracy"].append(accuracy_score(y_val_w, y_pred))
        fold_metrics["roc_auc"].append(roc_auc_score(y_val_w, y_pred_prob))
        fold_metrics["precision"].append(precision_score(y_val_w, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val_w, y_pred, zero_division=0))

        print(f"  Accuracy:  {fold_metrics['accuracy'][-1]:.4f}")
        print(f"  ROC-AUC:   {fold_metrics['roc_auc'][-1]:.4f}")
        print(f"  Precision: {fold_metrics['precision'][-1]:.4f}")
        print(f"  Recall:    {fold_metrics['recall'][-1]:.4f}")

        val_loss = model.evaluate(X_val_w, y_val_w, verbose=0)[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(lstm_path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

    print("\n=== Cross-Validation Summary ===")
    for metric, values in fold_metrics.items():
        print(f"{metric:>10}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    os.makedirs(results_path, exist_ok=True)

    cv_df = pd.DataFrame(fold_metrics)
    cv_df.index = [f"Fold {i+1}" for i in range(n_splits)]
    cv_df.loc["Mean"] = cv_df.mean()
    cv_df.loc["Std"]  = cv_df.std()
    cv_df.to_csv(os.path.join(results_path, "cv_summary.csv"))
    print(f"CV summary saved to {results_path}/cv_summary.csv")

    save_fold_curves(fold_histories)


def output_results(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    os.makedirs(results_path, exist_ok=True)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bear", "Bull"],
                yticklabels=["Bear", "Bull"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "confusion_matrix.png"), dpi=150)
    plt.show()
    plt.close()

    report = classification_report(y_test, y_pred, target_names=["Bear", "Bull"])
    print("\n=== Classification Report (Test Set) ===")
    print(report)

    with open(os.path.join(results_path, "classification_report.txt"), "w") as f:
        f.write(report)

    test_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
    }

    metrics_df = pd.DataFrame(test_metrics, index=["Test Set"])
    metrics_df.to_csv(os.path.join(results_path, "test_metrics.csv"))
    print(f"\nResults saved to {results_path}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train with cross-validation")
    args = parser.parse_args()

    X_cv, y_cv, X_test, y_test = load_data()

    X_test_w, y_test_w = create_windows(X_test, y_test, window_size)

    if args.train:
        train_crossval(X_cv, y_cv)
    else:
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"No saved model found at {lstm_path}. Run with --train first.")

    model = tf.keras.models.load_model(lstm_path)
    output_results(model, X_test_w, y_test_w)


if __name__ == "__main__":
    main()