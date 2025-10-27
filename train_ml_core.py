# train_ml_core.py  — Tier 4.3+ stable version (no callbacks / no early stopping)
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    RocCurveDisplay,
)

print(">>> USING train_ml_core.py FROM:", __file__)

# --- Modellwahl (XGB, sonst GBM) ---
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from features_store import add_indicators, prepare_xy


def plot_feature_importance(model, features, top_n=20, save_to=None):
    """Wichtigste Features anzeigen/optional speichern."""
    if hasattr(model, "feature_importances_"):
        imp = (
            pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False)
            .head(top_n)
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(x=imp.values, y=imp.index)
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            plt.savefig(save_to, dpi=140, bbox_inches="tight")
        plt.show()


def train_and_evaluate(df: pd.DataFrame, save_reports=True):
    """Trainiert & evaluiert das Modell auf Candle-Daten – sicher ohne callbacks/early_stopping."""
    print("🧠 Starte Tier 4.3+ ML-Training ...")

    # Grundsanity: sortieren, Duplikate raus, numerische Typen
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "high", "low", "volume"])

    # Feature-Engineering + Labels (Wrapper sorgt für y_bin/y_reg)
    feats = add_indicators(df)
    X, y = prepare_xy(feats, "y_bin")

    # Zeitlicher Split (keine Shuffles)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Optional robust casten (XGB mag float-Input)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Modell
    if XGB_AVAILABLE:
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        spw = (neg / max(pos, 1.0)) if pos and neg else 1.0

        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=spw,
            tree_method="hist",
        )
        print(f"✅ XGBoost aktiviert (scale_pos_weight≈{spw:.2f})")

        # --- Wichtig: KEINE callbacks / KEIN early_stopping_rounds ---
        # Erst versuchen wir mit eval_set, sonst ohne (max. Kompatibilität).
        try:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        except TypeError:
            # Falls deine XGB-Version eval_set nicht akzeptiert:
            model.fit(X_train, y_train)
    else:
        print("⚠️ XGBoost nicht installiert – fallback zu GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

    # Vorhersagen
    preds = model.predict(X_test)
    probs = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    )

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan

    print("\n📊 PERFORMANCE REPORT")
    print(classification_report(y_test, preds, digits=4))
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_reports:
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/confusion_matrix.png", dpi=140, bbox_inches="tight")
    plt.show()

    # ROC-Kurve (falls probabilistisch)
    if hasattr(model, "predict_proba"):
        RocCurveDisplay.from_predictions(y_test, probs)
        plt.title("ROC Curve")
        plt.tight_layout()
        if save_reports:
            plt.savefig("reports/roc_curve.png", dpi=140, bbox_inches="tight")
        plt.show()

    plot_feature_importance(
        model,
        list(X.columns),
        top_n=20,
        save_to="reports/feature_importance.png" if save_reports else None,
    )

    print("\n💾 Model training complete – Tier 4.3+ ready.")
    return model


if __name__ == "__main__":
    # Parquet laden
    df = pd.read_parquet("data/bybit_BTCUSDT_15.parquet")

    # robustes Setzen des Zeitindex (egal ob Spalte oder Index)
    if "open_time" in df.columns:
        df.index = pd.to_datetime(df["open_time"], utc=True)
        df = df.drop(columns=["open_time"])
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    # sicherstellen, dass Index monoton ist
    df = df.sort_index()

    # Training
    _ = train_and_evaluate(df, save_reports=True)
