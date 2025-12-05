"""
Preprocessing pipeline for Supervised Baseline (HistGradientBoosting)
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import load
import json
from pathlib import Path


def build_features_patient_centric(
    df: pd.DataFrame, window: int = 7
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Feature engineering centré patient avec rolling windows.
    Identique à celui utilisé pour l'entraînement.
    """
    df = df.copy()
    
    # Identifier la colonne timestamp
    timestamp_col = None
    for col in ["timestamp", "date"]:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        raise ValueError("Colonne timestamp/date introuvable")
    
    # Convertir date en datetime si nécessaire
    if timestamp_col == "date" and pd.api.types.is_string_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Trier par patient_id et timestamp
    df = df.sort_values(by=["patient_id", timestamp_col]).reset_index(drop=True)
    
    # Variables continues pour rolling statistics
    continuous_vars = [
        "heart_rate", "hr_variability", "steps", "mood_score",
        "sleep_duration_hours", "sleep_efficiency", "num_awakenings"
    ]
    continuous_vars = [v for v in continuous_vars if v in df.columns]
    
    # Calculer rolling mean et std pour chaque variable continue
    for col in continuous_vars:
        roll_mean = df.groupby("patient_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=3).mean()
        )
        roll_std = df.groupby("patient_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=3).std()
        )
        df[f"{col}_delta"] = df[col] - roll_mean
        z_score = df[f"{col}_delta"] / (roll_std + 1e-6)
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        df[f"{col}_z"] = z_score
    
    # Dérivées utiles
    if "steps" in df.columns:
        df["steps_log1p"] = np.log1p(df["steps"])
    
    if "num_awakenings" in df.columns and "sleep_duration_hours" in df.columns:
        df["awakenings_per_hour"] = df["num_awakenings"] / df["sleep_duration_hours"].clip(lower=0.5)
    
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    if "heart_rate" in df.columns and "hr_variability" in df.columns:
        hrv_clipped = df["hr_variability"].clip(lower=1e-3)
        df["hr_hrv_ratio"] = df["heart_rate"] / hrv_clipped
    
    if "sleep_duration_hours" in df.columns:
        df["sleep_debt"] = np.maximum(0, 7.5 - df["sleep_duration_hours"])
    
    # Construire les listes de features
    num_cols = []
    cat_cols = []
    
    for col in continuous_vars:
        if f"{col}_delta" in df.columns:
            num_cols.append(f"{col}_delta")
        if f"{col}_z" in df.columns:
            num_cols.append(f"{col}_z")
    
    if "steps_log1p" in df.columns:
        num_cols.append("steps_log1p")
    if "awakenings_per_hour" in df.columns:
        num_cols.append("awakenings_per_hour")
    if "hr_hrv_ratio" in df.columns:
        num_cols.append("hr_hrv_ratio")
    if "sleep_debt" in df.columns:
        num_cols.append("sleep_debt")
    if "age" in df.columns:
        num_cols.append("age")
    if "dow_sin" in df.columns:
        num_cols.append("dow_sin")
    if "dow_cos" in df.columns:
        num_cols.append("dow_cos")
    
    for col in ["weekend", "medication_taken", "is_female"]:
        if col in df.columns:
            cat_cols.append(col)
    
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    keep_cols = ["patient_id", timestamp_col] + num_cols + cat_cols
    keep_cols = [col for col in keep_cols if col in df.columns]
    df_enrichi = df[keep_cols].copy()
    
    feature_cols = num_cols + cat_cols
    df_enrichi = df_enrichi.dropna(subset=feature_cols)
    
    return df_enrichi, num_cols, cat_cols


def get_preprocessing_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    """
    Retourne le pipeline de preprocessing utilisé pour l'entraînement.
    
    Parameters:
    -----------
    num_cols : list
        Liste des colonnes numériques
    cat_cols : list
        Liste des colonnes catégorielles
    
    Returns:
    --------
    Pipeline avec ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    
    return Pipeline([("preprocessor", preprocessor)])


def load_supervised_model(model_path: str = "artifacts/supervised_pipeline.joblib"):
    """
    Charge le modèle supervisé entraîné.
    
    Returns:
    --------
    Pipeline complet (preprocessor + classifier)
    """
    return load(model_path)


def predict_with_supervised_model(
    pipeline,
    X: pd.DataFrame,
    threshold: float = None,
    threshold_path: str = "artifacts/supervised_threshold.json"
):
    """
    Prédit avec le modèle supervisé.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Pipeline entraîné
    X : pd.DataFrame
        Features (doit contenir num_cols + cat_cols)
    threshold : float, optional
        Seuil (si None, charge depuis threshold_path)
    threshold_path : str
        Chemin vers le fichier de seuil
    
    Returns:
    --------
    dict avec 'scores', 'predictions', 'threshold_used'
    """
    if threshold is None:
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
        threshold = threshold_data['tau']
    
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    X_prep = preprocessor.transform(X)
    scores = classifier.predict_proba(X_prep)[:, 1]
    predictions = (scores >= threshold).astype(int)
    
    return {
        'scores': scores,
        'predictions': predictions,
        'threshold_used': threshold
    }


# Exemple d'utilisation
if __name__ == "__main__":
    print("="*70)
    print("PREPROCESSING PIPELINE FOR SUPERVISED BASELINE")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv("data/clinical_alerts.csv")
    
    # Feature engineering
    print("\n[1] Feature engineering...")
    df_feat, num_cols, cat_cols = build_features_patient_centric(df, window=7)
    print(f"   Numerical features: {len(num_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")
    
    # Créer le preprocessing pipeline
    print("\n[2] Creating preprocessing pipeline...")
    preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)
    print("   [OK] Preprocessing pipeline created")
    
    # Charger le modèle
    print("\n[3] Loading trained model...")
    pipeline = load_supervised_model()
    print("   [OK] Model loaded")
    
    # Exemple de prédiction
    print("\n[4] Example prediction...")
    X_sample = df_feat[num_cols + cat_cols].iloc[:10]
    results = predict_with_supervised_model(pipeline, X_sample)
    print(f"   Scores: {results['scores']}")
    print(f"   Predictions: {results['predictions']}")
    print(f"   Threshold used: {results['threshold_used']:.4f}")
    
    print("\n" + "="*70)
    print("[OK] PREPROCESSING PIPELINE READY")
    print("="*70)

