"""
Training script for Supervised Baseline (HistGradientBoosting) only
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from itertools import product


def build_features_patient_centric(
    df: pd.DataFrame, window: int = 7
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Feature engineering centré patient avec rolling windows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec colonnes: patient_id, timestamp|date, et variables continues
    window : int
        Taille de la fenêtre glissante en jours
    
    Returns:
    --------
    df_enrichi : pd.DataFrame
        DataFrame enrichi avec nouvelles features
    num_cols : list
        Liste des colonnes numériques pour le modèle
    cat_cols : list
        Liste des colonnes catégorielles pour le modèle
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
        # Rolling mean
        roll_mean = df.groupby("patient_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=3).mean()
        )
        
        # Rolling std
        roll_std = df.groupby("patient_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=3).std()
        )
        
        # Delta = x - roll_mean
        df[f"{col}_delta"] = df[col] - roll_mean
        
        # Z-score = (x - roll_mean) / (roll_std + 1e-6)
        z_score = df[f"{col}_delta"] / (roll_std + 1e-6)
        # Remplacer ±inf par NaN
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        df[f"{col}_z"] = z_score
    
    # Dérivées utiles
    
    # steps_log1p = log1p(steps)
    if "steps" in df.columns:
        df["steps_log1p"] = np.log1p(df["steps"])
    
    # awakenings_per_hour = num_awakenings / clip(sleep_duration_hours, 0.5, None)
    if "num_awakenings" in df.columns and "sleep_duration_hours" in df.columns:
        df["awakenings_per_hour"] = df["num_awakenings"] / df["sleep_duration_hours"].clip(lower=0.5)
    
    # Encodage cyclique de day_of_week
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # hr_hrv_ratio = heart_rate / clip(hr_variability, 1e-3, None)
    if "heart_rate" in df.columns and "hr_variability" in df.columns:
        hrv_clipped = df["hr_variability"].clip(lower=1e-3)
        df["hr_hrv_ratio"] = df["heart_rate"] / hrv_clipped
    
    # sleep_debt = max(0, 7.5 - sleep_duration_hours)
    if "sleep_duration_hours" in df.columns:
        df["sleep_debt"] = np.maximum(0, 7.5 - df["sleep_duration_hours"])
    
    # Construire les listes de features
    num_cols = []
    cat_cols = []
    
    # Features numériques: *_delta, *_z, steps_log1p, awakenings_per_hour, hr_hrv_ratio, sleep_debt, age, dow_sin, dow_cos
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
    
    # Features catégorielles: weekend, medication_taken, is_female
    for col in ["weekend", "medication_taken", "is_female"]:
        if col in df.columns:
            cat_cols.append(col)
    
    # Garder seulement les features qui existent
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # Sélectionner les colonnes nécessaires
    keep_cols = ["patient_id", timestamp_col] + num_cols + cat_cols
    keep_cols = [col for col in keep_cols if col in df.columns]
    df_enrichi = df[keep_cols].copy()
    
    # Drop NaN dans les features (pas d'imputation)
    feature_cols = num_cols + cat_cols
    df_enrichi = df_enrichi.dropna(subset=feature_cols)
    
    return df_enrichi, num_cols, cat_cols


def sweep_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "accuracy",
    recall_min: float | None = 0.25,
    target_accuracy: float | None = None,
    target_tolerance: float = 0.005
) -> dict:
    """
    Sweep thresholds to maximize specified metric with optional recall constraint.
    When target_accuracy is provided, pick the threshold whose accuracy is closest
    to the target (preferring values below the target within tolerance).
    """
    quantiles = np.linspace(0.01, 0.995, 200)
    thresholds = np.quantile(scores, quantiles)
    
    best_result = None
    best_metric_value = -1
    candidate_results = []
    
    for tau in thresholds:
        y_pred = (scores >= tau).astype(int)
        
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        if recall_min is not None and rec < recall_min:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        candidate = {
            'tau': float(tau),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        candidate_results.append(candidate)
        
        if metric not in candidate:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_value = candidate[metric]
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_result = candidate
    
    if best_result is None and recall_min is not None:
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, scores)
        recall_mask = recalls >= recall_min
        if recall_mask.any():
            best_idx = np.where(recall_mask)[0][0]
            tau = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else thresholds_pr[-1]
        else:
            best_idx = np.argmax(recalls)
            tau = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else thresholds_pr[-1]
        
        y_pred = (scores >= tau).astype(int)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fallback_candidate = {
            'tau': float(tau),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
        candidate_results.append(fallback_candidate)
        best_result = fallback_candidate
    
    if target_accuracy is not None and candidate_results:
        allowable = [
            c for c in candidate_results
            if c['accuracy'] <= target_accuracy + target_tolerance
        ]
        if not allowable:
            allowable = candidate_results
        selected = min(allowable, key=lambda c: abs(c['accuracy'] - target_accuracy))
        return selected
    
    return best_result


def train_supervised_baseline(
    df, num_cols, cat_cols, idx_train, idx_val,
    metric="accuracy", recall_min=0.25, random_state=42,
    target_accuracy: float | None = None, target_tolerance: float = 0.005,
    class_weight="balanced",
):
    """
    Train HistGradientBoostingClassifier as supervised baseline.
    
    Returns:
    --------
    (best_pipe, best_tau, best_val_report_dict, best_params)
    """
    X_train = df.loc[idx_train, num_cols + cat_cols]
    y_train = df.loc[idx_train, "alert_flag"].values
    X_val = df.loc[idx_val, num_cols + cat_cols]
    y_val = df.loc[idx_val, "alert_flag"].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    
    # Grid search
    grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [None, 6, 10],
        "max_iter": [200, 400]
    }
    
    best_metric_value = -1
    best_pipe = None
    best_tau = None
    best_val_report = None
    best_params = None
    
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))
    
    print(f"  Grid search: {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        classifier = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_iter=params["max_iter"],
            random_state=random_state,
            class_weight=class_weight,
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)
    
    X_val_prep = preprocessor.transform(X_val)
    scores_val = classifier.predict_proba(X_val_prep)[:, 1]
    threshold_result = sweep_threshold(
        y_val,
        scores_val,
        metric=metric,
        recall_min=recall_min,
            target_accuracy=target_accuracy,
            target_tolerance=target_tolerance,
    )

    metric_value = threshold_result[metric]

    if metric_value > best_metric_value:
        best_metric_value = metric_value
        best_tau = threshold_result['tau']
        best_val_report = threshold_result.copy()
        best_params = params.copy()
        best_pipe = pipeline

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(combinations)}] Best {metric}: {best_metric_value:.4f}")
    
    return best_pipe, best_tau, best_val_report, best_params


def evaluate_on_test(
    pipe, tau, df, num_cols, cat_cols, idx_test, out_dir: Path, prefix="supervised_test"
) -> dict:
    """
    Evaluate model on test set and save plots.
    """
    X_test = df.loc[idx_test, num_cols + cat_cols]
    y_test = df.loc[idx_test, "alert_flag"].values
    
    preprocessor = pipe.named_steps["preprocessor"]
    X_test_prep = preprocessor.transform(X_test)
    
    scores_test = pipe.named_steps["classifier"].predict_proba(X_test_prep)[:, 1]
    y_pred = (scores_test >= tau).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_test, scores_test)
    except ValueError:
        auc_roc = np.nan
    
    try:
        auc_pr = average_precision_score(y_test, scores_test)
    except ValueError:
        auc_pr = np.nan
    
    prevalence = y_test.mean()
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=['Normal', 'Anomalie'])
    
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_pr': float(auc_pr) if not np.isnan(auc_pr) else None,
        'auc_roc': float(auc_roc) if not np.isnan(auc_roc) else None,
        'prevalence': float(prevalence),
        'threshold_used': float(tau),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_str
    }
    
    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, scores_test)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    """Main function to train and evaluate supervised model."""
    
    parser = argparse.ArgumentParser(
        description="Train HistGradientBoostingClassifier for clinical alert detection."
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="Optional target accuracy for threshold selection (0-1 or 0-100).",
    )
    parser.add_argument(
        "--target-accuracy-tolerance",
        type=float,
        default=0.005,
        help="Tolerance applied when enforcing target accuracy (absolute, default=0.005).",
    )
    parser.add_argument(
        "--recall-min",
        type=float,
        default=0.25,
        help="Minimum recall constraint for threshold sweep (default=0.25).",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=None,
        help="Optional multiplier applied to class 1; overrides automatic balancing when provided.",
    )
    args = parser.parse_args()
    
    target_accuracy = args.target_accuracy
    if target_accuracy is not None:
        if target_accuracy > 1:
            target_accuracy /= 100.0
        if not 0 < target_accuracy < 1:
            raise ValueError("target_accuracy must be between 0 and 1 (exclusive).")
    target_tolerance = args.target_accuracy_tolerance
    if target_tolerance <= 0:
        raise ValueError("target_accuracy_tolerance must be positive.")
    recall_min = args.recall_min
    if recall_min is not None and recall_min < 0:
        raise ValueError("recall_min must be >= 0.")
    if args.positive_class_weight is not None and args.positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be > 0.")
    
    if args.positive_class_weight is not None:
        class_weight = {0: 1.0, 1: float(args.positive_class_weight)}
    else:
        class_weight = "balanced"
    
    print("="*70)
    print("TRAINING SUPERVISED BASELINE (HistGradientBoosting)")
    print("="*70)
    if target_accuracy is not None:
        print(
            f"Target accuracy requested: {target_accuracy*100:.2f}% "
            f"(± {target_tolerance*100:.2f}pp)"
        )
    if recall_min is not None:
        print(f"Minimum recall constraint: {recall_min*100:.2f}%")
    if isinstance(class_weight, dict):
        print(
            "Custom class weights: "
            + ", ".join(f"class {k} → {v:.3f}" for k, v in class_weight.items())
        )
    else:
        print(f"Class weighting strategy: {class_weight}")
    
    data_path = Path("data/clinical_alerts.csv")
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print(f"\n[1] Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    print(f"   Loaded {len(df_raw)} rows")
    
    # 2. Feature engineering
    print(f"\n[2] Building patient-centric features...")
    df_feat, num_cols, cat_cols = build_features_patient_centric(df_raw, window=7)
    print(f"   Numerical features: {len(num_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")
    print(f"   Samples after feature engineering: {len(df_feat)}")
    
    df_feat = df_feat.copy()
    df_feat["alert_flag"] = df_raw.loc[df_feat.index, "alert_flag"].values
    
    # 3. Grouped split
    print(f"\n[3] Grouped split by patient_id...")
    groups = df_feat["patient_id"].values
    y = df_feat["alert_flag"].values
    X = df_feat[num_cols + cat_cols]
    
    random_state = 42
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    X_temp = X.iloc[temp_idx]
    y_temp = y[temp_idx]
    groups_temp = groups[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    
    val_idx_abs = df_feat.index[temp_idx[val_idx]]
    test_idx_abs = df_feat.index[temp_idx[test_idx]]
    train_idx_abs = df_feat.index[train_idx]
    
    print(f"   Train: {len(train_idx_abs)} samples ({len(np.unique(groups[train_idx]))} patients)")
    print(f"   Val:   {len(val_idx_abs)} samples")
    print(f"   Test:  {len(test_idx_abs)} samples")
    
    # 4. Train supervised
    print(f"\n[4] Training supervised baseline...")
    best_pipe, best_tau, best_val_report, best_params = train_supervised_baseline(
        df_feat, num_cols, cat_cols, train_idx_abs, val_idx_abs,
        metric="accuracy",
        recall_min=recall_min,
        random_state=random_state,
        target_accuracy=target_accuracy,
        target_tolerance=target_tolerance,
        class_weight=class_weight,
    )
    
    print(f"\n  Best params: {best_params}")
    print(f"  Best threshold (val): {best_tau:.4f}")
    print(f"  Validation report @ threshold:")
    print(f"    Accuracy: {best_val_report['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {best_val_report['balanced_accuracy']:.4f}")
    print(f"    Precision: {best_val_report['precision']:.4f}")
    print(f"    Recall: {best_val_report['recall']:.4f}")
    print(f"    F1: {best_val_report['f1']:.4f}")
    
    # 5. Refit on train+val
    print(f"\n[5] Refitting on train+val...")
    train_val_idx = np.concatenate([train_idx_abs, val_idx_abs])
    X_train_val = df_feat.loc[train_val_idx, num_cols + cat_cols]
    y_train_val = df_feat.loc[train_val_idx, "alert_flag"].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    
    classifier_final = HistGradientBoostingClassifier(
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        max_iter=best_params["max_iter"],
        random_state=random_state,
        class_weight=class_weight,
    )
    
    pipeline_final = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier_final),
    ])
    
    pipeline_final.fit(X_train_val, y_train_val)
    
    tau_final = best_tau
    
    # 6. Evaluate on test
    print(f"\n[6] Evaluating on test set...")
    test_metrics = evaluate_on_test(
        pipeline_final, tau_final, df_feat, num_cols, cat_cols,
        test_idx_abs, out_dir, prefix="supervised_test"
    )
    
    print(f"\n  Test metrics:")
    print(f"    Accuracy:        {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"    Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f} ({test_metrics['balanced_accuracy']*100:.2f}%)")
    print(f"    Precision:       {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"    Recall:          {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"    F1:              {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"    AUC-PR:          {test_metrics['auc_pr']:.4f} ({test_metrics['auc_pr']*100:.2f}%)")
    print(f"    AUC-ROC:         {test_metrics['auc_roc']:.4f} ({test_metrics['auc_roc']*100:.2f}%)")
    
    print(f"\n  Confusion matrix:")
    cm = test_metrics['confusion_matrix']
    print(f"                Predicted Normal  Predicted Anomaly")
    print(f"    True Normal      {cm[0][0]:6d}         {cm[0][1]:6d}")
    print(f"    True Anomaly     {cm[1][0]:6d}         {cm[1][1]:6d}")
    
    # 7. Save artifacts
    print(f"\n[7] Saving artifacts...")
    dump(pipeline_final, out_dir / "supervised_pipeline.joblib")
    
    threshold_data = {
        'tau': tau_final,
        'val_report': best_val_report,
        'params': best_params,
        'target_accuracy': target_accuracy,
        'target_accuracy_tolerance': target_tolerance,
        'recall_min': recall_min,
        'class_weight': class_weight,
    }
    with open(out_dir / "supervised_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    
    with open(out_dir / "supervised_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"   Saved: supervised_pipeline.joblib")
    print(f"   Saved: supervised_threshold.json")
    print(f"   Saved: supervised_test_metrics.json")
    print(f"   Saved: supervised_test_pr.png, supervised_test_roc.png")
    
    print("\n" + "="*70)
    print("[OK] TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

