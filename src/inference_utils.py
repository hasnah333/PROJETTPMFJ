"""
Utilitaires pour l'inférence avec sélection automatique du meilleur seuil
"""

import json
from pathlib import Path
from typing import Optional


def get_best_threshold(
    thresholds_path: str = "artifacts/thresholds.json",
    strategy: str = "recall",  # "recall", "precision", "balanced"
) -> tuple[float, str]:
    """
    Retourne le meilleur seuil selon la stratégie choisie.
    
    Parameters:
    -----------
    thresholds_path : str
        Chemin vers le fichier thresholds.json
    strategy : str
        - "recall": Maximise le recall (tau_top7)
        - "precision": Maximise la précision (tau_p80)
        - "balanced": Équilibre (tau_f1)
    
    Returns:
    --------
    (threshold_value, threshold_key)
    """
    thresholds_file = Path(thresholds_path)
    
    if not thresholds_file.exists():
        raise FileNotFoundError(f"Fichier {thresholds_path} introuvable. Executez d'abord l'entrainement.")
    
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    
    if strategy == "recall":
        # Priorité: tau_recall_optimized > tau_top7 > tau_top10 > tau_f1
        tau = (thresholds.get("tau_recall_optimized") or 
               thresholds.get("tau_top7") or 
               thresholds.get("tau_top10") or 
               thresholds.get("tau_f1", 0.0))
        if "tau_recall_optimized" in thresholds:
            key = "tau_recall_optimized"
        elif "tau_top7" in thresholds:
            key = "tau_top7"
        elif "tau_top10" in thresholds:
            key = "tau_top10"
        else:
            key = "tau_f1"
        return float(tau), key
    
    elif strategy == "precision":
        # Priorité: tau_p80 > tau_p90 > tau_f1
        tau = thresholds.get("tau_p80") or thresholds.get("tau_p90") or thresholds.get("tau_f1", 0.0)
        key = "tau_p80" if "tau_p80" in thresholds else ("tau_p90" if "tau_p90" in thresholds else "tau_f1")
        return float(tau), key
    
    else:  # balanced
        # Utiliser tau_f1 par défaut
        tau = thresholds.get("tau_f1", 0.0)
        return float(tau), "tau_f1"


def predict_alert_auto(
    pipeline,
    sample_dict: dict,
    num_cols: list,
    cat_cols: list,
    thresholds_path: str = "artifacts/thresholds.json",
    strategy: str = "recall",
    patient_history: Optional[dict] = None,
    window: int = 7,
):
    """
    Prédiction avec sélection automatique du meilleur seuil.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        Pipeline entraîné
    sample_dict : dict
        Dictionnaire avec valeurs brutes
    num_cols : list
        Liste des colonnes numériques
    cat_cols : list
        Liste des colonnes catégorielles
    thresholds_path : str
        Chemin vers thresholds.json
    strategy : str
        "recall", "precision", ou "balanced"
    patient_history : dict, optional
        Historique du patient
    window : int
        Taille de fenêtre pour rolling statistics
    
    Returns:
    --------
    dict avec anomaly_score, alert_flag_pred, tau_used, strategy_used
    """
    from src.iso_fast import predict_with_threshold
    
    # Obtenir le meilleur seuil
    tau, tau_key = get_best_threshold(thresholds_path, strategy)
    
    # Prédire
    result = predict_with_threshold(
        pipeline,
        sample_dict,
        num_cols,
        cat_cols,
        tau,
        patient_history,
        window,
    )
    
    result["strategy_used"] = strategy
    result["threshold_key"] = tau_key
    
    return result

