# Système d'Alerte Clinique Multi-Paramètres

## 📋 Description du Projet

Ce projet implémente un système de détection d'alertes cliniques basé sur un modèle supervisé (HistGradientBoostingClassifier). Le système analyse des données multi-paramètres (fréquence cardiaque, sommeil, activité, etc.) pour détecter des anomalies cliniques avec une précision de **95.21%**.

## 🎯 Objectifs

- Détecter les anomalies cliniques à partir de données multi-paramètres
- Utiliser un feature engineering centré patient pour capturer les déviations intra-patient
- Maximiser l'accuracy tout en maintenant un recall acceptable (≥ 25%)
- Fournir un pipeline reproductible pour l'entraînement et l'inférence

## 📊 Données

Le dataset contient les colonnes suivantes :

### Identifiants
- `patient_id` : Identifiant unique du patient
- `timestamp` ou `date` : Date/heure de l'enregistrement

### Variables Continues
- `heart_rate` : Fréquence cardiaque (bpm)
- `hr_variability` : Variabilité de la fréquence cardiaque
- `steps` : Nombre de pas
- `mood_score` : Score d'humeur
- `sleep_duration_hours` : Durée du sommeil (heures)
- `sleep_efficiency` : Efficacité du sommeil
- `num_awakenings` : Nombre de réveils
- `age` : Âge du patient

### Variables Catégorielles
- `weekend` : Indicateur week-end (0/1)
- `medication_taken` : Prise de médicament (0/1)
- `is_female` : Genre féminin (0/1)
- `day_of_week` : Jour de la semaine (0-6)

### Label
- `alert_flag` : Indicateur d'alerte (0 = Normal, 1 = Anomalie)

## 🔧 Étapes de Preprocessing

### 1. Tri et Préparation des Données

**Objectif** : Organiser les données par patient et par ordre chronologique.

```python
# Trier par patient_id et timestamp
df = df.sort_values(by=["patient_id", timestamp_col]).reset_index(drop=True)
```

**Pourquoi** : Les features basées sur les fenêtres glissantes nécessitent un ordre temporel correct.

### 2. Calcul des Statistiques Glissantes (Rolling Windows)

**Objectif** : Calculer la moyenne et l'écart-type sur une fenêtre de 7 jours pour chaque patient.

**Variables traitées** :
- `heart_rate`, `hr_variability`, `steps`, `mood_score`
- `sleep_duration_hours`, `sleep_efficiency`, `num_awakenings`

**Méthode** :
- Fenêtre glissante de 7 jours
- Minimum 3 observations requises
- Calcul séparé pour chaque patient (groupby)

**Formule** :
```
roll_mean_t = moyenne(x_{t-6}, ..., x_t)
roll_std_t = écart-type(x_{t-6}, ..., x_t)
```

### 3. Features Delta

**Objectif** : Capturer l'écart absolu par rapport à la moyenne récente du patient.

**Formule** :
```
delta_t = x_t - roll_mean_t
```

**Exemple** : Si la fréquence cardiaque moyenne d'un patient sur 7 jours est 70 bpm et qu'aujourd'hui elle est de 85 bpm, alors `heart_rate_delta = 15`.

**Pourquoi** : Élimine l'effet du profil patient (certains patients ont naturellement une fréquence cardiaque plus élevée).

### 4. Features Z-Score

**Objectif** : Normaliser l'écart par rapport à la variabilité normale du patient.

**Formule** :
```
z_score_t = delta_t / (roll_std_t + ε)
```
où ε = 10⁻⁶ pour éviter la division par zéro.

**Exemple** : Si un patient a une variabilité normale de 5 bpm et un écart de 15 bpm, alors `heart_rate_z = 3` (3 écarts-types).

**Pourquoi** : Un écart de 15 bpm est plus significatif pour un patient avec une faible variabilité que pour un patient avec une forte variabilité.

### 5. Features Dérivées

#### 5.1 Transformation Logarithmique des Pas
```
steps_log1p = log(1 + steps)
```
**Pourquoi** : Réduit l'impact des valeurs extrêmes et normalise la distribution.

#### 5.2 Ratio d'Éveils par Heure
```
awakenings_per_hour = num_awakenings / max(sleep_duration_hours, 0.5)
```
**Pourquoi** : Normalise le nombre de réveils par rapport à la durée du sommeil.

#### 5.3 Encodage Cyclique du Jour de la Semaine
```
dow_sin = sin(2π × day_of_week / 7)
dow_cos = cos(2π × day_of_week / 7)
```
**Pourquoi** : Capture la cyclicité hebdomadaire (lundi et dimanche sont proches dans l'espace cyclique).

#### 5.4 Ratio Fréquence Cardiaque / Variabilité
```
hr_hrv_ratio = heart_rate / max(hr_variability, 10⁻³)
```
**Pourquoi** : Ratio physiologique important pour la santé cardiovasculaire.

#### 5.5 Dette de Sommeil
```
sleep_debt = max(0, 7.5 - sleep_duration_hours)
```
**Pourquoi** : Quantifie le déficit de sommeil par rapport à une référence de 7.5 heures.

### 6. Sélection des Features Finales

#### Features Numériques (utilisées par le modèle)
- **Delta features** : `heart_rate_delta`, `hr_variability_delta`, `steps_delta`, `mood_score_delta`, `sleep_duration_hours_delta`, `sleep_efficiency_delta`, `num_awakenings_delta`
- **Z-score features** : `heart_rate_z`, `hr_variability_z`, `steps_z`, `mood_score_z`, `sleep_duration_hours_z`, `sleep_efficiency_z`, `num_awakenings_z`
- **Dérivées** : `steps_log1p`, `awakenings_per_hour`, `hr_hrv_ratio`, `sleep_debt`
- **Autres** : `age`, `dow_sin`, `dow_cos`

#### Features Catégorielles
- `weekend` (0/1)
- `medication_taken` (0/1)
- `is_female` (0/1)

### 7. Pipeline de Preprocessing

**Objectif** : Normaliser les features numériques et encoder les features catégorielles.

**Méthode** :
- **Features numériques** : Normalisation avec `StandardScaler` (moyenne = 0, écart-type = 1)
- **Features catégorielles** : Encodage one-hot avec gestion des valeurs inconnues

**Code** :
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)
```

### 8. Gestion des Valeurs Manquantes

**Stratégie** : Suppression des lignes avec valeurs manquantes dans les features.

**Pourquoi** : Garantit que toutes les features sont calculables et cohérentes. Pas d'imputation pour éviter d'introduire des biais.

## 🤖 Modèle Supervisé

### Choix du Modèle : HistGradientBoostingClassifier

**Pourquoi ce modèle** :
- ✅ Efficacité : Entraînement rapide sur de grands datasets
- ✅ Performance : Excellent pour les problèmes de classification binaire
- ✅ Robustesse : Gère bien les features mixtes (numériques + catégorielles)
- ✅ Interprétabilité : Importance des features disponible

### Architecture du Pipeline

Le pipeline complet est composé de deux étapes :

```python
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(...)),  # Preprocessing
    ("classifier", HistGradientBoostingClassifier(...)),  # Modèle
])
```

## 🎓 Entraînement et Optimisation

### 1. Division des Données

**Méthode** : `GroupShuffleSplit` pour maintenir l'intégrité des patients.

**Répartition** :
- **Train** : 60% des patients
- **Validation** : 20% des patients
- **Test** : 20% des patients

**Pourquoi** : Évite le **data leakage** inter-patient (même patient dans train et test).

### 2. Recherche d'Hyperparamètres (Grid Search)

**Hyperparamètres optimisés** :

| Hyperparamètre | Valeurs testées |
|----------------|-----------------|
| `learning_rate` | 0.05, 0.1 |
| `max_depth` | None, 6, 10 |
| `max_iter` | 200, 400 |

**Total** : 2 × 3 × 2 = **12 combinaisons** testées

### 3. Optimisation du Seuil de Classification

Pour chaque combinaison d'hyperparamètres :

1. **Entraînement** du modèle sur l'ensemble d'entraînement
2. **Calcul des scores** de probabilité sur l'ensemble de validation
3. **Balayage de 100 seuils** (quantiles de 50% à 99.5%)
4. **Sélection du seuil** qui maximise l'accuracy
5. **Contrainte** : recall ≥ 25% (évite la solution triviale "tout prédire Normal")

**Formule de prédiction** :
```
y_pred = 1 si score ≥ τ
y_pred = 0 sinon
```

### 4. Sélection du Meilleur Modèle

**Critères de sélection** (par ordre de priorité) :
1. **Accuracy sur validation** (critère principal)
2. **Balanced Accuracy** (en cas d'égalité)
3. **AUC-PR** (en cas d'égalité supplémentaire)

### 5. Entraînement Final

Une fois le meilleur modèle sélectionné :

1. **Ré-entraînement** sur train + validation
2. **Re-calibration** du seuil sur train + validation
3. **Évaluation finale** sur l'ensemble de test

## 📈 Évaluation

### Métriques Utilisées

| Métrique | Formule | Description |
|----------|---------|-------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Proportion de prédictions correctes |
| **Balanced Accuracy** | 0.5 × (TP/(TP+FN) + TN/(TN+FP)) | Accuracy équilibrée pour les classes déséquilibrées |
| **Precision** | TP / (TP + FP) | Proportion de vrais positifs parmi les prédictions positives |
| **Recall** | TP / (TP + FN) | Proportion d'anomalies détectées |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Moyenne harmonique de précision et recall |
| **AUC-ROC** | Aire sous la courbe ROC | Capacité à distinguer les classes |
| **AUC-PR** | Aire sous la courbe Precision-Recall | Performance sur classe minoritaire |

### Résultats sur l'Ensemble de Test

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **95.21%** |
| Balanced Accuracy | 90.34% |
| Precision | 86.53% |
| Recall | 83.17% |
| F1-Score | 84.82% |
| AUC-ROC | 98.85% |
| AUC-PR | 94.71% |

### Matrice de Confusion

| | Prédit Normal | Prédit Anomalie |
|---|---|---|
| **Réel Normal** | 1571 | 40 |
| **Réel Anomalie** | 52 | 257 |

**Seuil utilisé** : 0.5078

### Analyse des Résultats

✅ **Précision élevée** : 95.21% d'accuracy, excellent pour un problème de détection d'anomalies

✅ **Bon équilibre** : Balanced Accuracy de 90.34% indique que le modèle performe bien sur les deux classes malgré le déséquilibre (prévalence = 16.09%)

✅ **Recall acceptable** : 83.17% de recall signifie que le modèle détecte 83% des anomalies réelles

✅ **Peu de faux positifs** : Seulement 40 faux positifs sur 1611 cas normaux (2.48%)

✅ **Peu de faux négatifs** : 52 faux négatifs sur 309 anomalies réelles (16.83%)

## 🚀 Utilisation

### 1. Entraînement du Modèle

```bash
python -m src.train_supervised
```

> Besoin de limiter l'accuracy (ex: viser ~95%) ? Utilisez l'argument optionnel :

```bash
python -m src.train_supervised --target-accuracy 0.95 --target-accuracy-tolerance 0.01
```

`--target-accuracy` accepte une valeur entre 0 et 1 (ou 0-100) et ajuste automatiquement le seuil de décision pour se rapprocher de cette accuracy tout en respectant la contrainte de recall minimale.

> Besoin d'augmenter la sensibilité aux anomalies ? Ajustez le poids de la classe positive :

```bash
python -m src.train_supervised --positive-class-weight 2.0
```

Par défaut le modèle utilise `class_weight="balanced"` pour compenser l'imbalance. Avec `--positive-class-weight`, vous imposez manuellement un ratio (classe 0 → 1.0, classe 1 → valeur fournie) tout en conservant les données originales.

**Sorties** :
- `artifacts/supervised_pipeline.joblib` : Pipeline complet
- `artifacts/supervised_threshold.json` : Seuil optimal
- `artifacts/supervised_test_metrics.json` : Métriques sur test
- `artifacts/supervised_test_roc.png` : Courbe ROC
- `artifacts/supervised_test_pr.png` : Courbe Precision-Recall

### 📓 Processus complet (Notebook)

Besoin d'un fil conducteur unique qui regroupe toutes les étapes (préparation des données, feature engineering centré patient, entraînement, optimisation du seuil, évaluation et inférence) ?  
Consultez le notebook `notebooks/complete_preprocessing_and_model.ipynb`. Il documente pas à pas le pipeline complet, avec du code exécutable et des commentaires pour reproduire exactement les résultats présentés dans ce dépôt.

### 2. Preprocessing des Données

```python
from src.preprocessing_supervised import build_features_patient_centric

# Charger les données
df = pd.read_csv("data/clinical_alerts.csv")

# Feature engineering
df_feat, num_cols, cat_cols = build_features_patient_centric(df, window=7)
```

### 3. Chargement du Modèle

```python
from src.preprocessing_supervised import load_supervised_model

# Charger le pipeline
pipeline = load_supervised_model("artifacts/supervised_pipeline.joblib")
```

### 4. Prédiction

```python
from src.preprocessing_supervised import predict_with_supervised_model

# Préparer les features
X = df_feat[num_cols + cat_cols]

# Prédire
results = predict_with_supervised_model(pipeline, X)

# Résultats
scores = results['scores']  # Probabilités
predictions = results['predictions']  # Prédictions binaires (0/1)
threshold = results['threshold_used']  # Seuil utilisé
```

### 5. Prédiction sur un Nouvel Échantillon

```python
from src.preprocessing_supervised import build_features_patient_centric, load_supervised_model, predict_with_supervised_model

# Charger le modèle
pipeline = load_supervised_model()

# Nouvel échantillon (doit contenir l'historique du patient pour les features delta/z-score)
sample_df = pd.DataFrame([{
    'patient_id': 1,
    'timestamp': '2024-01-15',
    'heart_rate': 85,
    'hr_variability': 45,
    # ... autres colonnes
}])

# Feature engineering
df_feat, num_cols, cat_cols = build_features_patient_centric(sample_df, window=7)

# Prédire
X = df_feat[num_cols + cat_cols]
results = predict_with_supervised_model(pipeline, X)
```

## 📁 Structure du Projet

```
P_AI/
├── data/
│   └── clinical_alerts.csv          # Données d'entraînement
├── src/
│   ├── preprocessing_supervised.py  # Feature engineering et preprocessing
│   ├── train_supervised.py          # Script d'entraînement
│   └── inference_utils.py           # Utilitaires d'inférence
├── artifacts/
│   ├── supervised_pipeline.joblib   # Modèle entraîné
│   ├── supervised_threshold.json    # Seuil optimal
│   ├── supervised_test_metrics.json # Métriques de test
│   ├── supervised_test_roc.png     # Courbe ROC
│   └── supervised_test_pr.png      # Courbe Precision-Recall
└── notebooks/
    └── complete_preprocessing_and_model.ipynb  # Notebook d'analyse complète
```

## 📦 Dépendances

- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `scikit-learn` : Machine learning
- `matplotlib` : Visualisation
- `joblib` : Sauvegarde/chargement de modèles

## 🔍 Points Clés du Preprocessing

### Pourquoi Centré Patient ?

Les valeurs absolues (ex: fréquence cardiaque = 85 bpm) ne sont pas significatives sans contexte. Un patient avec une fréquence cardiaque normale de 60 bpm et un autre avec une normale de 90 bpm ont des profils différents.

**Solution** : Utiliser des features relatives au profil du patient (delta, z-score) plutôt que des valeurs absolues.

### Pourquoi Rolling Windows ?

Les statistiques sur une fenêtre glissante de 7 jours capturent :
- La tendance récente du patient
- La variabilité normale du patient
- Les changements progressifs ou soudains

### Pourquoi Pas d'Imputation ?

Les valeurs manquantes dans les features delta/z-score indiquent souvent :
- Données insuffisantes pour calculer les statistiques glissantes
- Nouveaux patients sans historique

L'imputation pourrait introduire des biais, donc on préfère supprimer ces cas.

## 📝 Notes Importantes

1. **Historique Patient Requis** : Les features delta/z-score nécessitent un historique de 7 jours minimum pour chaque patient. Pour les nouveaux patients, ces features seront NaN et seront supprimées.

2. **Ordre Temporel** : Les données doivent être triées par `patient_id` et `timestamp` avant le feature engineering.

3. **Seuil Optimal** : Le seuil de classification (0.5078) a été optimisé sur l'ensemble de validation. Il peut être ajusté selon les besoins cliniques (priorité recall vs precision).

4. **Reproductibilité** : Le modèle utilise `random_state=42` pour garantir la reproductibilité.

## 🎯 Conclusion

Le système de détection d'alertes cliniques utilise un preprocessing sophistiqué centré patient et un modèle supervisé performant (HistGradientBoostingClassifier). Les résultats montrent :

- ✅ **Performance élevée** : 95.21% d'accuracy avec un bon équilibre entre précision et recall
- ✅ **Robustesse** : Le feature engineering centré patient capture efficacement les déviations intra-patient
- ✅ **Reproductibilité** : Pipeline complet sauvegardé pour l'inférence
- ✅ **Flexibilité** : Support de différentes stratégies de seuil selon les besoins cliniques

Le modèle est prêt pour l'intégration dans un système de production pour la détection d'alertes cliniques en temps réel.

