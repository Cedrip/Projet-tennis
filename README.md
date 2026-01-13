# Prédiction des Issues de Matchs de Tennis Professionnels (2000-2023)


## Contexte et Objectifs

Ce projet vise à prédire l'issue d'un match de tennis professionnel en utilisant des données historiques ATP/WTA et des techniques avancées de machine learning. L'enjeu est de modéliser la probabilité de victoire d'un joueur en s'appuyant sur des indicateurs de performance construits à partir de son historique.

**Objectif principal:** Proposer un modèle robuste, interprétable et performant, capable d'estimer la probabilité de victoire d'un joueur avant un match.

---

## Méthodologie

Le projet s'articule autour de cinq étapes clés :

1. **Collecte et préparation des données**
   - Nettoyage des données brutes
   - Structuration chronologique
   - Gestion des valeurs manquantes
   - Parsing des scores pour extraire des métriques (sets joués, tie-breaks)

2. **Feature Engineering avancé**
   - Transformation en format "long" (1 ligne = 1 joueur dans 1 match)
   - Calcul d'historiques de performance temporels
   - Création de features d'interaction et de momentum

3. **Entraînement de modèles**
   - Régression Logistique (baseline)
   - Random Forest
   - XGBoost (modèle principal)

4. **Évaluation rigoureuse**
   - Split temporel (80/20) pour éviter le data leakage
   - Métriques multiples : AUC, Accuracy, F1-Score, Log Loss, Brier Score
   - Analyse de calibration par déciles

5. **Interprétation et benchmarking**
   - Feature importance (gain et permutation)
   - Comparaison avec modèles naïfs et probabilités bookmaker

---

## Dataset

**Source:** Kaggle - ATP Tennis 2000-2023 Daily Pull  
**Volume:** 66,681 matchs professionnels ATP  
**Période:** 2000-2023  
**Variables principales:**
- Informations match : date, tournoi, surface, tour
- Joueurs : Player_1, Player_2, Winner
- Classements : Rank_1, Rank_2
- Scores détaillés
- Cotes bookmaker (Odd_1, Odd_2)

---

## Feature Engineering

### Transformation en Format Long
Conversion du format initial (1 ligne = 1 match avec 2 joueurs) en format long (1 ligne = 1 joueur dans 1 match) pour permettre le calcul d'historiques cohérents.

### Features Temporelles Calculées

**Historiques globaux:**
- `WR_All` : Win Rate historique du joueur (avant le match)
- `Matches_Before` : Nombre de matchs joués avant
- `Wins_Before` : Nombre de victoires cumulées

**Momentum récent:**
- `WR_Last5` : Win Rate sur les 5 derniers matchs
- `WR_Last100` : Win Rate sur les 100 derniers matchs

**Features par surface:**
- `WR_Surface` : Win Rate historique sur la surface spécifique

**Features adversaire:**
- Mêmes statistiques calculées pour l'adversaire (préfixe `Opp_`)

### Features d'Interaction

- `Rank_Ratio` : Ratio des classements (Rank / OppRank)
- `WR_Diff` : Différence de Win Rate (WR_All - Opp_WR_All)
- `WR_Ratio` : Ratio des Win Rates
- `Momentum_Diff` : Différence de momentum récent (5 matchs)
- `Momentum_Last100_Diff` : Différence de momentum (100 matchs)
- `Experience_Diff` : Différence d'expérience (nombre de matchs)

### Encodage
- Surface encodée en variable numérique
- Variables temporelles : Year, Month

**Total : 21 features** sélectionnées pour la modélisation


## Modélisation

### Split Temporel
- **Train:** 80% des données (matches les plus anciens)
- **Test:** 20% des données (matches les plus récents)
- Justification : Validation réaliste avec données futures

### Prétraitement
- Standardisation (StandardScaler) pour Régression Logistique
- Traitement des valeurs infinies et NaN (remplacement par médiane)
- Filtrage : joueurs avec minimum 5 matchs

### Modèles Entraînés

#### 1. Régression Logistique
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```
- Baseline interprétable
- Coefficients linéaires

#### 2. Random Forest
```python
RandomForestClassifier(
    n_estimators=300, max_depth=20, 
    min_samples_split=20, min_samples_leaf=10,
    max_features='sqrt', class_weight='balanced'
)
```
- Capture des interactions non-linéaires
- Robuste au bruit

#### 3. XGBoost (Modèle retenu)
```python
XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, 
    colsample_bytree=0.8
)
```
- Meilleure performance globale
- Gestion optimale des interactions complexes


## Résultats

### Comparaison des Modèles

| Modèle                | Accuracy | AUC-ROC | F1-Score | Log Loss | Brier Score |
|-----------------------|----------|---------|----------|----------|-------------|
| **XGBoost**           | **0.657**| **0.724**| **0.660**| **0.608**| **0.211**   |
| Random Forest         | 0.656    | 0.723   | 0.656    | 0.610    | 0.212       |
| Régression Logistique | 0.652    | 0.712   | 0.657    | 0.620    | 0.216       |

### Benchmarks Additionnels

| Modèle          | Accuracy | AUC-ROC | Brier Score |
|-----------------|----------|---------|-------------|
| **Bookmaker**   | 0.685    | 0.758   | 0.200       |
| Modèle Naïf (rang seul) | 0.645 | 0.645 | 0.355 |

**Observations:**
- XGBoost surpasse significativement le modèle naïf
- Les probabilités bookmaker restent supérieures (information privilégiée, expertise)
- XGBoost offre le meilleur compromis performance/interprétabilité


## Analyse des Performances

### 1. Feature Importance

**Top 5 Features (Gain XGBoost):**
1. `Rank_Ratio` (85.9) - Variable la plus discriminante
2. `Momentum_Last100_Diff` (29.3)
3. `WR_Diff` (27.2)
4. `Momentum_Diff` (10.7)
5. `OppRank` (10.6)

**Top 5 Features (Permutation Importance):**
1. `Rank_Ratio` (0.011)
2. `WR_Surface` (0.007)
3. `Momentum_Last100_Diff` (0.003)
4. `WR_Diff` (0.002)
5. `Experience_Diff` (0.002)

**Conclusion:** Le `Rank_Ratio` domine clairement, capturant le rapport de force global. Les features de momentum et de surface apportent une information complémentaire significative.

### 2. Stabilité Temporelle

**Brier Score par année (test set):**
- 2020: 0.190 (année COVID, calendrier réduit)
- 2021: 0.199
- 2022: 0.217
- 2023: 0.216
- 2024: 0.212
- 2025: 0.214

**Observation:** Performance stable dans le temps (fourchette 0.19-0.22), légère dégradation post-COVID probablement liée à la normalisation du calendrier.

### 3. Stabilité par Surface

**Brier Score par surface:**
- Hard : 0.210
- Grass : 0.211
- Clay : 0.213

**Observation:** Performance homogène sur toutes les surfaces, écarts négligeables.

### 4. Calibration du Modèle

**Analyse par déciles de probabilité:**

| Décile | Probabilité Moyenne | Win Rate Observé | Écart |
|--------|---------------------|------------------|-------|
| 1      | 0.153               | 0.155            | 0.002 |
| 5      | 0.476               | 0.473            | 0.003 |
| 10     | 0.855               | 0.851            | 0.004 |

**Conclusion:** Excellente calibration sur l'ensemble des déciles. Les probabilités prédites correspondent fidèlement aux taux de victoire observés.

### 5. Impact du Clipping

**Brier Score après clipping des probabilités [0.05, 0.95]:** 0.211  
**Conclusion:** Aucun gain mesurable, le modèle ne produit pas de probabilités extrêmes problématiques.

---

## Technologies Utilisées

**Langages et Environnements:**
- Python 3.10+
- Jupyter Notebook / Google Colab

**Bibliothèques principales:**
```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
```

**Outils:**
- Kaggle API (téléchargement dataset)
- StandardScaler (normalisation)
- LabelEncoder (encodage catégoriel)

---

## Structure du Projet

```
tennis-prediction/
│
├── data/
│   └── atp_tennis.csv          # Dataset brut
│
├── notebooks/
│   └── projet_tennis.ipynb     # Notebook principal
│
├── outputs/
│   ├── model_comparison.png    # Graphiques de comparaison
│   └── feature_importance.png  # Importance des variables
│
├── models/
│   └── xgboost_final.pkl       # Modèle XGBoost sauvegardé
│
├── README.md                    # Documentation
└── requirements.txt             # Dépendances Python
```

---

## Installation et Utilisation

### Prérequis
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn kagglehub
```

### Téléchargement des Données
```python
import kagglehub
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
```

### Exécution du Notebook
1. Ouvrir `projet_tennis.ipynb` dans Jupyter ou Google Colab
2. Exécuter les cellules séquentiellement
3. Le modèle XGBoost sera entraîné et évalué automatiquement

### Prédiction sur un Match Aléatoire
```python
# Exemple inclus dans le notebook
i = np.random.randint(0, len(X_test))
match_features = X_test.iloc[[i]]
proba = xgb_model.predict_proba(match_features)[0, 1]




Ce projet est réalisé dans un cadre académique.  
Dataset source : Kaggle (dissfya/atp-tennis-2000-2023daily-pull)
