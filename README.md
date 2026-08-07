# IDS — Système de Détection d'Intrusion basé sur le Machine Learning

Projet réalisé dans le cadre d'un stage d'été. Ce système détecte et classifie des intrusions réseau à partir du dataset académique NSL-KDD, en utilisant un modèle de Machine Learning de type Random Forest. Le projet inclut une interface web permettant de visualiser les résultats de détection.

## Objectif

Construire un pipeline complet de Machine Learning permettant de :
1. Préparer et nettoyer des données de trafic réseau brutes.
2. Entraîner un modèle de classification capable de distinguer une connexion normale d'une connexion malveillante, et d'identifier le type d'attaque.
3. Évaluer la fiabilité du modèle sur des données jamais vues à l'entraînement.
4. Visualiser les résultats de détection via une interface web interactive.

## Structure du projet

```
ids/
├── data/                        # Dataset NSL-KDD (non inclus, voir Installation)
│   ├── KDDTrain.txt
│   └── KDDTest.txt
├── models/
│   ├── random_forest_v1.pkl     # Modèle v1 (classification binaire)
│   ├── random_forest_v2.pkl     # Modèle v2 (classification multi-classes)
│   ├── encoders.pkl             # Encodeurs des variables catégorielles
│   └── scaler.pkl               # Normalisation des variables numériques
├── src/
│   ├── data_prep.py             # Préparation des données (multi-classes)
│   ├── train_model.py           # Entraînement du modèle Random Forest
│   ├── evaluate_model.py        # Évaluation du modèle
│   └── app.py                   # Interface web Streamlit
├── requirements.txt
└── README.md
```

## Technologies utilisées

- Python 3.14
- pandas, numpy — manipulation de données
- scikit-learn — Machine Learning (RandomForestClassifier, LabelEncoder, StandardScaler)
- joblib — sauvegarde/chargement des modèles entraînés
- streamlit — interface web interactive

## Installation

### 1. Créer et activer un environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset NSL-KDD

Placer les deux fichiers suivants dans le dossier `data/` :

- `KDDTrain.txt` : https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt
- `KDDTest.txt` : https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt

## Utilisation

### Pipeline d'entraînement

Exécuter les scripts dans l'ordre suivant :

```bash
python src/data_prep.py        # Étape 1 : préparation des données
python src/train_model.py      # Étape 2 : entraînement du modèle
python src/evaluate_model.py   # Étape 3 : évaluation du modèle
```

### Interface web

```bash
streamlit run src/app.py
```

Ouvre automatiquement une interface dans le navigateur permettant de charger un fichier de connexions réseau (format NSL-KDD) et de visualiser les résultats de détection : statistiques globales, répartition par catégorie d'attaque, tableau détaillé coloré, et export des résultats en CSV.

## Méthodologie

- **Dataset** : NSL-KDD, 125 973 connexions pour l'entraînement et 22 544 pour le test.
- **Type de tâche** : classification multi-classes — identification de la catégorie d'attaque (normal, dos, probe, r2l, u2r).
- **Prétraitement** : encodage des variables catégorielles (`protocol_type`, `service`, `flag`) via LabelEncoder, normalisation des variables numériques via StandardScaler (ajusté uniquement sur le train pour éviter toute fuite de données).
- **Modèle** : Random Forest (100 arbres, profondeur maximale de 20).
- **Catégories d'attaques** : les ~23 types d'attaques du dataset sont regroupés en 4 familles (DoS, Probe, R2L, U2R), en plus de la classe "normal".

## Résultats (v2 — classification multi-classes)

| Classe  | Précision | Rappel | F1-score | Support |
|---------|-----------|--------|----------|---------|
| dos     | 0.96      | 0.77   | 0.85     | 7460    |
| normal  | 0.65      | 0.97   | 0.78     | 9711    |
| probe   | 0.88      | 0.69   | 0.77     | 2421    |
| r2l     | 0.99      | 0.04   | 0.07     | 2885    |
| u2r     | 0.67      | 0.06   | 0.11     | 67      |

**Accuracy globale** : 75.1 %
**F1-score pondéré** : 70.9 %

### Interprétation

Le modèle détecte efficacement les catégories bien représentées dans les données d'entraînement (DoS, Normal, Probe). En revanche, les classes minoritaires — R2L (995 exemples sur 125 973, soit moins de 1 %) et U2R (52 exemples) — sont très mal détectées malgré une précision élevée : le modèle est extrêmement prudent sur ces classes rares et préfère prédire "normal" plutôt que de risquer une fausse alerte, ce qui se traduit par un rappel très faible.

### Tentative d'amélioration : `class_weight='balanced'`

Un test a été mené en ajustant le poids des classes lors de l'entraînement pour compenser le déséquilibre :

| Classe | Rappel (sans balanced) | Rappel (avec balanced) |
|--------|------------------------|--------------------------|
| r2l    | 4 %                    | 1 % (dégradé)            |
| u2r    | 6 %                    | 10 % (amélioré)          |

**Conclusion** : `class_weight='balanced'` améliore la détection de la classe la plus rare (u2r) mais dégrade celle de r2l, ce qui suggère que ce paramètre seul ne suffit pas à résoudre le déséquilibre de manière cohérente sur toutes les classes minoritaires. Des techniques plus avancées (SMOTE, ajustement fin des poids par classe) seraient nécessaires pour une amélioration robuste.

## Résultats (v1 — détection binaire, conservé pour référence)

| Métrique  | Valeur |
|-----------|--------|
| Accuracy  | 77.3 % |
| Precision (attaque) | 96.7 % |
| Recall (attaque)    | 62.3 % |
| F1-score  | 75.7 % |

## Pistes d'amélioration futures

- Comparaison avec d'autres algorithmes (Gradient Boosting, SVM)
- Techniques de rééquilibrage des classes (SMOTE)
- Validation croisée pour une évaluation plus robuste
- Migration vers un dataset plus récent et représentatif des menaces actuelles (ex. CICIDS2017)

## Auteur

Khalfallah Mohamed Salim — ISITCOM, Université de Sousse
