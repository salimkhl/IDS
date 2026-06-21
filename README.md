    # IDS v1 — Système de Détection d'Intrusion basé sur le Machine Learning

    Projet réalisé dans le cadre d'un stage d'été. Ce système détecte des intrusions réseau (connexions normales vs attaques) à partir du dataset académique NSL-KDD, en utilisant un modèle de Machine Learning de type Random Forest.

    ## Objectif

    Construire un pipeline complet de Machine Learning permettant de :
    1. Préparer et nettoyer des données de trafic réseau brutes.
    2. Entraîner un modèle de classification capable de distinguer une connexion normale d'une connexion malveillante.
    3. Évaluer la fiabilité du modèle sur des données jamais vues à l'entraînement.

    ## Structure du projet
    ids_v1/

    ├── data/                        # Dataset NSL-KDD (non inclus dans l'archive, voir Installation)

    │   ├── KDDTrain.txt

    │   └── KDDTest.txt

    ├── models/

    │   ├── random_forest_v1.pkl     # Modèle entraîné (Random Forest)

    │   ├── encoders.pkl             # Encodeurs des variables catégorielles (protocol_type, service, flag)

    │   └── scaler.pkl               # Normalisation des variables numériques (StandardScaler)

    ├── src/

    │   ├── data_prep.py             # Chargement, encodage, normalisation, sauvegarde des données

    │   ├── train_model.py           # Entraînement du modèle Random Forest

    │   └── evaluate_model.py        # Évaluation du modèle (accuracy, precision, recall, F1)

    ├── requirements.txt

    └── README.md

    ## Technologies utilisées

    - Python 3.14
    - pandas, numpy — manipulation de données
    - scikit-learn — Machine Learning (RandomForestClassifier, LabelEncoder, StandardScaler)
    - joblib — sauvegarde/chargement des modèles entraînés

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

    Exécuter les scripts dans l'ordre suivant :

    ```bash
    python src/data_prep.py        # Étape 1 : préparation des données
    python src/train_model.py      # Étape 2 : entraînement du modèle
    python src/evaluate_model.py   # Étape 3 : évaluation du modèle
    ```

    ## Méthodologie

    - **Dataset** : NSL-KDD, 125 973 connexions pour l'entraînement et 22 544 pour le test.
    - **Type de tâche** : classification binaire (normal vs attaque).
    - **Prétraitement** : encodage des variables catégorielles (`protocol_type`, `service`, `flag`) via LabelEncoder, normalisation des variables numériques via StandardScaler (ajusté uniquement sur le train pour éviter toute fuite de données).
    - **Modèle** : Random Forest (100 arbres, profondeur maximale de 20).

    ## Résultats (v1 — détection binaire)

    | Métrique  | Valeur |
    |-----------|--------|
    | Accuracy  | 77.3 % |
    | Precision (attaque) | 96.7 % |
    | Recall (attaque)    | 62.3 % |
    | F1-score  | 75.7 % |

    **Interprétation** : le modèle est très fiable lorsqu'il signale une attaque (peu de fausses alertes), mais ne détecte qu'environ 62 % des attaques réelles. Cet écart s'explique en grande partie par la présence, dans le jeu de test, de types d'attaques absents du jeu d'entraînement — une caractéristique volontaire du dataset NSL-KDD destinée à évaluer la capacité de généralisation des modèles.

    ## Pistes d'amélioration (v2)

    - Classification multi-classe (identification du type précis d'attaque : DoS, Probe, R2L, U2R)
    - Amélioration du Recall via l'ajustement des hyperparamètres ou l'équilibrage des classes
    - Comparaison avec d'autres algorithmes (Gradient Boosting, SVM, réseaux de neurones)

    ## Auteur

    Khalfallah Mohamed Salim — ISITCOM, Université de Sousse