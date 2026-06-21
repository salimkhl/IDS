import pandas as pd

# Les 41 noms de colonnes + label + difficulty
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Chargement du dataset
train_df = pd.read_csv("data/KDDTrain.txt", names=column_names)
test_df = pd.read_csv("data/KDDTest.txt", names=column_names)

print("Forme du train :", train_df.shape)
print("Forme du test :", test_df.shape)
print(train_df.head())

# Création du label binaire : 0 = normal, 1 = attaque
train_df["binary_label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
test_df["binary_label"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

print("\nRépartition train :")
print(train_df["binary_label"].value_counts())

print("\nRépartition test :")
print(test_df["binary_label"].value_counts())


from sklearn.preprocessing import LabelEncoder

categorical_cols = ["protocol_type", "service", "flag"]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # On entraîne l'encodeur sur le train ET le test combinés
    # pour être sûr qu'aucune valeur du test ne soit "inconnue"
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)

    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

    encoders[col] = le

print("\nExemple après encodage (protocol_type, service, flag) :")
print(train_df[["protocol_type", "service", "flag"]].head())

from sklearn.preprocessing import StandardScaler

# Colonnes à exclure de la normalisation 
exclude_cols = ["protocol_type", "service", "flag", "label", "difficulty", "binary_label"]
numeric_cols = [col for col in train_df.columns if col not in exclude_cols]

scaler = StandardScaler()

# On apprend la moyenne/écart-type UNIQUEMENT sur le train (règle d'or en ML)
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

print("\nExemple après normalisation (duration, src_bytes, dst_bytes) :")
print(train_df[["duration", "src_bytes", "dst_bytes"]].head())

import joblib
import os

# Colonnes qu'on utilise vraiment pour entraîner (on retire label texte, difficulty, binary_label)
feature_cols = [col for col in train_df.columns if col not in ["label", "difficulty", "binary_label"]]

X_train = train_df[feature_cols]
y_train = train_df["binary_label"]

X_test = test_df[feature_cols]
y_test = test_df["binary_label"]

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Création du dossier models si besoin
os.makedirs("models", exist_ok=True)

# Sauvegarde des données prêtes à l'emploi
joblib.dump(X_train, "data/X_train.pkl")
joblib.dump(y_train, "data/y_train.pkl")
joblib.dump(X_test, "data/X_test.pkl")
joblib.dump(y_test, "data/y_test.pkl")

# Sauvegarde des encodeurs et du scaler 
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nFichiers sauvegardés avec succès dans data/ et models/")