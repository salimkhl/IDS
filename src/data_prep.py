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

train_df = pd.read_csv("data/KDDTrain.txt", names=column_names)
test_df = pd.read_csv("data/KDDTest.txt", names=column_names)

print("Forme du train :", train_df.shape)
print("Forme du test :", test_df.shape)
print(train_df.head())

# ===== CHANGÉ : multi-classe au lieu de binaire =====
attack_mapping = {
    'normal': 'normal',
    'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos',
    'pod': 'dos', 'land': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
    'processtable': 'dos', 'mailbomb': 'dos', 'worm': 'dos',
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
    'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
    'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
    'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

train_df["attack_category"] = train_df["label"].map(attack_mapping)
test_df["attack_category"] = test_df["label"].map(attack_mapping)


print("\nValeurs manquantes après mapping (train) :", train_df["attack_category"].isnull().sum())
print("Valeurs manquantes après mapping (test) :", test_df["attack_category"].isnull().sum())

print("\nRépartition train :")
print(train_df["attack_category"].value_counts())
print("\nRépartition test :")
print(test_df["attack_category"].value_counts())


from sklearn.preprocessing import LabelEncoder

categorical_cols = ["protocol_type", "service", "flag"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    encoders[col] = le

print("\nExemple après encodage (protocol_type, service, flag) :")
print(train_df[["protocol_type", "service", "flag"]].head())

from sklearn.preprocessing import StandardScaler

# ===== CHANGÉ : on exclut attack_category au lieu de binary_label =====
exclude_cols = ["protocol_type", "service", "flag", "label", "difficulty", "attack_category"]
numeric_cols = [col for col in train_df.columns if col not in exclude_cols]

scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

print("\nExemple après normalisation (duration, src_bytes, dst_bytes) :")
print(train_df[["duration", "src_bytes", "dst_bytes"]].head())

import joblib
import os

# ===== CHANGÉ : feature_cols exclut attack_category, y_train/y_test utilisent attack_category =====
feature_cols = [col for col in train_df.columns if col not in ["label", "difficulty", "attack_category"]]

X_train = train_df[feature_cols]
y_train = train_df["attack_category"]

X_test = test_df[feature_cols]
y_test = test_df["attack_category"]

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

os.makedirs("models", exist_ok=True)

joblib.dump(X_train, "data/X_train.pkl")
joblib.dump(y_train, "data/y_train.pkl")
joblib.dump(X_test, "data/X_test.pkl")
joblib.dump(y_test, "data/y_test.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nFichiers sauvegardés avec succès dans data/ et models/")