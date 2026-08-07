import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IDS - Détection d'intrusion", layout="wide")

st.title("🛡️ Système de Détection d'Intrusion (IDS)")
st.write("Chargez un fichier CSV de connexions réseau pour détecter les intrusions.")

# Charger le modèle et les outils de prétraitement (une seule fois, mis en cache)
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/random_forest_v2.pkl")
    encoders = joblib.load("models/encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, encoders, scaler

model, encoders, scaler = load_artifacts()

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

uploaded_file = st.file_uploader("Choisir un fichier CSV (format NSL-KDD, ex: KDDTest.txt)", type=["txt", "csv"])

if uploaded_file is not None:
    # Lecture du fichier (sans en-tête, comme les fichiers NSL-KDD)
    raw_df = pd.read_csv(uploaded_file, names=column_names)

    st.subheader("Aperçu des données chargées")
    st.dataframe(raw_df.head())

    # Préparation des données (même pipeline que data_prep.py)
    df = raw_df.copy()

    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        le = encoders[col]
        # Gérer les valeurs inconnues éventuelles
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    feature_cols = [col for col in column_names if col not in ["label", "difficulty"]]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    X = df[feature_cols]

    if st.button("🔍 Lancer l'analyse"):
        predictions = model.predict(X)

        results_df = raw_df.copy()
        results_df["Prédiction"] = predictions

        st.subheader("Résultats de l'analyse")

        # Statistiques résumées
        col1, col2, col3 = st.columns(3)
        total = len(results_df)
        attacks = (results_df["Prédiction"] != "normal").sum()
        normal = (results_df["Prédiction"] == "normal").sum()

        col1.metric("Total connexions", total)
        col2.metric("Connexions normales", normal)
        col3.metric("Attaques détectées", attacks)

        # Répartition par catégorie
        st.subheader("Répartition des prédictions")
        st.bar_chart(results_df["Prédiction"].value_counts())

        # Tableau détaillé avec mise en couleur
        st.subheader("Détail des connexions")

        def highlight_attacks(row):
            if row["Prédiction"] != "normal":
                return ["background-color: #ffcccc"] * len(row)
            else:
                return ["background-color: #ccffcc"] * len(row)

        display_cols = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "Prédiction"]

        st.write(f"Affichage des 500 premières lignes sur {len(results_df)} au total.")

        st.dataframe(
            results_df[display_cols].head(500).style.apply(highlight_attacks, axis=1),
            height=400
        )

        # Télécharger les résultats
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv,
            file_name="resultats_ids.csv",
            mime="text/csv"
        )
else:
    st.info("Veuillez charger un fichier pour commencer l'analyse. Vous pouvez utiliser data/KDDTest.txt comme exemple.")
