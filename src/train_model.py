import joblib
from sklearn.ensemble import RandomForestClassifier

# Chargement des données déjà préparées 
X_train = joblib.load("data/X_train.pkl")
y_train = joblib.load("data/y_train.pkl")

print("Données chargées :", X_train.shape, y_train.shape)

# Création du modèle
model = RandomForestClassifier(
    n_estimators=100,   # nombre d'arbres dans la forêt
    max_depth=20,        # profondeur max de chaque arbre
    random_state=42,     # pour que les résultats soient reproductibles
    n_jobs=-1             # utilise tous les cœurs du CPU pour aller plus vite
)

print("Entraînement en cours...")
model.fit(X_train, y_train)
print("Entraînement terminé !")

# Sauvegarde du modèle entraîné
joblib.dump(model, "models/random_forest_v1.pkl")
print("Modèle sauvegardé : models/random_forest_v1.pkl")