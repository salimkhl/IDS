import joblib
from sklearn.ensemble import RandomForestClassifier

X_train = joblib.load("data/X_train.pkl")
y_train = joblib.load("data/y_train.pkl")

print("Données chargées :", X_train.shape, y_train.shape)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("Entraînement en cours...")
model.fit(X_train, y_train)
print("Entraînement terminé !")

joblib.dump(model, "models/random_forest_v2.pkl")
print("Modèle sauvegardé : models/random_forest_v2.pkl")