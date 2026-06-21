import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Chargement du modèle entraîné et des données de test
model = joblib.load("models/random_forest_v1.pkl")
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Métriques principales
print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test, y_pred))
print("F1-score  :", f1_score(y_test, y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=["normal", "attaque"]))

# Accuracy : % de prédictions correctes au total. Attention, peut être trompeur si les classes sont déséquilibrées.
# Precision : parmi tout ce que le modèle a déclaré "attaque", combien étaient vraiment des attaques ? (mesure les fausses alertes)
# Recall : parmi toutes les vraies attaques, combien le modèle en a détecté ? (mesure les attaques ratées — critique pour un IDS, rater une vraie attaque est dangereux)
# F1-score : moyenne équilibrée entre precision et recall.
# Matrice de confusion : un tableau 2×2 montrant exactement où le modèle se trompe (vrais positifs, faux positifs, vrais négatifs, faux négatifs).