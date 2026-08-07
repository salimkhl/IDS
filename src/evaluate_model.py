import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

model = joblib.load("models/random_forest_v2.pkl")
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

y_pred = model.predict(X_test)

print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall    :", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1-score  :", f1_score(y_test, y_pred, average='weighted', zero_division=0))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, zero_division=0))