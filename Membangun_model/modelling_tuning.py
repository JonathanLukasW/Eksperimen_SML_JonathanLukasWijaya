import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import mlflow
import dagshub

REPO_OWNER = "JonathanLukasW" 
REPO_NAME = "Eksperimen_SML_JonathanLukasWijaya"

print("Menghubungkan ke DagsHub")
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

TRAIN_PATH = os.path.join("csgo_preprocessed", "train_clean.csv")
TEST_PATH  = os.path.join("csgo_preprocessed", "test_clean.csv")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["winningSide"])
y_train = train_data["winningSide"]
X_test = test_data.drop(columns=["winningSide"])
y_test = test_data["winningSide"]

base_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(f"[SUCCESS] Hyperparameter terbaik ditemukan: {best_params}")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import f1_score
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"Hasil Evaluasi -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

mlflow.autolog(disable=True)

with mlflow.start_run(run_name="Random_Forest_Tuning_Jonathan"):
    print("[MLFLOW] Memulai proses manual logging ke DagsHub...")
    
    for param_name, param_val in best_params.items():
        mlflow.log_param(param_name, param_val)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("developer", "Jonathan Lukas Wijaya")
    mlflow.log_param("cv_folds", 3) 
    mlflow.log_param("tuning_method", "GridSearchCV") 

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
    plt.title('Confusion Matrix - CS:GO Predictor')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)

    mlflow.sklearn.log_model(best_model, "csgo_rf_model")
    
    print("Semua parameter, metrik, dan 2 artefak berhasil di-upload ke DagsHub!")

if os.path.exists(cm_path): os.remove(cm_path)
if os.path.exists(roc_path): os.remove(roc_path)