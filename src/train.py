import os
import sys
from comet_ml import Experiment, API
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from dotenv import load_dotenv
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY") 
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "hugo-alvarez-3377")
COMET_PROJECT = os.getenv("COMET_PROJECT", "mlops-software-inteligente")

if not COMET_API_KEY:
    print("🚨 ERROR: Variable de entorno COMET_API_KEY faltante.")
    sys.exit(1)

# Variables de trazabilidad Git inyectadas por el pipeline
GIT_COMMIT = os.getenv("GITHUB_SHA", "local-dev")
GIT_AUTHOR = os.getenv("GITHUB_ACTOR", "local-user")

print(f"Iniciando CT Pipeline para el commit: {GIT_COMMIT[:7]}")

# 1. Obtener Umbral del Champion Actual desde Comet
api = API(api_key=COMET_API_KEY)
experiments = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)

f1_champion_actual = 0.0
old_champion_id = None

for exp in experiments:
    if "production-champion" in exp.get_tags():
        old_champion_id = exp.id
        metrics_summary = exp.get_metrics_summary()
        for m in metrics_summary:
            if m["name"] == "f1_score":
                f1_champion_actual = float(m["valueCurrent"])
                break
        break

print(f"🏆 F1-Score del Champion actual en Comet: {f1_champion_actual:.4f}")

# 2. Entrenar el Modelo Candidato
X, y = make_classification(n_samples=20000, n_features=20, n_classes=2, weights=[0.98, 0.02], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

experiment = Experiment(api_key=COMET_API_KEY, workspace=COMET_WORKSPACE, project_name=COMET_PROJECT)
experiment.set_name("CI-CD-Candidate-Run")

# Registrar Trazabilidad Git de forma explícita
experiment.log_other("git_commit_sha", GIT_COMMIT)
experiment.log_other("git_author", GIT_AUTHOR)

config_nueva = {"n_estimators": 160, "max_depth": 14}
experiment.log_parameters(config_nueva)

modelo_candidato = RandomForestClassifier(n_estimators=config_nueva["n_estimators"], max_depth=config_nueva["max_depth"], random_state=42, class_weight="balanced")
modelo_candidato.fit(X_train, y_train)

f1_nuevo = f1_score(y_test, modelo_candidato.predict(X_test))
experiment.log_metric("f1_score", f1_nuevo)

print(f"F1-Score del Candidato Nuevo: {f1_nuevo:.4f}")

# 3. Regla de Gobernanza MLOps
if f1_nuevo > f1_champion_actual:
    print("🔥 ¡ÉXITO! El candidato supera al campeón. Actualizando estados en Comet...")
    
    # Remover Tag del viejo Champion usando la API
    if old_champion_id:
        old_exp = api.get_experiment_by_key(old_champion_id)
        old_exp.remove_tag("production-champion")
        old_exp.add_tag("archived")
        
    # Asignar Tag al nuevo Champion
    experiment.add_tag("production-champion")
    
    # Subir Binario Certificado
    joblib.dump(modelo_candidato, "model.pkl")
    experiment.log_model("credit-risk-model", "model.pkl")
    os.remove("model.pkl")
else:
    print("❌ El modelo nuevo no supera los umbrales exigidos. Se descarta.")
    experiment.add_tag("rejected-candidate")

experiment.end()