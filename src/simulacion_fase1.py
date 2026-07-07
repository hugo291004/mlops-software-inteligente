from comet_ml import Experiment, API
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from dotenv import load_dotenv
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY") 
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "hugo-alvarez-3377")
COMET_PROJECT = os.getenv("COMET_PROJECT", "mlops-software-inteligente")

X, y = make_classification(
    n_samples=15000, n_features=20, n_classes=2, 
    weights=[0.98, 0.02], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

experiment_configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 150, "max_depth": 12}, 
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 250, "max_depth": None}
]

mejor_f1 = 0.0
mejor_run_id = None

for i, config in enumerate(experiment_configs, start=1):
    print(f"\n--- Ejecución {i}/{len(experiment_configs)} en Comet ---")
    
    experiment = Experiment(
        api_key=COMET_API_KEY,
        workspace=COMET_WORKSPACE,
        project_name=COMET_PROJECT,
        auto_output_logging="simple"
    )
    experiment.set_name(f"RandomForest-Sim-Run-{i}")
    experiment.log_parameters(config)
    
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    experiment.log_metrics(metrics)
    
    joblib.dump(model, "model.pkl")
    experiment.log_model("credit-risk-model", "model.pkl")
    
    experiment.add_tag("experimentation")
    
    print(f"Métricas registradas -> F1: {metrics['f1_score']:.4f}")
    
    if metrics["f1_score"] > mejor_f1:
        mejor_f1 = metrics["f1_score"]
        mejor_run_id = experiment.get_key() 
        
    experiment.end()

if os.path.exists("model.pkl"):
    os.remove("model.pkl")

print(f"\n🏆 Bucle de experimentación finalizado.")
print(f"El F1-Score más alto fue: {mejor_f1:.4f}")
print(f"Conectando a la API para promover el experimento '{mejor_run_id}' a Champion...")

api = API(api_key=COMET_API_KEY)
experimento_ganador = api.get_experiment_by_key(mejor_run_id)
experimento_ganador.add_tag("production-champion")

print("¡Etiqueta 'production-champion' asignada automáticamente con éxito!")