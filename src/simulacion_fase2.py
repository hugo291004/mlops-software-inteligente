from comet_ml import Experiment
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import os
from dotenv import load_dotenv
load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY") 
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "hugo-alvarez-3377")
COMET_PROJECT = os.getenv("COMET_PROJECT", "mlops-software-inteligente")

X, y = make_classification(n_samples=15000, n_features=20, n_classes=2, weights=[0.98, 0.02], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

champion_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight="balanced")
champion_model.fit(X_train, y_train)

noise_levels = [0.4, 1.0, 1.6, 2.3]

for i, noise in enumerate(noise_levels, start=6):
    print(f"\n--- Ejecución {i}/9 (Data Drift) ---")
    
    experiment = Experiment(api_key=COMET_API_KEY, workspace=COMET_WORKSPACE, project_name=COMET_PROJECT)
    experiment.set_name(f"Data-Drift-Run-{i}")
    experiment.log_parameter("noise_level", noise)
    
    np.random.seed(42)
    noise_matrix = np.random.normal(loc=0.0, scale=noise, size=X_test.shape)
    X_test_drifted = X_test + noise_matrix
    
    y_pred = champion_model.predict(X_test_drifted)
    y_prob = champion_model.predict_proba(X_test_drifted)[:, 1]
    
    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    experiment.log_metrics(metrics)
    experiment.add_tags(["data-drift-simulation", "monitoring"])
    print(f"Ruido {noise} -> F1 Caído a: {metrics['f1_score']:.4f}")
    
    experiment.end()