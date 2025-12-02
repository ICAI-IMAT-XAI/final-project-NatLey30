import torch
import numpy as np
from flask import Flask, request, jsonify, Response
# from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

import sys
import os
sys.path.append(os.path.abspath("."))

from src.utils import load_model
from src.prediction import predict_with_scores

# -----------------------------
# Prometheus metrics
# -----------------------------
# PREDICTION_COUNTER = Counter(
#     'toxic_prediction_count',
#     'Contador de predicciones del modelo Toxicity Detection por categoría',
#     ['label']
# )

# -----------------------------
# Load the model
# -----------------------------
MODEL_PATH = "models/distilbert_toxic"

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(MODEL_PATH, device)
    id2label = list(model.config.id2label.values())
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None
    tokenizer = None
    id2label = []

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)


# @app.route('/metrics')
# def metrics():
#     return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500

    try:
        data = request.get_json(force=True)
        text = data["text"]

        result = predict_with_scores(
            model=model,
            tokenizer=tokenizer,
            text=text,
            id2label=id2label,
            device=device,
            threshold=0.5,
            top_k=3
        )

        # update Prometheus counters
        # for label, score in result["scores"].items():
        #     if score > 0.5:
        #         PREDICTION_COUNTER.labels(label=label).inc()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Iniciando API Toxicity en puerto 5000...")
    app.run(host='0.0.0.0', port=5000)
