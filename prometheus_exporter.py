import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from flask import Flask, request, jsonify, Response
import joblib
import pandas as pd
from threading import Thread
import os

app = Flask(__name__)

# Load model dengan error handling
try:
    model = joblib.load('models/best_random_forest_model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    # Fallback: buat model dummy jika diperlukan
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    print("⚠️ Using dummy model")

# Daftar fitur
selected_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_approved',
    'Age_at_enrollment'
]

# Prometheus metrics dengan bucket yang didefinisikan
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP Requests',
    ['method', 'endpoint']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP Request Latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
)
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'RAM Usage Percentage')
MODEL_PREDICTION_COUNT = Counter(
    'model_prediction_count', 
    'Jumlah prediksi model',
    ['status']
)

# Background thread untuk update metrik sistem
def update_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            RAM_USAGE.set(psutil.virtual_memory().percent)
        except Exception as e:
            print(f"Metrics update error: {str(e)}")
        time.sleep(5)

@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    try:
        data = request.get_json()
        if not data:
            MODEL_PREDICTION_COUNT.labels(status='error').inc()
            return jsonify({"error": "No input data"}), 400
        
        # Pastikan input sesuai fitur
        df = pd.DataFrame([data])
        df = df.reindex(columns=selected_features, fill_value=0)
        
        # Prediksi
        prediction = model.predict(df)
        status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        predicted_label = status_mapping.get(int(prediction[0]), "Unknown")
        
        MODEL_PREDICTION_COUNT.labels(status='success').inc()
        return jsonify({"prediction": predicted_label})
    
    except Exception as e:
        MODEL_PREDICTION_COUNT.labels(status='error').inc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(duration)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": os.path.exists('models/best_random_forest_model.pkl')})

if __name__ == '__main__':
    # Jalankan thread untuk update metrik sistem
    Thread(target=update_system_metrics, daemon=True).start()
    
    # Jalankan aplikasi
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
# import time
# import psutil
# from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
# from flask import Flask, request, jsonify, Response
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model (pastikan path benar dan sudah di-mount ke container)
# model = joblib.load('models/best_random_forest_model.pkl')

# # Daftar fitur yang digunakan saat training
# selected_features = [
#     'Curricular_units_2nd_sem_grade',
#     'Curricular_units_2nd_sem_approved',
#     'Curricular_units_1st_sem_grade',
#     'Tuition_fees_up_to_date',
#     'Curricular_units_1st_sem_approved',
#     'Age_at_enrollment'
# ]

# # Prometheus metrics
# REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
# REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
# CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
# RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
# MODEL_PREDICTION_COUNT = Counter('model_prediction_count', 'Jumlah prediksi model')

# @app.route('/metrics', methods=['GET'])
# def metrics():
#     CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
#     RAM_USAGE.set(psutil.virtual_memory().percent)
#     return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# @app.route('/predict', methods=['POST'])
# def predict():
#     start_time = time.time()
#     REQUEST_COUNT.inc()
#     MODEL_PREDICTION_COUNT.inc()
#     data = request.get_json()
#     # Pastikan input sesuai fitur
#     df = pd.DataFrame([data])
#     df = df.reindex(columns=selected_features, fill_value=0)
#     # Prediksi
#     prediction = model.predict(df)
#     status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
#     predicted_label = status_mapping.get(int(prediction[0]), prediction[0])
#     duration = time.time() - start_time
#     REQUEST_LATENCY.observe(duration)
#     return jsonify({"prediction": predicted_label})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)