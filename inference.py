import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

from prometheus_client import start_http_server, Counter  # Tambahkan ini

model_prediction_count = Counter('model_prediction_count', 'Jumlah prediksi model')  # Tambahkan ini

# Daftar fitur yang digunakan saat training (urutan harus sama)
selected_features = [
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_approved',
    'Age_at_enrollment'
]

def predict_status(input_data, model_path, imputer_path=None):
    # Pastikan input dalam bentuk DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Reindex agar urutan dan jumlah kolom sama dengan training
    df_aligned = df.reindex(columns=selected_features, fill_value=0)

    # Imputasi missing values (gunakan imputer yang sama seperti training jika ada)
    if imputer_path:
        imputer = joblib.load(imputer_path)
        X = imputer.transform(df_aligned)
    else:
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(df_aligned)

    # Load model
    model = joblib.load(model_path)

    # Prediksi
    prediction = model.predict(X)

    # Update Prometheus metric
    model_prediction_count.inc()  # Tambahkan ini

    # Mapping hasil prediksi ke label asli
    status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}  # Sesuaikan dengan encoding saat training
    predicted_label = status_mapping.get(int(prediction[0]), prediction[0])

    return predicted_label

# Contoh penggunaan
if __name__ == "__main__":
    start_http_server(8000)  # Tambahkan ini agar metrics bisa di-scrape Prometheus

    input_data_raw = pd.DataFrame({
        'Curricular_units_2nd_sem_grade': [8.0],
        'Curricular_units_2nd_sem_approved': [1],
        'Curricular_units_1st_sem_grade': [8.0],
        'Tuition_fees_up_to_date': [1],
        'Curricular_units_1st_sem_approved': [0],
        'Age_at_enrollment': [25]
    })

    model_path = '../MSML-Submission/models/best_random_forest_model.pkl'
    imputer_path = None  # Ganti jika Anda menyimpan imputer

    hasil = predict_status(input_data_raw, model_path, imputer_path)
    print(f"Hasil Prediksi: {hasil}")

    # Prediksi 5000 kali untuk simulasi load
    for i in range(100):
        hasil = predict_status(input_data_raw, model_path, imputer_path)
        if (i+1) % 500 == 0:
            print(f"Prediksi ke-{i+1}: {hasil}")