from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Muat model regresi linear (dilatih dengan 3 fitur)
MODEL_PATH = os.path.join("model", "model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Siapkan evaluasi performa model pada dataset pelatihan
metrics = None
try:
    # Coba beberapa kemungkinan path dataset (Windows case-insensitive, tapi jaga-jaga)
    possible_paths = [
        os.path.join("Dataset", "advertising.csv"),
        os.path.join("dataset", "advertising.csv"),
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is not None:
        df_eval = pd.read_csv(data_path)
        feature_columns = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']
        X_eval = df_eval[feature_columns]
        y_eval = df_eval['Sales ($)']

        # Sinkron dengan notebook: evaluasi pada test set dengan split tertentu
        X_train, X_test, y_train, y_test = train_test_split(
            X_eval, y_eval, test_size=0.3, random_state=101
        )

        # Gunakan model yang ter-load untuk memprediksi test set
        y_pred_eval = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred_eval))
        mae = float(mean_absolute_error(y_test, y_pred_eval))
        mse = float(mean_squared_error(y_test, y_pred_eval))
        rmse = float(np.sqrt(mse))
        metrics = { 'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse }
except Exception:
    metrics = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    tv_budget = None
    radio_budget = None
    newspaper_budget = None
    feature_names = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']

    # Ambil koefisien dan intercept dari model
    try:
        coefficients = [float(c) for c in np.ravel(model.coef_)]
    except Exception:
        coefficients = None
    try:
        intercept = float(model.intercept_)
    except Exception:
        intercept = None

    base_sales = intercept if intercept is not None else None
    contributions = None
    total_media_contribution = None

    if request.method == 'POST':
        try:
            tv_budget = float(request.form.get('tv', ''))
            radio_budget = float(request.form.get('radio', ''))
            newspaper_budget = float(request.form.get('newspaper', ''))

            features = np.array([[tv_budget, radio_budget, newspaper_budget]])
            prediction = float(model.predict(features)[0])

            if coefficients is not None:
                tv_contrib = tv_budget * coefficients[0]
                radio_contrib = radio_budget * coefficients[1]
                news_contrib = newspaper_budget * coefficients[2]
                contributions = {
                    'TV': float(tv_contrib),
                    'Radio': float(radio_contrib),
                    'Newspaper': float(news_contrib),
                }
                total_media_contribution = float(tv_contrib + radio_contrib + news_contrib)
            
            if total_media_contribution is None and base_sales is not None and prediction is not None:
                total_media_contribution = float(prediction - base_sales)
        except Exception as e:
            prediction = f"Input tidak valid: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        tv=tv_budget,
        radio=radio_budget,
        newspaper=newspaper_budget,
        coefficients=coefficients,
        intercept=intercept,
        feature_names=feature_names,
        base_sales=base_sales,
        contributions=contributions,
        total_media_contribution=total_media_contribution,
        metrics=metrics,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
