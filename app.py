from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Muat model regresi linear
MODEL_PATH = os.path.join("model", "model.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    tv_budget = None

    if request.method == 'POST':
        try:
            tv_budget = float(request.form['tv'])
            prediction = model.predict(np.array([[tv_budget]]))[0]
        except Exception as e:
            prediction = f"Input tidak valid: {e}"

    return render_template('index.html', prediction=prediction, tv=tv_budget)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))   
    app.run(host="0.0.0.0", port=port)         
