import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Baca dataset (sesuaikan dengan nama folder yang benar: 'Dataset')
df = pd.read_csv("Dataset/advertising.csv")

# Gunakan 3 variabel independen: TV, Radio, Newspaper
feature_columns = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']
X = df[feature_columns]
y = df['Sales ($)']

# Latih model
model = LinearRegression()
model.fit(X, y)

# Simpan model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model (3 fitur) berhasil dilatih dan disimpan sebagai model.pkl")
