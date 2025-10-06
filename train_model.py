import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Baca dataset
df = pd.read_csv("dataset/advertising.csv")

# Gunakan 1 variabel independen
X = df[['TV Ad Budget ($)']]
y = df['Sales ($)']

# Latih model
model = LinearRegression()
model.fit(X, y)

# Simpan model
pickle.dump(model, open("model/model.pkl", "wb"))

print("✅ Model berhasil dilatih dan disimpan sebagai model.pkl")
