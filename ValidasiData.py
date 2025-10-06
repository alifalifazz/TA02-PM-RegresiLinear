import pandas as pd

# Ganti dengan path atau link ke dataset 

df = pd.read_csv("D:/Documents/Kuliah/Semester 5/Pembelajaran Mesin/PraktikumPembelajaranMesin/TA02/Dataset/advertising.csv")

print("--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Nama Kolom ---")
print(df.columns)