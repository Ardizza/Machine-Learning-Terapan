# Laporan Proyek Machine Learning - Rafi Ardizza Fadhillah Setiadi
## Domain Proyek
### Latar Belakang
Penyakit jantung adalah salah satu penyebab utama kematian di seluruh dunia. Deteksi dini dan penanganan yang tepat dapat menyelamatkan banyak nyawa. Menggunakan data medis, kita dapat mengembangkan model machine learning yang mampu memprediksi apakah seorang pasien memiliki penyakit jantung, yang pada akhirnya dapat membantu dokter dalam mengambil keputusan yang lebih baik dan lebih cepat.

### Mengapa dan bagaimana masalah ini harus diselesaikan:
* Penyakit jantung sering kali tidak menunjukkan gejala awal yang jelas, sehingga banyak kasus baru terdeteksi saat sudah pada tahap lanjut.
* Model prediksi dapat memberikan alat tambahan bagi tenaga medis untuk screening awal dan pencegahan.

## Business Understanding
### Problem Statements
* Bagaimana cara memprediksi apakah seorang pasien memiliki penyakit jantung berdasarkan data medis yang tersedia?
* Algoritma machine learning mana yang memberikan hasil prediksi terbaik untuk kasus ini?

### Goals
* Mengembangkan model machine learning yang dapat memprediksi penyakit jantung dengan akurasi tinggi.
* Membandingkan beberapa algoritma untuk menentukan model terbaik.

### Solution Statements
* Menggunakan Logistic Regression sebagai baseline model.
* Membandingkan hasil prediksi dengan algoritma Random Forest.
* Melakukan hyperparameter tuning pada model Random Forest untuk meningkatkan akurasi.

## Data Understanding
Dataset yang digunakan adalah dari Kaggle dengan link [Heart Disease Dataset](https://www.kaggle.com/datasets/data855/heart-disease) Dataset ini berisi 1025 sampel dan 14 atribut.

### Variabel pada Heart Disease UCI dataset adalah sebagai berikut:
age: Umur pasien.
sex: Jenis kelamin (1 = laki-laki, 0 = perempuan).
cp: Tipe nyeri dada (0-3).
trestbps: Tekanan darah saat istirahat.
chol: Kolesterol serum dalam mg/dl.
fbs: Gula darah puasa > 120 mg/dl (1 = benar, 0 = salah).
restecg: Hasil elektrokardiografi istirahat (0-2).
thalach: Detak jantung maksimal yang tercapai.
exang: Angina akibat olahraga (1 = ya, 0 = tidak).
oldpeak: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.
slope: Kemiringan segmen ST latihan puncak (0-2).
ca: Jumlah pembuluh darah utama (0-4) yang diwarnai oleh fluoroskopi.
thal: Thalassemia (1-3).
target: Diagnosis penyakit jantung (1 = memiliki penyakit jantung, 0 = tidak memiliki penyakit jantung).

### Exploratory Data Analysis (EDA):
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
print(df.head())
print(df.describe())
print(df.info())

# Visualisasi data
sns.countplot(x='target', data=df)
plt.show()

sns.pairplot(df, hue='target')
plt.show()
```

## Data Preparation
### Data Cleaning
Mengatasi missing values dan normalisasi data:
```
# Tidak ada missing values pada dataset ini
df.isnull().sum()

# Normalisasi data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('target', axis=1))

# Memisahkan fitur dan label
X = df_scaled
y = df['target']
```

### Proses Data Preparation:
Normalisasi dilakukan untuk memastikan semua fitur berada dalam skala yang sama.
Teknik ini diperlukan agar algoritma machine learning dapat bekerja lebih efisien dan akurat.

## Modeling
### Model Selection
Logistic Regression sebagai baseline model.
Random Forest sebagai model pembanding.

### Hyperparameter Tuning
Melakukan tuning pada Random Forest untuk meningkatkan performa:
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("Best Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))
```

### Kelebihan dan Kekurangan Algoritma:
* Logistic Regression: Mudah diinterpretasi, cepat, tetapi mungkin kurang akurat untuk data yang kompleks.
* Random Forest: Akurasi tinggi, robust terhadap overfitting, tetapi lebih kompleks dan membutuhkan lebih banyak waktu untuk pelatihan.

## Evaluation
### Metrik Evaluasi
Untuk kasus klasifikasi ini, metrik evaluasi yang digunakan adalah akurasi, precision, recall, dan F1 score.

### Penjelasan Metrik:
* Akurasi: Proporsi prediksi benar dari keseluruhan prediksi.
* Precision: Proporsi prediksi positif yang benar dari keseluruhan prediksi positif.
* Recall: Proporsi prediksi positif yang benar dari keseluruhan data aktual positif.
* F1 Score: Harmonic mean dari precision dan recall.

### Hasil Proyek Berdasarkan Metrik Evaluasi
* Logistic Regression Accuracy: 0.79
* Random Forest Accuracy: 0.98
* Best Random Forest Accuracy setelah tuning: 0.98

### Kesimpulan:
Model Random Forest dengan hyperparameter tuning memberikan akurasi terbaik dalam memprediksi penyakit jantung pada dataset ini. Model ini direkomendasikan untuk digunakan dalam aplikasi klinis untuk membantu deteksi dini penyakit jantung.
