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
* age: Umur pasien.
* sex: Jenis kelamin (1 = laki-laki, 0 = perempuan).
* cp: Tipe nyeri dada (0-3).
* trestbps: Tekanan darah saat istirahat.
* chol: Kolesterol serum dalam mg/dl.
* fbs: Gula darah puasa > 120 mg/dl (1 = benar, 0 = salah).
* restecg: Hasil elektrokardiografi istirahat (0-2).
* thalach: Detak jantung maksimal yang tercapai.
* exang: Angina akibat olahraga (1 = ya, 0 = tidak).
* oldpeak: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.
* slope: Kemiringan segmen ST latihan puncak (0-2).
* ca: Jumlah pembuluh darah utama (0-4) yang diwarnai oleh fluoroskopi.
* thal: Thalassemia (1-3).
* target: Diagnosis penyakit jantung (1 = memiliki penyakit jantung, 0 = tidak memiliki penyakit jantung).

### Exploratory Data Analysis (EDA):
Saya memulai dengan melakukan analisis data eksploratif untuk memahami karakteristik data:
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/Ardizza/Machine-Learning-Terapan/main/heart.csv')
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
Normalisasi dilakukan untuk memastikan semua fitur berada dalam skala yang sama. Teknik ini diperlukan agar algoritma machine learning dapat bekerja lebih efisien dan akurat.

## Modeling
### Model Selection
Pada tahap ini, dua model utama yang dipilih untuk proyek ini adalah Logistic Regression dan Random Forest. Logistic Regression digunakan sebagai model baseline karena kesederhanaannya dan interpretasinya yang mudah. Random Forest dipilih karena kemampuannya dalam menangani dataset yang kompleks dan memberikan hasil yang lebih akurat.

#### Logistic Regression
Logistic Regression adalah metode statistik yang digunakan untuk analisis prediktif ketika hasilnya adalah variabel biner. Model ini cocok digunakan sebagai baseline karena cepat dan mudah diinterpretasi.
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

```
Parameter utama yang digunakan:
* penalty: Regulasi yang digunakan untuk menghindari overfitting.
* c: Inversi dari kekuatan regulasi, dengan nilai yang lebih kecil berarti regulasi yang lebih kuat.
* solver: Algoritma yang digunakan untuk optimisasi.

#### Random Forest
Random Forest adalah algoritma ensemble yang terdiri dari beberapa decision tree. Setiap tree dilatih pada subset data yang berbeda dan hasil akhirnya adalah rata-rata dari hasil setiap tree. Ini membuat Random Forest robust terhadap overfitting dan lebih akurat dibandingkan model individual.
```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

```
Parameter utama yang digunakan:
* n_estimators: Jumlah pohon keputusan dalam model Random Forest.
* max_depth: Kedalaman maksimum pohon individu.
* min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node internal.
* min_samples_leaf: Jumlah minimum sampel yang diperlukan untuk berada di node daun.

#### Hyperparameter Tuning
Untuk meningkatkan performa model Random Forest, dilakukan tuning terhadap beberapa hyperparameter menggunakan GridSearchCV. GridSearchCV membantu menemukan kombinasi terbaik dari hyperparameter dengan melakukan pencarian grid pada ruang parameter yang diberikan.
```
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

```
Parameter yang di-tuning:
* n_estimators: Jumlah pohon keputusan dalam model Random Forest.
* max_depth: Kedalaman maksimum pohon individu.
* min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node internal.

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
