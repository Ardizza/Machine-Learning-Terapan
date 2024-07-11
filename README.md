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

### Kondisi Dataset:
* Nilai Null: Dataset ini tidak memiliki nilai null, sehingga tidak diperlukan penanganan missing values.
* Sebaran Data: Dataset ini terdiri dari 14 atribut yang memiliki sebaran nilai yang bervariasi, seperti umur, tekanan darah, kolesterol, dan lain-lain.
* Distribusi Kelas: Distribusi kelas pada variabel target cukup seimbang antara pasien yang memiliki penyakit jantung dan yang tidak, sehingga tidak diperlukan penanganan untuk masalah class imbalance.

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
1. Informasi Dataset
Menampilkan informasi mengenai tipe data dari masing-masing kolom serta jumlah nilai non-null di setiap kolom.

2. Statistik Deskriptif
Menampilkan statistik deskriptif dari dataset seperti mean, standar deviasi, nilai minimum dan maksimum, serta kuartil.

3. Visualisasi Data
* Distribusi Variabel Targe
* Pairplot
* Heatmap Korelasi Fitur

## Data Preparation
### Data Cleaning
1. Data Cleaning
Langkah ini memastikan bahwa dataset bersih dan siap untuk digunakan dalam analisis. Dalam kasus ini, tidak ada nilai yang hilang (missing values) pada dataset, sehingga tidak diperlukan penanganan lebih lanjut untuk nilai yang hilang.

2. Normalisasi Data
Normalisasi adalah proses penskalaan fitur-fitur sehingga mereka berada dalam skala yang sama. Ini penting karena banyak algoritma pembelajaran mesin bekerja lebih baik ketika fitur-fitur memiliki rentang nilai yang serupa.

3. Memisahkan Fitur dan Label
Memisahkan fitur dan label adalah langkah di mana kita memisahkan atribut input (fitur) dari target output (label) yang ingin diprediksi oleh model.

5. Split Data Menjadi Training dan Testing Set
Membagi data menjadi training dan testing set adalah langkah untuk memastikan model dapat dievaluasi secara objektif. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk menguji performa model pada data yang belum pernah dilihat sebelumnya.

## Modeling
### Proses dan Tahapan:
1. Logistic Regression
Logistic Regression adalah metode statistik yang digunakan untuk analisis prediktif ketika hasilnya adalah variabel biner. Model ini cocok digunakan sebagai baseline karena cepat dan mudah diinterpretasi.

* Parameter Utama:
** penalty: Regulasi yang digunakan untuk menghindari overfitting.
** c: Inversi dari kekuatan regulasi, dengan nilai yang lebih kecil berarti regulasi yang lebih kuat.
** solver: Algoritma yang digunakan untuk optimisasi.

2. Random Forest
Random Forest adalah algoritma ensemble yang terdiri dari beberapa decision tree. Setiap tree dilatih pada subset data yang berbeda dan hasil akhirnya adalah rata-rata dari hasil setiap tree. Ini membuat Random Forest robust terhadap overfitting dan lebih akurat dibandingkan model individual.

Parameter Utama:
n_estimators: Jumlah pohon keputusan dalam model Random Forest.
max_depth: Kedalaman maksimum pohon individu.
min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node internal.
min_samples_leaf: Jumlah minimum sampel yang diperlukan untuk berada di node daun.

3. Hyperparameter Tuning
Untuk meningkatkan performa model Random Forest, dilakukan tuning terhadap beberapa hyperparameter menggunakan GridSearchCV. GridSearchCV membantu menemukan kombinasi terbaik dari hyperparameter dengan melakukan pencarian grid pada ruang parameter yang diberikan.

Parameter Terbaik:
* n_estimators: 200
* max_depth: 10
* min_samples_split: 2

### Kelebihan dan Kekurangan Algoritma:
* Logistic Regression: Mudah diinterpretasi, cepat, tetapi mungkin kurang akurat untuk data yang kompleks.
* Random Forest: Akurasi tinggi, robust terhadap overfitting, tetapi lebih kompleks dan membutuhkan lebih banyak waktu untuk pelatihan.

## Evaluation
### Metrik Evaluasi
Untuk kasus klasifikasi ini, metrik evaluasi yang digunakan adalah akurasi, precision, recall, dan F1 score.

### Hasil Proyek Berdasarkan Metrik Evaluasi
* Logistic Regression Accuracy: 0.79
* Random Forest Accuracy: 0.98
* Best Random Forest Accuracy setelah tuning: 0.98

### Kesimpulan:
Model Random Forest dengan hyperparameter tuning memberikan akurasi terbaik dalam memprediksi penyakit jantung pada dataset ini. Model ini direkomendasikan untuk digunakan dalam aplikasi klinis untuk membantu deteksi dini penyakit jantung.
