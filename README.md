# UTS Data Mining — Klasifikasi Kualitas Anggur (Wine Quality Classification)

**Nama**: Elsabila Nurbaity  
**NIM**: 2304020021  
**Rombel**: 1 PMAT 23  
**Mata Kuliah**: Data Mining

---

## Deskripsi Proyek

Repository ini berisi hasil pengerjaan UTS Data Mining dengan topik klasifikasi kualitas anggur menggunakan dataset Wine Quality. Tujuan utama adalah membangun model machine learning yang mampu memprediksi nilai kualitas anggur (`quality`, skala 0–10) berdasarkan 11 fitur kimiawi yang tersedia, kemudian menerapkan model tersebut untuk memprediksi kualitas pada dataset testing yang tidak memiliki label.

---

## Struktur Repository

```
├── UTS_DataMining_WineQuality_021.ipynb    # Notebook analisis lengkap dengan interpretasi
├── hasilprediksi_021.csv               # File hasil prediksi (Id + quality)
└── README.md                           # Dokumentasi proyek ini
```

---

## Dataset

Dataset yang digunakan adalah Wine Quality dataset yang memuat data anggur merah dan putih beserta fitur-fitur kimiawinya.

| Dataset | Jumlah Baris | Jumlah Kolom | Keterangan |
|---|---|---|---|
| data_training.csv | 857 | 13 | Memiliki label quality (target) |
| data_testing.csv | 286 | 12 | Tidak memiliki label quality |

Fitur-fitur yang digunakan sebagai input model:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

Variabel target: `quality` (nilai integer, rentang 3–8 pada dataset ini)

---

## Alur Pengerjaan

### Langkah 1 — Persiapan dan Pemuatan Data

Dataset training dan testing dimuat menggunakan pandas. Dataset training berisi 857 sampel dengan 11 fitur kimiawi dan satu kolom target `quality`. Dataset testing berisi 286 sampel tanpa kolom target.

Distribusi kelas pada data training:
- Quality 3: 6 data (0.7%)
- Quality 4: 26 data (3.0%)
- Quality 5: 362 data (42.2%)
- Quality 6: 341 data (39.8%)
- Quality 7: 109 data (12.7%)
- Quality 8: 13 data (1.5%)

Terlihat jelas bahwa terdapat class imbalance di mana kelas 5 dan 6 mendominasi, sementara kelas 3 dan 8 sangat sedikit.

### Langkah 2 — Pembersihan Data

Dilakukan pemeriksaan terhadap:
- **Missing values**: Tidak ditemukan nilai yang hilang pada dataset training maupun testing (total = 0). Tidak diperlukan imputasi.
- **Duplikasi**: Tidak terdapat baris duplikat pada dataset training.

Data dapat langsung digunakan untuk proses selanjutnya tanpa perlu pembersihan tambahan.

### Langkah 3 — Eksplorasi Data (EDA)

Beberapa temuan utama dari eksplorasi data:
- Fitur `alcohol` memiliki korelasi positif tertinggi dengan `quality` (r = 0.48), artinya anggur dengan kadar alkohol lebih tinggi cenderung berkualitas lebih baik.
- Fitur `volatile acidity` memiliki korelasi negatif terkuat dengan `quality` (r = -0.39), artinya keasaman volatil yang tinggi berkaitan dengan kualitas yang lebih rendah.
- Beberapa fitur seperti `residual sugar` dan `free sulfur dioxide` memiliki distribusi yang skewed ke kanan, menunjukkan adanya nilai-nilai ekstrem (outlier).

### Langkah 4 — Preprocessing Data

Dilakukan normalisasi menggunakan **StandardScaler** pada seluruh fitur. Normalisasi penting dilakukan karena fitur-fitur memiliki rentang nilai yang sangat berbeda — misalnya `total sulfur dioxide` bisa mencapai ratusan, sementara `chlorides` bernilai di bawah 1.

Proses scaling dilakukan dengan:
- `fit_transform` pada data training untuk menghitung mean dan standar deviasi
- `transform` pada data testing menggunakan parameter dari data training (untuk menghindari data leakage)

Setelah scaling, semua fitur memiliki mean mendekati 0 dan standar deviasi mendekati 1.

### Langkah 5 — Pembuatan Model

Model yang dipilih adalah **Random Forest Classifier** dengan konfigurasi:
- `n_estimators = 300`: Dibangun 300 pohon keputusan
- `max_depth = None`: Pohon tumbuh penuh tanpa batas kedalaman
- `min_samples_split = 2`: Minimum 2 sampel untuk pemisahan node
- `random_state = 42`: Seed untuk reproduktivitas

Random Forest dipilih karena kemampuannya menangani permasalahan klasifikasi multi-kelas, ketahanannya terhadap overfitting dibanding single decision tree, dan kemampuannya memberikan informasi feature importance.

### Langkah 6 — Evaluasi Model

Evaluasi dilakukan dengan dua cara:

**Akurasi pada data training**: 100%  
Model mampu mengklasifikasikan seluruh data training dengan sempurna. Nilai ini wajar karena Random Forest dilatih pada data yang sama.

**5-Fold Stratified Cross-Validation**:
| Fold | Akurasi |
|---|---|
| Fold 1 | 57.56% |
| Fold 2 | 64.53% |
| Fold 3 | 64.33% |
| Fold 4 | 67.25% |
| Fold 5 | 68.42% |
| **Rata-rata** | **64.42%** |
| **Std Deviasi** | **0.0377** |

Rata-rata cross-validation accuracy sebesar 64.42% merupakan estimasi performa yang lebih realistis terhadap data baru. Standar deviasi 0.0377 menunjukkan bahwa performa model relatif konsisten di setiap fold.

**Feature Importance**:
| Fitur | Importance Score |
|---|---|
| alcohol | 0.1424 |
| sulphates | 0.1165 |
| volatile acidity | 0.1122 |
| total sulfur dioxide | 0.0978 |
| density | 0.0879 |
| chlorides | 0.0816 |
| citric acid | 0.0772 |
| fixed acidity | 0.0753 |
| pH | 0.0748 |
| residual sugar | 0.0672 |
| free sulfur dioxide | 0.0670 |

Tiga fitur paling berpengaruh adalah `alcohol`, `sulphates`, dan `volatile acidity`.

### Langkah 7 — Prediksi Data Testing

Model diterapkan pada 286 data testing. Distribusi hasil prediksi:
- Quality 5: 132 data (46.2%)
- Quality 6: 125 data (43.7%)
- Quality 7: 29 data (10.1%)

Distribusi ini konsisten dengan pola pada data training.

### Langkah 8 — Penyimpanan Hasil

Hasil prediksi disimpan dalam file `hasilprediksi_3digitNIMterakhir.csv` yang hanya memuat dua kolom: `Id` dan `quality`.

---

## Hasil Ringkasan

| Aspek | Hasil |
|---|---|
| Algoritma | Random Forest Classifier |
| Jumlah Estimator | 300 |
| Metode Normalisasi | StandardScaler |
| Akurasi Training | 100% |
| CV Accuracy (5-Fold) | 64.42% |
| Std Deviasi CV | 0.0377 |
| Fitur Terpenting | alcohol, sulphates, volatile acidity |
| Jumlah Data Testing | 286 |

---

## Cara Menjalankan Notebook

1. Buka Google Colab di [colab.research.google.com](https://colab.research.google.com)
2. Upload file `UTS_DataMining_WineQuality_021.ipynb`
3. Jalankan cell pertama untuk mengimpor library
4. Upload `data_training.csv` dan `data_testing.csv` ketika diminta
5. Jalankan seluruh cell secara berurutan
6. File CSV hasil prediksi akan otomatis terunduh setelah cell terakhir dijalankan

---

## Dependensi

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Semua library di atas tersedia secara default di Google Colab dan tidak memerlukan instalasi tambahan.
