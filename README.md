# UTS Data Mining — Klasifikasi Kualitas Anggur (Wine Quality Classification)

**Nama**: Elsabila Nurbaity  
**NIM**: 2304020021  
**Rombel**: 1 PMAT 23  
**Mata Kuliah**: Data Mining

---

## Deskripsi Proyek

Repository ini berisi hasil pengerjaan UTS Data Mining dengan topik klasifikasi kualitas anggur menggunakan dataset Wine Quality. Tujuan utama adalah membangun model machine learning yang mampu memprediksi nilai kualitas anggur (`quality`, skala 0–10) berdasarkan 11 fitur kimiawi yang tersedia, kemudian menerapkan model tersebut untuk memprediksi kualitas pada dataset testing yang tidak memiliki label.

---

## Pendahuluan
Anggur merupakan salah satu minuman fermentasi yang kualitasnya sangat dipengaruhi oleh komposisi kimiawi yang terkandung di dalamnya. Penilaian kualitas anggur secara konvensional dilakukan oleh para ahli (sommelier) melalui uji organoleptik, yaitu penilaian berdasarkan rasa, aroma, dan penampilan. Metode ini bersifat subjektif dan membutuhkan keahlian khusus serta waktu yang tidak sedikit.

Seiring berkembangnya ilmu data, pendekatan berbasis machine learning menawarkan cara yang lebih objektif dan efisien untuk memprediksi kualitas anggur. Dengan memanfaatkan data kimiawi seperti kadar alkohol, keasaman, dan kandungan sulfat, sebuah model klasifikasi dapat dilatih untuk mengenali pola yang membedakan anggur berkualitas tinggi dari yang berkualitas rendah.

Proyek ini merupakan bagian dari UTS mata kuliah Data Mining. Pada proyek ini dibangun model klasifikasi menggunakan algoritma Random Forest untuk memprediksi nilai kualitas anggur berdasarkan 11 fitur kimiawi. Dataset yang digunakan merupakan gabungan data anggur merah dan putih dari berbagai sampel dengan label kualitas yang diberikan oleh para ahli wine tasting.


## Struktur Repository

```
├── UTS_DataMining_WineQuality_021.ipynb    # Notebook analisis lengkap dengan interpretasi
├── hasilprediksi_021.csv                   # File hasil prediksi (Id + quality)
└── README.md                               # Dokumentasi proyek ini
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

Random Forest adalah metode ensemble yang bekerja dengan membangun sejumlah pohon keputusan (decision tree) secara paralel pada subset data dan subset fitur yang dipilih secara acak (bootstrap sampling). Prediksi final ditentukan melalui voting mayoritas dari seluruh pohon.

Alasan pemilihan Random Forest:
- Mampu menangani permasalahan klasifikasi multi-kelas dengan baik
- Lebih tahan terhadap overfitting dibandingkan single decision tree
- Tidak memerlukan asumsi distribusi data tertentu
- Memberikan informasi feature importance yang dapat diinterpretasikan
- Robust terhadap outlier dan noise dalam data

Konfigurasi hyperparameter yang digunakan:

| Hyperparameter | Nilai | Alasan |
|---|---|---|
| n_estimators | 300 | Jumlah pohon yang cukup besar untuk stabilitas prediksi |
| max_depth | None | Pohon tumbuh penuh tanpa batasan kedalaman |
| min_samples_split | 2 | Minimum 2 sampel untuk melakukan split pada node |
| random_state | 42 | Seed tetap untuk memastikan reproduktivitas hasil |
| n_jobs | -1 | Menggunakan seluruh core CPU yang tersedia |

### Langkah 6 — Evaluasi Model

Evaluasi dilakukan dengan dua cara:

**Akurasi pada data training**: 100%  
Model mampu mengklasifikasikan seluruh data training dengan sempurna. Nilai ini wajar karena Random Forest dilatih pada data yang sama.

**5-Fold Stratified Cross-Validation**:

Untuk mendapatkan estimasi performa yang lebih realistis, digunakan Stratified K-Fold Cross-Validation dengan 5 fold. Stratified dipilih agar setiap fold mempertahankan proporsi kelas yang sama dengan distribusi keseluruhan.

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

Tiga fitur paling berpengaruh adalah `alcohol` (0.1424), `sulphates` (0.1165), dan `volatile acidity` (0.1122). Distribusi importance yang relatif merata antar fitur menunjukkan bahwa tidak ada satu fitur pun yang sangat mendominasi, dan model memanfaatkan kombinasi informasi dari semua fitur untuk membuat prediksi.

### Langkah 7 — Prediksi Data Testing

Setelah model dievaluasi dan dinilai memadai, model diterapkan pada 286 data testing menggunakan fitur yang sudah dinormalisasi dengan parameter scaler dari data training.

Distribusi hasil prediksi:
- Quality 5: 132 data (46.2%)
- Quality 6: 125 data (43.7%)
- Quality 7: 29 data (10.1%)

Distribusi ini konsisten dengan pola pada data training.

### Langkah 8 — Penyimpanan Hasil

Hasil prediksi disimpan dalam file `hasilprediksi_021.csv` yang hanya memuat dua kolom: `Id` dan `quality`.

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

---

## Analisis dan Pembahasan

### Performa Model

Model Random Forest yang dibangun menunjukkan akurasi cross-validation sebesar 64.42% pada data training. Nilai ini perlu diinterpretasikan dalam konteks permasalahannya. Dataset Wine Quality adalah permasalahan klasifikasi multi-kelas dengan 6 kelas berbeda (quality 3 hingga 8), di mana distribusi kelasnya sangat tidak merata. Kelas mayoritas (quality 5 dan 6) menyumbang sekitar 82% dari total data, sementara kelas minoritas (quality 3 dan 8) masing-masing hanya memiliki 6 dan 13 sampel.

Sebagai baseline perbandingan, jika model hanya menebak kelas yang paling sering muncul (quality 5) untuk semua prediksi, akurasi yang diperoleh hanya sekitar 42.2%. Dengan demikian, akurasi 64.42% menunjukkan bahwa model berhasil mempelajari pola yang bermakna dari data, jauh di atas baseline tersebut.

### Keterbatasan Model

Ada beberapa keterbatasan yang perlu diperhatikan dari model yang dibangun:

Pertama, model tidak menghasilkan prediksi untuk kelas 3, 4, dan 8 pada data testing. Hal ini disebabkan oleh jumlah sampel yang sangat sedikit untuk kelas-kelas tersebut pada data training, sehingga model belum memiliki cukup contoh untuk mengenali pola kelas minoritas secara andal.

Kedua, akurasi training yang mencapai 100% menunjukkan bahwa model mengalami overfitting terhadap data training. Perbedaan yang cukup besar antara akurasi training (100%) dan cross-validation (64.42%) mengindikasikan bahwa model sangat menghafal data training dan belum sepenuhnya mampu menggeneralisasi.

Ketiga, sifat subjektif dari label kualitas anggur yang diberikan oleh penilai manusia turut berkontribusi pada noise dalam data, sehingga batasan antar kelas tidak selalu jelas dan konsisten.

### Potensi Peningkatan

Beberapa pendekatan yang berpotensi meningkatkan performa model antara lain menerapkan teknik oversampling seperti SMOTE untuk menangani class imbalance, melakukan hyperparameter tuning menggunakan GridSearchCV atau RandomizedSearchCV, mencoba algoritma lain seperti XGBoost atau LightGBM, serta mempertimbangkan penggabungan kelas yang berdekatan (misalnya quality 3 dan 4 digabung menjadi satu kelas "rendah") untuk menyederhanakan permasalahan klasifikasi.

---

## Kesimpulan

Berdasarkan seluruh tahapan yang telah dilakukan, dapat ditarik beberapa kesimpulan sebagai berikut:

1. Dataset Wine Quality dalam kondisi bersih tanpa missing values maupun duplikasi, sehingga tidak diperlukan penanganan khusus pada tahap pembersihan data. Namun dataset memiliki ketidakseimbangan kelas yang cukup signifikan antara kelas mayoritas dan minoritas.

2. Model Random Forest Classifier berhasil dibangun dengan konfigurasi 300 estimator dan normalisasi StandardScaler. Model mencapai rata-rata cross-validation accuracy sebesar 64.42% dengan standar deviasi 0.0377, yang menunjukkan performa yang cukup konsisten dan berada jauh di atas baseline acak.

3. Dari analisis feature importance, kadar alkohol (`alcohol`) terbukti menjadi faktor paling berpengaruh dalam menentukan kualitas anggur, diikuti oleh kandungan sulfat (`sulphates`) dan keasaman volatil (`volatile acidity`). Temuan ini konsisten dengan literatur di bidang enologi yang menyatakan bahwa ketiga komponen tersebut memang berperan penting dalam menentukan profil rasa dan kualitas anggur secara keseluruhan.

4. Model berhasil menghasilkan prediksi untuk seluruh 286 data testing dengan distribusi yang konsisten terhadap pola distribusi data training, yaitu didominasi oleh kelas 5 (46.2%), diikuti kelas 6 (43.7%), dan kelas 7 (10.1%).

5. Pendekatan machine learning terbukti mampu menangkap pola dari data kimiawi anggur untuk melakukan klasifikasi kualitas secara otomatis, meski masih terdapat ruang yang cukup besar untuk peningkatan, terutama dalam menangani kelas-kelas minoritas.

---

## Referensi

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547–553.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Dataset Wine Quality: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

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
