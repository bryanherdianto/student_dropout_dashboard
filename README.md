# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

### Permasalahan Bisnis

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

Apabila masalah ini tidak segera diselesaikan, maka hal-hal berikut dapat saja terjadi:
1. Pengurangan Pendapatan Institusi: Jika banyak siswa melakukan dropout, institusi akan kehilangan pendapatan dari biaya pendaftaran, uang kuliah, dan sumber daya lainnya. Ini dapat mengganggu keberlanjutan operasional institusi.
2. Pengurangan Reputasi: Tingkat dropout yang tinggi dapat mempengaruhi reputasi institusi. Calon siswa dan orang tua mungkin ragu untuk memilih institusi yang memiliki masalah serius dengan tingkat kelulusan.
3. Ketidakstabilan Akademik: Siswa yang dropout mungkin mengalami ketidakstabilan akademik dan emosional. Mereka kehilangan kesempatan untuk belajar dan mengembangkan diri, yang dapat memengaruhi masa depan mereka.
4. Ketidaksetaraan Peluang: Dropout cenderung mempengaruhi kelompok-kelompok tertentu, seperti siswa dari latar belakang ekonomi rendah atau minoritas. Ini dapat memperburuk kesenjangan pendidikan dan sosial.

Oleh karena itu, penting bagi Jaya Jaya Institut untuk mengambil tindakan segera untuk mengurangi tingkat dropout dan memberikan bimbingan lebih baik kepada murid-murid yang membutuhkan.

### Cakupan Proyek

Untuk mengatasi masalah dropout, akan dilakukan upaya mengembangkan model machine learning menggunakan random forest classifier untuk mengidentifikasi faktor-faktor yang berkontribusi terhadap dropout dan model juga mampu memprediksi outcome berdasarkan feature-feature tertentu. Selanjutnya, akan dibuat dashboard visualisasi data dan laporan analisis data yang mendalam. Dashboard akan menggunakan *Metadata* untuk menunjukkan hasil analisis dari model machine learning. Model machine learning juga akan ditampilkan melalui *Streamlit* sehingga bisa dipakai oleh semua orang untuk mengetes performa dari model machine learning.

Sebelum membuat model, pentingnya untuk memahami dataset yang telah diberikan. Studi mengenai perilaku dropout dapat dilakukan dengan menganalisis berbagai faktor yang terkait dengan latar belakang dan performa akademik mereka. Beberapa kolom yang relevan dalam studi ini meliputi status perkawinan, mode aplikasi, urutan aplikasi, kursus, kehadiran di siang atau malam hari, kualifikasi sebelumnya, dan nilai kualifikasi sebelumnya. Selain itu, data mengenai kewarganegaraan, kualifikasi ibu dan ayah, pekerjaan ibu dan ayah, serta nilai penerimaan juga penting untuk dipertimbangkan. Faktor-faktor lainnya yang perlu dianalisis adalah apakah siswa tersebut terdampak pengungsian, memiliki kebutuhan pendidikan khusus, status sebagai debitur, apakah biaya kuliah mereka telah terbayar, jenis kelamin, status beasiswa, usia saat pendaftaran, serta apakah mereka adalah siswa internasional. Performa akademik juga dilihat dari jumlah kredit, jumlah mata kuliah yang diambil, jumlah evaluasi, jumlah mata kuliah yang disetujui, dan nilai rata-rata pada semester pertama dan kedua. Dengan menganalisis data ini, kita dapat memahami lebih baik faktor-faktor yang mempengaruhi siswa untuk putus sekolah dan mengambil langkah-langkah yang tepat untuk mencegahnya.

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

Setup Environment - Anaconda:
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

Setup Environment - Shell/Terminal:
```
mkdir student-performance-dashboard
cd student-performance-dashboard
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Business Dashboard

Dalam *business dashboard* yang telah dibuat, dilakukan upaya untuk mengeksplorasi feature-feature yang dimiliki oleh murid-murid. Hal ini mencakup usia, pekerjaan dari ayah, pekerjaan dari ibu, jenis kelamin, asal negara, dll. Setelah melakukan visualisasi dari feature-feature tersebut, dilakukan upaya untuk membuat sebuah model machine learning yang bisa memprediksi status seorang pelajar berdasarkan feature tersebut. Model tersebut berupa random forest classifier guna memprediksi. Untuk melihat performa dari model, dibuat sebuah confusion matrix sehingga bisa mencari tahu presisi dan akurasi.

Link 1: http://localhost:3000/public/dashboard/39a70b45-1755-4d78-aabe-0669dbb97f87
Link 2: https://drive.google.com/file/d/1seW6H69SOzETXibfoT5Y1WPGsWH7525X/view?usp=sharing

## Menjalankan Sistem Machine Learning
Sistem pembelajaran mesin ini menggunakan model Random Forest Classifier, yang merupakan metode ensemble learning untuk klasifikasi, regresi, dan tugas lainnya yang menggabungkan sejumlah pohon keputusan individu untuk meningkatkan akurasi dan mengurangi overfitting. Model ini dikembangkan menggunakan pustaka PySpark dan diunggah ke Streamlit. Model tersebut telah dilatih dengan dataset yang disediakan. Dengan demikian, model ini mampu memprediksi status seorang siswa, apakah dia akan putus studi (dropout), masih terdaftar (enrolled), atau lulus (graduated) berdasarkan karakteristik tertentu. Model machine learning telah berhasil dideploy dan dapat diakses melalui link berikut atau dengan download file dan dijalankan sendiri.

Link: https://studentdropoutdashboard-skgeqtxnzf3uahvvsjkhu9.streamlit.app/

### Run streamlit app
```
streamlit run app.py
```

## Conclusion

Dalam membuat model machine learning, model yang digunakan adalah Random Forest Classifier. Dengan model ini, dapat diidentifikasi berbagai feature yang berkontribusi terhadap label status pada pelajar. Label disini adalah dropout, graduated, dan enrolled. Setelah berhasil train model, model dapat dideploy dengan streamlit sehingga dapat diakses secara online.

### Rekomendasi Action Items

Berikut beberapa rekomendasi action items yang harus dilakukan institusi guna menyelesaikan permasalahan atau mencapai target mereka.

- Mengurangi jumlah pelajar yang memiliki utang
- Memberikan keringanan pembayaran bagi mereka yang membutuhkan
