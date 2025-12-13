# Dokumentasi Proyek Robot Lengan 3-Sendi dengan PPO (Reinforcement Learning)

Dokumen ini menjelaskan struktur dan cara kerja dari simulasi robot lengan yang dilatih menggunakan algoritma **Proximal Policy Optimization (PPO)** untuk tugas memindahkan kotak ke target.

## 📁 Struktur File & Komponen

### 1. World File (`worlds/NEW.wbt`)
Ini adalah lingkungan simulasi utama di Webots.
- **Arena**: Lantai datar dengan tekstur papan catur dan langit biru.
- **Robot**: Menggunakan referensi ke `ThreeJointArm.proto`. Ditempatkan di tengah (0, 0, 0).
- **Objek Kotak (`DEF box`)**: Kotak merah yang harus dipindahkan oleh robot. Memiliki fisika (berat/massa).
- **Target (`DEF target`)**: Area persegi hijau di lantai yang menandakan tujuan akhir kotak. Ini bersifat visual (sensorik) untuk robot.

### 2. PROTO File (`protos/ThreeJointArm.proto`)
Ini adalah definisi blueprint dari robot lengan itu sendiri.
- **Struktur**: Robot 3-DOF (Degree of Freedom) dengan 3 sendi putar (Revolute Joints):
  1.  **Waist (Joint1)**: Putaran horizontal dasar.
  2.  **Shoulder (Joint2)**: Anggukan bahu vertikal.
  3.  **Elbow (Joint3)**: Anggukan siku vertikal.
- **Sensor**:
  - `PositionSensor` pada setiap sendi untuk mengetahui sudut kemiringan.
  - **GPS** pada ujung lengan (End-Effector) untuk mengetahui posisi tangan robot di ruang 3D (X, Y, Z).
- **Fisika**:
  - Base robot **terkunci mati (`locked TRUE`)** dan tidak memiliki node `Physics` di root, memastikan robot berdiri kokoh dan tidak goyang/jatuh.
  - Komponen lengan (link) memiliki massa dan fisika agar bisa berinteraksi (mendorong) kotak.

### 3. Controller (`controllers/ppo_controller/ppo_controller.py`)
Ini adalah otak dari robot yang berisi implementasi algoritma PPO menggunakan PyTorch.

---

## 🧠 Penjelasan Reinforcement Learning (RL)

Agar robot dapat belajar, kita mendefinisikan 3 komponen utama RL: **State (Keadaan)**, **Action (Aksi)**, dan **Reward (Hadiah)**.

### 1. State Space (Ruang Keadaan - 15 Dimensi)
Input yang diberikan ke otak robot (Neural Network) setiap langkah untuk "melihat" dunianya.

| Indeks | Deskripsi | Penjelasan |
| :--- | :--- | :--- |
| **0-2** | `Sin(Joint Angles)` | Nilai Sinus dari sudut ketiga sendi. |
| **3-5** | `Cos(Joint Angles)` | Nilai Cosinus dari sudut ketiga sendi. Menggunakan Sin/Cos lebih baik daripada sudut mentah (derajat) agar NN mengerti kontinuitas putaran. |
| **6-8** | `Box Position` | Posisi X, Y, Z dari kotak merah di dunia. |
| **9-11** | `End-Effector POS` | Posisi X, Y, Z dari tangan robot (dari sensor GPS). Ini agar robot "sadar" posisi tangannya sendiri. |
| **12-14** | `Target Vector` | Vektor jarak dari Kotak ke Target (`Target - Box`). Memberi tahu robot ke arah mana kotak harus didorong. |

### 2. Action Space (Ruang Aksi - 3 Dimensi)
Output yang dihasilkan oleh otak robot untuk menggerakkan tubuhnya.

- Terdiri dari 3 nilai bilangan riil (float) antara **-1.0 sampai 1.0**.
- Setiap nilai merepresentasikan **Target Kecepatan (Velocity)** untuk masing-masing motor (Joint1, Joint2, Joint3).
- **Mekanisme**: Output dikalikan dengan `max_speed` (misal 3.0 rad/s).
  - `-1.0`: Putar maksimal ke kiri/bawah.
  - `0.0`: Diam.
  - `1.0`: Putar maksimal ke kanan/atas.

### 3. Reward Function (Fungsi Hadiah)
Cara kita memberi tahu robot apakah ia melakukan hal yang benar atau salah. Kita menggunakan **Dense Distance Reward** (Hadiah Jarak Padat) agar robot cepat belajar.

Rumus Dasar:
```python
Reward = -(Jarak_Tangan_ke_Kotak * 2.0) - (Jarak_Kotak_ke_Target * 4.0)
```

**Penjelasan:**
1.  **Semakin Dekat, Semakin Baik**: Karena nilai jarak adalah positif (misal 2 meter), kita memberinya tanda negatif (`-2`). Jika robot mendekat (jarak jadi 0.1 meter), nilainya menjadi `-0.1` (lebih besar dari -2). Robot selalu ingin memaksimalkan nilai ini mendekati 0.
2.  **Prioritas Target**: Jarak Kotak-ke-Target dikalikan 4, sedangkan Tangan-ke-Kotak dikalikan 2. Ini berarti robot lebih "tergoda" untuk mendorong kotak ke target daripada sekadar menyentuh kotak.
3.  **Bonus Sukses**: Jika kotak sampai di target (< 0.2m), robot dapat **+200 poin**.
4.  **Hukuman Jatuh**: Jika kotak jatuh dari arena, robot dapat **-50 poin**.

---

## 🚀 Cara Kerja Training Loop

1.  **Reset**: Setiap episode, posisi robot dan kotak direset.
2.  **Exploration**: Di awal, robot akan bergerak acak (karena noise/entropy tinggi) untuk mencari tahu cara menggerakkan badannya.
3.  **Collection**: Data pengalaman (State, Action, Reward) disimpan dalam memori buffer.
4.  **Update (PPO)**: Setiap 2000 langkah (timestep), robot akan "berpikir" (training Neural Network) menggunakan data di buffer untuk memperbaiki kebijakannya.
5.  **Iterasi**: Proses ini berulang ribuan kali hingga robot semakin pintar dan nilai Reward rata-rata meningkat (dari sangat negatif menjadi mendekati positif).
