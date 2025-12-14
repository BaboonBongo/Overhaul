# Dokumentasi Proyek Robot Lengan 3-Sendi dengan PPO (Reinforcement Learning)

Dokumentasi struktur dan cara kerja dari simulasi robot lengan yang dilatih menggunakan algoritma **Proximal Policy Optimization (PPO)** untuk tugas memindahkan kotak ke target.

## 📁 Struktur File & Komponen

### 1. World File (`worlds/NEW.wbt`)
Lingkungan simulasi utama di Webots.
- **Arena**: Lantai datar dengan tekstur papan catur dan langit biru.
- **Robot**: Menggunakan referensi ke `ThreeJointArm.proto`. Ditempatkan di tengah (0, 0, 0).
- **Objek Kotak (`DEF box`)**: Kotak merah yang harus dipindahkan oleh robot. Memiliki fisika (berat/massa) dan kecepatan linear yang dapat dilacak.
- **Target (`DEF target`)**: Area persegi hijau di lantai yang menandakan tujuan akhir kotak.

### 2. PROTO File (`protos/ThreeJointArm.proto`)
Definisi blueprint dari robot lengan itu sendiri.
- **Struktur**: Robot 3-DOF (Degree of Freedom) dengan 3 sendi putar (Revolute Joints):
  1.  **Waist (Joint1)**: Putaran horizontal dasar.
  2.  **Shoulder (Joint2)**: Anggukan bahu vertikal.
  3.  **Elbow (Joint3)**: Anggukan siku vertikal.
- **Sensor**:
  - `PositionSensor` pada setiap sendi untuk mengetahui sudut kemiringan.
  - **GPS** pada ujung lengan (End-Effector) untuk mengetahui posisi tangan robot di ruang 3D (X, Y, Z).
- **Fisika**:
  - Base robot **terkunci mati (`locked TRUE`)**.
  - Menggunakan kontrol berbasis Kecepatan (Velocity Control).

### 3. Controller (`controllers/ppo_controller/ppo_controller.py`)
Otak utama robot yang mengimplementasikan algoritma PPO dengan PyTorch. Kode ini telah dioptimalkan untuk mengatasi masalah "stagnasi" dalam pembelajaran.

---

## 🧠 Penjelasan Reinforcement Learning (RL)

Sistem State dan Reward telah diperkaya agar robot lebih cepat belajar memahami fisika dan momentum objek.

### 1. State Space (Ruang Keadaan - 18 Dimensi)
Input yang diberikan ke otak robot (Neural Network) telah diperkaya dengan informasi kecepatan dan vektor relatif agar robot lebih paham "konteks" ruang.

| Indeks | Nama Fitur | Penjelasan |
| :--- | :--- | :--- |
| **0-2** | `Sin(Joint)` | Representasi trigonometri sudut sendi (menghindari diskontinuitas sudut). |
| **3-5** | `Cos(Joint)` | Representasi trigonometri sudut sendi. |
| **6-8** | **Vektor EE ke Box** | Jarak relatif `(Box_Pos - Hand_Pos)`. Memberi tahu robot "di mana kotak itu relatif terhadap tangan saya". |
| **9-11** | **Vektor Box ke Target** | Jarak relatif `(Target_Pos - Box_Pos)`. Memberi tahu robot "ke arah mana kotak harus didorong". |
| **12-14** | `End-Effector POS` | Posisi absolut tangan robot (GPS). Penting agar robot tahu batas jangkauan fisiknya. |
| **15-17** | **Kecepatan Linear Box** | `Box Velocity (Vx, Vy, Vz)`. **Sangat Penting**: Agar robot paham momentum (misal: "Jika saya pukul keras, dia meluncur"). |

### 2. Action Space (Ruang Aksi - 3 Dimensi)
Output kebijakan untuk menggerakkan motor.
- **Tipe**: Kontinu (Continuous), Nilai -1.0 s/d 1.0.
- **Mapping**: Dikalikan dengan `max_speed` (Saat ini diset ke **2.0 rad/s** agar pergerakan lebih terkontrol dan tidak "melempar" kotak terlalu jauh).
- **Mekanisme**: Robot mengontrol **Kecepatan Rotasi** setiap sendi, bukan posisi.

### 3. Reward Function (Fungsi Hadiah Berbasis Peningkatan)
Sistem reward telah diubah dari *Absolute Distance* menjadi **Delta/Improvement Reward**. Robot tidak dinilai berdasarkan "seberapa jauh dia sekarang", tapi "apakah dia bergerak mendekat atau menjauh dibanding langkah sebelumnya?".

Rumus Konsep:
`Reward = (Jarak_Lama - Jarak_Baru) * Bobot`

Komponen Reward:
1.  **Reaching Reward (Bobot: 150.0)**: Diberikan positif jika tangan mendekati kotak.
2.  **Pushing Reward (Bobot: 250.0)**: Diberikan positif jika kotak bergerak mendekati target. Bobot lebih besar karena ini tujuan utama.
3.  **Touch Bonus (+0.1)**: Bonus kecil konstan jika tangan berada sangat dekat dengan kotak (< 15cm) untuk menjaga kontak ("menempel").
4.  **Time Penalty (-0.01)**: Hukuman sangat kecil setiap langkah agar robot tidak malas, tapi cukup santai agar robot tidak panik/terburu-buru.
5.  **Success Bonus (+100.0)**: Hadiah besar jika kotak mencapai target (< 20cm).
6.  **Fail Penalty (-10.0)**: Hukuman jika kotak jatuh dari meja. Dikurangi agar robot tidak takut bereksperimen.

---

## ⚙️ Hyperparameter Training

Konfigurasi parameter pelatihan yang digunakan untuk mencapai konvergensi dalam 1000 episode:

| Parameter | Nilai | Penjelasan |
| :--- | :--- | :--- |
| `max_episodes` | **1000** | Batas episode pelatihan (Requirement Project). |
| `learning_rate` | **0.0003** | Pembelajaran lambat tapi stabil (Stable Baseline standar). Mencegah lupa kemampuan lama. |
| `gamma` | **0.99** | Faktor diskon (Discount Factor). Robot peduli masa depan jangka panjang. |
| `K_epochs` | **10** | Berapa kali data lama dilatih ulang (Replay). Dikurangi untuk mencegah overfitting pada noise. |
| `eps_clip` | **0.2** | Batas perubahan kebijakan PPO agar tidak terlalu drastis. |
| `action_std` | **1.0 -> 0.3** | Tingkat eksplorasi (noise). Dimulai tinggi (1.0) dan berkurang (Decay) setiap 50 episode, tapi dijaga minimal 0.3 agar robot tetap mau mencoba hal baru. |

## 📊 Output
Setelah pelatihan selesai, sistem akan menghasilkan dua grafik:
1.  `reward_graph.png`: Grafik total reward per episode.
2.  `step_graph.png`: Grafik durasi (jumlah langkah) per episode.

File model akan disimpan secara berkala dengan nama `ppo_model_X.pth`.
