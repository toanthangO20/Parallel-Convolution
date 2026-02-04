# Parallel 2D Convolution Benchmark (MPI / MPI+OpenMP / CUDA)

## 1) Bài toán và vấn đề cần giải quyết
Ta xét ảnh đầu vào $I$ kích thước $H \times W$ và kernel $3 \times 3$ (Gaussian blur).  
Mỗi pixel đầu ra là tổ hợp tuyến tính của lân cận $3 \times 3$.

### 1.1 Convolution cho ảnh GREY
Với ảnh xám $I \in \{0,\dots,255\}^{H \times W}$ và kernel $h \in \mathbb{R}^{3 \times 3}$, đầu ra $O$ được tính bởi:
$$
O[x,y] = \sum_{\Delta x=-1}^{1}\sum_{\Delta y=-1}^{1}
h[\Delta x+1,\Delta y+1]\cdot I[x+\Delta x,\; y+\Delta y].
$$

### 1.2 Convolution cho ảnh RGB
Với ảnh RGB, tích chập áp dụng độc lập cho từng kênh $c \in \{R,G,B\}$:
$$
O_c[x,y] = \sum_{\Delta x=-1}^{1}\sum_{\Delta y=-1}^{1}
h[\Delta x+1,\Delta y+1]\cdot I_c[x+\Delta x,\; y+\Delta y],\quad c\in\{R,G,B\}.
$$

### 1.3 Lặp nhiều vòng (multiple iterations)
Chương trình lặp phép tích chập trong `loops` vòng:
$$
I^{(0)} = I,\qquad I^{(t+1)} = \mathrm{Conv}\!\left(I^{(t)}, h\right),\quad t=0,\dots,(\text{loops}-1).
$$

### 1.4 Kernel Gaussian blur dùng trong code
Kernel Gaussian blur $3 \times 3$ chuẩn hoá:
$$
h = \frac{1}{16}
\begin{bmatrix}
1 & 2 & 1\\
2 & 4 & 2\\
1 & 2 & 1
\end{bmatrix}.
$$
Trong cài đặt dạng số nguyên, phép chia $1/16$ có thể được hiện thực bằng dịch bit (ví dụ `sum >> 4`).

### 1.5 Xử lý biên (tương đương zero padding)
Ở biên ảnh, các phần tử ngoài miền được coi bằng 0 (zero padding):
$$
I[x,y] = 0 \quad \text{khi } (x,y)\notin [0,H-1]\times[0,W-1].
$$

### 1.6 Vấn đề cần giải quyết
- **Khối lượng tính toán lớn**: mỗi iteration cần $\mathcal{O}(HW)$ phép tính, và lặp nhiều vòng làm tổng chi phí tăng tuyến tính theo `loops`.
- **Memory-bandwidth bound**: mỗi pixel đọc nhiều phần tử lân cận và ghi kết quả → hiệu năng phụ thuộc mạnh vào băng thông bộ nhớ và locality.
- **MPI**: cần trao đổi **halo** ở biên subdomain giữa các tiến trình; khi tăng số process, overhead giao tiếp/đồng bộ có thể lấn át lợi ích chia nhỏ tính toán.
- **MPI+OpenMP**: ngoài overhead MPI, còn có overhead thread scheduling/synchronization và nguy cơ contention/oversubscription.
- **CUDA**: hiệu năng phụ thuộc vào tối ưu truy cập bộ nhớ (coalescing, cache/shared memory) và overhead cố định (cấp phát/copy/launch), đặc biệt khi số vòng lặp nhỏ.

---

## 2) Cài đặt / Build (Windows)
Xem hướng dẫn chi tiết trong: **`WINDOWS_SETUP.md`**.

---

## 3) Mô tả ngắn gọn các phương pháp

### 3.1 MPI
- Chia ảnh theo **domain decomposition** (lưới $row\_div \times col\_div$), mỗi rank giữ **subdomain + halo 1 pixel**.
- Mỗi iteration:
  - Trao đổi halo với láng giềng (Bắc/Nam/Tây/Đông) bằng `MPI_Isend/Irecv` và MPI datatypes (row/column).
  - Tính vùng **inner** trước, chờ halo về, rồi tính vùng **outer/corner**.
- Thời gian báo cáo: **wall-clock time** theo **rank chậm nhất** (max over ranks).

### 3.2 MPI + OpenMP (hybrid)
- Giữ nguyên decomposition/giao tiếp của MPI.
- Bên trong mỗi rank, song song hoá vòng lặp tích chập bằng **OpenMP** để khai thác đa lõi shared-memory.
- Hiệu năng nhạy với `OMP_NUM_THREADS`, pinning/binding, và việc phân bổ tài nguyên giữa ranks và threads.

### 3.3 CUDA (GPU)
- CUDA triển khai tích chập bằng kernel GPU tính theo từng pixel (thường mỗi thread xử lý 1 pixel hoặc 1 nhóm pixel).
- Để chạy benchmark CUDA: **upload file `.ipynb` lên Google Colab** và chạy các cell theo thứ tự (notebook đã bao gồm phần compile + benchmark).

---

## 4) Kịch bản thử nghiệm và kết quả thực nghiệm

### 4.1 Dữ liệu đầu vào
Benchmark chạy trên 8 input (4 GREY + 4 RGB), cùng độ rộng cố định $W=1920$ và các độ cao:
- $H \in \{630,\,1260,\,2520,\,5040\}$ tương ứng với hệ số (x/4, x/2, x, 2x).

Quy ước:
- **GREY**: 1 kênh (mỗi pixel 1 byte).
- **RGB**: 3 kênh (mỗi pixel 3 bytes).

---

### 4.2 MPI (20 iterations)
- `loops = 20`
- Số MPI processes: $p \in \{1,2,4,9,16,25\}$
- Metric: **max runtime theo ranks** (seconds)

| Image size | 1 | 2 | 4 | 9 | 16 | 25 |
|---|---:|---:|---:|---:|---:|---:|
| grey 1920×630 (x/4)  | 0.09 | 0.17 | 0.04 | 0.02 | 0.02 | 0.21 |
| grey 1920×1260 (x/2) | 0.19 | 0.12 | 0.18 | 0.03 | 0.03 | 0.24 |
| grey 1920×2520 (x)   | 0.41 | 0.24 | 0.11 | 0.15 | 0.06 | 0.10 |
| grey 1920×5040 (2x)  | 0.92 | 0.47 | 0.22 | 0.12 | 0.20 | 0.33 |
| rgb 1920×630 (x/4)   | 0.33 | 0.20 | 0.10 | 0.05 | 0.05 | 0.08 |
| rgb 1920×1260 (x/2)  | 0.66 | 0.34 | 0.18 | 0.11 | 0.08 | 0.10 |
| rgb 1920×2520 (x)    | 1.36 | 0.80 | 0.34 | 0.17 | 0.31 | 0.18 |
| rgb 1920×5040 (2x)   | 3.07 | 1.56 | 0.70 | 0.35 | 0.27 | 0.29 |

#### Nhận xét chi tiết (MPI)
- **Xu hướng scaling theo kích thước ảnh**:  
  Với ảnh nhỏ (đặc biệt GREY 1920×630), chi phí tính toán mỗi rank không đủ lớn để “che” chi phí giao tiếp, nên việc tăng process có thể gây **chậm hơn** (ví dụ $p=2$ hoặc $p=25$). Ngược lại, khi ảnh lớn (GREY/RGB 1920×5040), workload per-rank đủ lớn nên lợi ích chia nhỏ tính toán rõ rệt hơn ở vùng $p$ vừa phải.
- **Hành vi không đơn điệu khi tăng $p$**:  
  Dữ liệu cho thấy runtime không luôn giảm khi tăng process. Nguyên nhân điển hình là khi $p$ lớn, block mỗi rank nhỏ, tỉ lệ **biên/hay “surface-to-volume”** tăng lên: phần tính toán tỉ lệ với diện tích block, nhưng giao tiếp halo tỉ lệ với chu vi block. Khi chu vi chiếm ưu thế, tổng thời gian bị chi phối bởi halo exchange và đồng bộ.
- **Khác biệt GREY vs RGB**:  
  RGB có 3 kênh nên khối lượng tính toán lớn hơn rõ rệt. Vì vậy RGB thường hưởng lợi từ song song hoá ở dải $p$ rộng hơn (vẫn còn đủ compute để amortize overhead). Tuy nhiên, khi $p$ rất lớn, overhead MPI vẫn có thể thắng thế (ví dụ RGB 1920×2520 tại $p=16$ tăng so với $p=9$).

---

### 4.3 MPI + OpenMP (20 iterations)
- `loops = 20`
- Số MPI processes: $p \in \{1,2,4,9,16,25\}$
- Metric: **max runtime theo ranks** (seconds)

| Image size | 1 | 2 | 4 | 9 | 16 | 25 |
|---|---:|---:|---:|---:|---:|---:|
| grey 1920×630 (x/4)  | 0.14 | 0.06 | 0.03 | 0.05 | 0.12 | 0.05 |
| grey 1920×1260 (x/2) | 0.17 | 0.06 | 0.07 | 0.10 | 0.21 | 0.06 |
| grey 1920×2520 (x)   | 0.27 | 0.14 | 0.13 | 0.21 | 0.31 | 0.09 |
| grey 1920×5040 (2x)  | 0.40 | 0.23 | 0.16 | 0.32 | 0.17 | 0.14 |
| rgb 1920×630 (x/4)   | 0.21 | 0.13 | 0.08 | 0.13 | 0.23 | 0.06 |
| rgb 1920×1260 (x/2)  | 0.24 | 0.19 | 0.19 | 0.24 | 0.09 | 0.11 |
| rgb 1920×2520 (x)    | 0.46 | 0.34 | 0.19 | 0.14 | 0.16 | 0.19 |
| rgb 1920×5040 (2x)   | 0.93 | 0.61 | 0.32 | 0.33 | 0.45 | 0.64 |

#### Nhận xét chi tiết (MPI + OpenMP)
- **Vai trò của hybrid**:  
  Hybrid nhằm giảm nhu cầu tăng nhiều MPI ranks (giảm traffic halo) bằng cách đẩy một phần song song hoá sang OpenMP trong mỗi rank. Điều này thường hiệu quả khi CPU có nhiều core và mỗi rank vẫn giữ block đủ lớn.
- **Tính nhạy với cấu hình threads và tài nguyên CPU**:  
  Dữ liệu cho thấy có nhiều điểm không đơn điệu (ví dụ GREY 1920×2520: $p=4$ tốt hơn $p=9$ và $p=16$). Các nguyên nhân thường gặp:
  - **Oversubscription**: tổng số threads (MPI ranks × OMP threads) vượt số core vật lý.
  - **Thread scheduling / contention**: nhiều threads truy cập vùng nhớ gần nhau gây tranh chấp cache/memory bandwidth.
  - **Pinning/binding**: nếu threads/ranks không được gắn hợp lý lên core/NUMA, chi phí truy cập bộ nhớ tăng.
- **Khi nào hybrid có lợi rõ rệt**:  
  Ở nhiều hàng RGB, runtime hybrid nhỏ hơn nhiều so với MPI thuần ở cùng $p$ (ví dụ RGB 1920×2520: $p=4$ là 0.19s so với 0.34s). Điều này phù hợp với kỳ vọng: RGB nặng compute hơn nên OpenMP khai thác tốt hơn, đồng thời việc không cần tăng quá nhiều ranks giúp giảm overhead halo.
- **Khi nào hybrid kém ổn định**:  
  Với một số ảnh GREY, hybrid có thể bị “tụt” mạnh ở $p$ lớn (ví dụ GREY 1920×2520 tại $p=16$). Điều này phản ánh bài toán GREY nhẹ compute hơn nên thread overhead và memory contention có thể trở thành yếu tố chi phối.

---

### 4.4 CUDA (Colab – Tesla T4, varying iterations)
- Iterations: $\{10,20,40,60,80,100\}$
- Statistic: **median của 3 lần đo sau 1 lần warmup**
- Môi trường đo: Google Colab (GPU Tesla T4)

**GREY — CUDA runtime (sec, median)**

| Image size | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| grey 1920×630 (x/4)  | 0.080 | 0.101 | 0.109 | 0.120 | 0.104 | 0.109 |
| grey 1920×1260 (x/2) | 0.085 | 0.110 | 0.112 | 0.117 | 0.119 | 0.129 |
| grey 1920×2520 (x)   | 0.087 | 0.103 | 0.114 | 0.135 | 0.144 | 0.135 |
| grey 1920×5040 (2x)  | 0.088 | 0.111 | 0.126 | 0.143 | 0.178 | 0.191 |

**RGB — CUDA runtime (sec, median)**

| Image size | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| rgb 1920×630 (x/4)   | 0.083 | 0.084 | 0.092 | 0.104 | 0.120 | 0.128 |
| rgb 1920×1260 (x/2)  | 0.088 | 0.106 | 0.128 | 0.144 | 0.147 | 0.147 |
| rgb 1920×2520 (x)    | 0.098 | 0.108 | 0.133 | 0.155 | 0.199 | 0.224 |
| rgb 1920×5040 (2x)   | 0.113 | 0.143 | 0.179 | 0.237 | 0.293 | 0.340 |

#### Nhận xét chi tiết (CUDA)
- **Phụ thuộc theo số vòng lặp**:  
  Thời gian nhìn chung tăng theo `loops`, nhưng không hoàn toàn tuyến tính vì có một phần overhead gần như cố định (setup, quản lý bộ nhớ, launch). Khi `loops` còn nhỏ, overhead này chiếm tỉ trọng lớn nên đường cong có thể “phẳng” hơn hoặc có dao động nhỏ giữa các mốc.
- **Phụ thuộc theo kích thước ảnh**:  
  Khi tăng kích thước từ x/4 → 2x, runtime tăng rõ (đặc biệt ở RGB), phản ánh đúng bản chất $\mathcal{O}(HW \cdot \text{loops})$. Ở kích thước lớn, GPU được “nuôi” đủ dữ liệu nên thời gian có xu hướng ổn định hơn (ít bị nhiễu tương đối).
- **GREY vs RGB**:  
  RGB xử lý 3 kênh nên lượng tính toán và băng thông cần thiết cao hơn, do đó runtime lớn hơn GREY và tăng nhanh hơn theo `loops`/kích thước.
- **Dao động nhỏ giữa các mốc**:  
  Ví dụ GREY 1920×2520 ở 80 loops lớn hơn 100 loops (0.144 vs 0.135) là sai khác nhỏ do nhiễu đo, chia sẻ tài nguyên trên Colab, và đặc tính median theo số lần chạy hạn chế. Đây là hiện tượng chấp nhận được nếu chênh lệch không lớn và xu hướng tổng thể vẫn hợp lý.
- **Lưu ý khi so sánh với CPU (MPI / Hybrid)**:  
  CUDA được đo trên GPU T4 (Colab), trong khi MPI/Hybrid thường đo trên CPU khác. Vì khác phần cứng và môi trường chạy, so sánh tuyệt đối giữa CUDA và MPI/Hybrid cần ghi rõ bối cảnh; phù hợp nhất là so sánh *xu hướng* (tăng theo kích thước/loops) và *đặc điểm overhead* của từng mô hình.

---
