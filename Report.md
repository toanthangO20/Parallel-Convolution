# Báo cáo ngắn: Parallel 2D Convolution (MPI / MPI+OpenMP / CUDA)

## 1) Mục tiêu
Xây dựng và benchmark chương trình **lọc ảnh bằng tích chập (convolution) 2D** theo 3 mô hình song song:
- **MPI** (distributed-memory)
- **MPI + OpenMP** (hybrid, shared-memory trong mỗi process)
- **CUDA** (GPU)

Bài toán phù hợp để chứng minh ưu điểm tính toán song song vì:
- Tính toán theo pixel độc lập (stencil 2D), tải đều và dễ chia miền.
- Có điểm “nghẽn” rõ ràng để phân tích: **băng thông bộ nhớ** và **chi phí giao tiếp halo**.

---

## 2) Bài toán gốc & công thức toán học

### 2.1 Convolution 2D (kernel 3×3) – GREY
Cho ảnh xám $I$ kích thước $H\times W$ và kernel $h\in\mathbb{R}^{3\times 3}$.
Với mỗi pixel $(X,Y)$:

$$
O[X,Y] \;=\; \sum_{\Delta x=-1}^{1}\sum_{\Delta y=-1}^{1}
h[\Delta x+1,\Delta y+1]\cdot I[X+\Delta x,\; Y+\Delta y]
$$

Giá trị tích lũy tính bằng `float` và **ép về `uint8_t`** (trong code là cast trực tiếp).

\begin{figure}[h]
  \centering
  \begin{minipage}{0.49\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/grey_0.png}
  \end{minipage}
  \begin{minipage}{0.49\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/grey_20.png}
  \end{minipage}
  \caption{Grey: left 0 iterations (original), right 20 iterations (to be inserted).}
  \label{fig:grey_0_20}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.92\linewidth]{Figures/grey_40_60.png}
  \caption{Grey: left 40 iterations, right 60 iterations.}
  \label{fig:grey_40_60}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.92\linewidth]{Figures/rgb_0_20.png}
  \caption{RGB: left 0 iterations (original), right 20 iterations.}
  \label{fig:rgb_0_20}
\end{figure}

### 2.2 RGB
Áp dụng convolution **độc lập trên từng kênh** $c\in\{R,G,B\}$:

$$
O_c[X,Y] \;=\; \sum_{\Delta x=-1}^{1}\sum_{\Delta y=-1}^{1}
h[\Delta x+1,\Delta y+1]\cdot I_c[X+\Delta x,\; Y+\Delta y],
\quad c\in\{R,G,B\}
$$

### 2.3 Lặp nhiều lần (iterations)
Chương trình áp dụng convolution lặp `loops` lần:

$$
I^{(0)} = I,\quad
I^{(t+1)} = \mathrm{Conv}(I^{(t)}, h),\quad
t = 0..(\text{loops}-1)
$$

---

## 3) Điều kiện biên (boundary condition) theo code
Mỗi MPI process lưu **subdomain + halo 1 pixel**. Halo được trao đổi giữa các process lân cận.
Nếu ở biên global không có láng giềng (neighbor = -1) thì halo giữ **0** do cấp phát `calloc`.
=> Ở mức toàn cục tương đương **zero-padding** ngoài biên ảnh:

$$
I[X,Y] = 0 \quad \text{khi } (X,Y)\notin [0,H-1]\times[0,W-1]
$$

---

## 4) Thiết kế song song (MPI)

### 4.1 Chia miền (domain decomposition)
Ảnh được chia thành lưới block $row\_div\times col\_div$, với:

$$
rows = \frac{H}{row\_div},\quad cols = \frac{W}{col\_div}
$$

Với `process_id`, tọa độ block:

$$
p_r = \left\lfloor \frac{process\_id}{col\_div}\right\rfloor,\quad
p_c = process\_id\bmod col\_div
$$

Góc trên-trái block trong ảnh global:

$$
start\_row = p_r\cdot rows,\quad start\_col = p_c\cdot cols
$$

Trong bộ nhớ local, chỉ số chạy $x=1..rows$, $y=1..cols$ (vì có halo ở biên).

### 4.2 Trao đổi halo (communication)
Xác định 4 láng giềng: `north/south/west/east`.  
Mỗi iteration: gửi/nhận biên bằng **non-blocking** `MPI_Isend/Irecv`.

Code dùng MPI datatype để gửi biên hiệu quả:
- **Row type**: `MPI_Type_contiguous(...)`
- **Column type**: `MPI_Type_vector(...)` (stride theo bề rộng local có halo)

### 4.3 Lịch tính toán (overlap compute/comm)
Mẫu thực thi một iteration (đúng flow code):

1. `Isend/Irecv` biên (north, west, south, east)  
2. **Tính inner region** (không cần halo mới)  
3. `Wait` nhận halo theo từng hướng và tính **outer edge** tương ứng  
4. Tính **4 góc** (nếu có đủ halo)  
5. `Wait` các send hoàn thành  
6. Swap `src/dst`

---

## 5) Kernel trong code
Ví dụ kernel 3×3 (đang bật Gaussian blur):

$$
h \;=\; \frac{1}{16}
\begin{bmatrix}
1 & 2 & 1\\
2 & 4 & 2\\
1 & 2 & 1
\end{bmatrix}
$$

(Có thể đổi sang box blur hoặc edge kernel bằng cách bật dòng tương ứng.)

---

## 6) I/O song song (Parallel I/O)
Sử dụng `MPI_File_open`, mỗi process:
- Seek tới đúng offset theo `start_row/start_col`
- Đọc/ghi **chỉ phần subdomain** của mình theo từng hàng (row-by-row)

Lợi ích:
- Tránh root đọc toàn bộ ảnh rồi scatter
- Tăng tốc I/O với file lớn và nhiều process

---

## 7) Chỉ số benchmark
- **Runtime**: thời gian tối đa giữa các process (rank 0 lấy max)
- **Speedup**:

$$
S(p) = \frac{T_1}{T_p}
$$

- **Efficiency**:

$$
E(p) = \frac{S(p)}{p}
$$

Khuyến nghị báo cáo thêm:
- Strong scaling: giữ ảnh cố định, tăng $p$
- Weak scaling: tăng kích thước ảnh theo $p$ để giữ tải/process ~ không đổi

---

## 8) Cách chạy (ví dụ)
GREY:
```bash
mpirun -np 4 ./mpi_conv input_grey.raw 1920 2520 50 grey
