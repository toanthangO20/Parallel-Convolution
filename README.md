# Parallel Convolution — Benchmark Summary

Bài toán: áp dụng convolution 2D (Gaussian blur 3×3) cho ảnh GREY/RGB, lặp nhiều vòng (`loops`).

## Các phiên bản triển khai
- Sequential (C)
- MPI (phân chia miền + trao đổi halo)
- MPI + OpenMP (hybrid)
- CUDA (GPU)

## Thiết lập benchmark chính
- Kích thước ảnh: `W = 1920`, `H ∈ {630, 1260, 2520, 5040}` cho cả GREY và RGB (8 case).
- MPI / MPI+OpenMP: `loops = 20`, số process `p ∈ {1, 2, 4, 9, 16, 25}`.
- CUDA (Colab Tesla T4): iterations `{10, 20, 40, 60, 80, 100}`; thống kê **median** của 3 lần đo sau 1 lần warmup.
- Metric: thời gian chạy (seconds). Với MPI/Hybrid dùng **max runtime theo ranks**.

## Kết luận chính
- **MPI**: scaling không đơn điệu; ảnh nhỏ hoặc quá nhiều process khiến overhead giao tiếp/đồng bộ lấn át.
- **MPI+OpenMP**: thường tốt hơn MPI thuần ở nhiều case RGB/medium size, nhưng nhạy với cấu hình threads/ranks.
- **CUDA**: thời gian tăng theo kích thước ảnh và số vòng lặp; overhead cố định rõ hơn ở loops nhỏ.

## Cách chạy
- Windows (MPI/MPI+OMP/Sequential): xem `WINDOWS_SETUP.md`.
- CUDA (Colab): mở `cuda/cuda_colab.ipynb` và chạy từ trên xuống dưới.

## Báo cáo đầy đủ
- Xem `benchmark_report.md` để có mô tả, bảng số liệu và nhận xét chi tiết.
