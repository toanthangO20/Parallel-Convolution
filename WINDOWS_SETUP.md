# Windows Setup Guide (MSYS2 + MS-MPI)

Tài liệu này hướng dẫn cài đặt, build và chạy các phiên bản Sequential, MPI và MPI+OpenMP trên Windows (native).
Phiên bản CUDA không build trực tiếp trên Windows vì code dùng POSIX headers (unistd.h, fcntl.h) và Makefile kiểu Unix.
Nếu cần chạy CUDA, khuyến nghị dùng WSL2 (Ubuntu) hoặc port lại code sang Win32/MSVC.

## 0) Vị trí repo (đường dẫn tổng quát)
Gọi thư mục gốc repo là `<REPO_ROOT>`.

PowerShell:
```powershell
$REPO_ROOT = "C:\path\to\Parallel-Convolution"
cd $REPO_ROOT
```

MSYS2 MINGW64:
```bash
cd /c/path/to/Parallel-Convolution
```

Lưu ý quan trọng về đường dẫn:
- Trong `MSYS2 MINGW64` / `Git Bash`, luôn dùng dấu `/` trong path (ví dụ: `tools/make_fig4_rgb_40_60.py`, `./seq/seq_conv.exe`).
- Trong `PowerShell`, có thể dùng `\` (ví dụ: `tools\make_fig4_rgb_40_60.py`).

## 1) Cài MSYS2
1. Tải installer từ: https://www.msys2.org/
2. Cài vào đường dẫn ASCII (ví dụ: `C:\msys64`)
3. Mở "MSYS2 MSYS" và cập nhật:
   - `pacman -Syu`
   - khi được yêu cầu, đóng tất cả cửa sổ MSYS2
   - mở lại "MSYS2 MSYS"
   - `pacman -Su`
4. Mở "MSYS2 MINGW64" (không dùng Git Bash)

## 2) Cài toolchain + MPI cho MSYS2
Chạy trong "MSYS2 MINGW64":
```bash
pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-msmpi make
```
Gói `mingw-w64-x86_64-msmpi` cung cấp `mpicc.exe` để build MPI trong MSYS2.

## 3) Cài MS-MPI Runtime (Windows)
1. Tải bộ cài đặt MS-MPI từ Microsoft:
   https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
2. Cài đặt, đảm bảo `mpiexec.exe` nằm trong PATH Windows.

PowerShell (tạm thời cho phiên hiện tại):
```powershell
$env:Path += ";C:\Program Files\Microsoft MPI\Bin"
```

MSYS2 (thêm vĩnh viễn):
```bash
echo 'export PATH="/c/PROGRA~1/Microsoft MPI/Bin:$PATH"' >> ~/.bashrc
```
Đóng/mở lại terminal để PATH có hiệu lực.

## 4) Build (MSYS2 MINGW64)
```bash
cd /c/path/to/Parallel-Convolution

# MPI
mpicc -O3 -o mpi/mpi_conv mpi/mpi_conv.c

# MPI + OpenMP
mpicc -O3 -fopenmp -o mpi_omp/mpi_omp_conv mpi_omp/mpi_omp_conv.c

# Sequential
gcc -O2 -o seq/seq_conv seq/seq_conv.c
```

## 5) Chạy (MSYS2 MINGW64)
Cú pháp chung:
```
<exe> <image.raw> <width> <height> <loops> <rgb|grey>
```

Ví dụ với file mẫu trong repo:
```bash
# MPI
mpiexec -n 4 ./mpi/mpi_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI + OpenMP
OMP_NUM_THREADS=4 mpiexec -n 4 ./mpi_omp/mpi_omp_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# Sequential
./seq/seq_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
```

Ví dụ chạy RGB:
```bash
mpiexec -n 4 ./mpi/mpi_conv waterfall_1920_2520.raw 1920 2520 50 rgb
```

Kết quả sẽ tạo file `blur_<tên_ảnh_gốc>` tại thư mục đang chạy.

## 6) Benchmark (chạy nhiều lần, lấy trung bình)
Script `scripts/benchmark.sh` chạy lệnh nhiều lần và tính trung bình thời gian (chương trình in ra thời gian ở dòng cuối).
```bash
# Sequential
bash scripts/benchmark.sh 5 ./seq/seq_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI
bash scripts/benchmark.sh 5 mpiexec -n 4 ./mpi/mpi_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI + OpenMP
bash scripts/benchmark.sh 5 env OMP_NUM_THREADS=4 mpiexec -n 4 ./mpi_omp/mpi_omp_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
```

## 7) Benchmark Table 1 (MPI runtimes, loops=20)
Script: `mpi/benchmark_table1_mpi.py` tự tạo dữ liệu nhị phân trong `data/`, chạy tất cả case và in LaTeX table ra stdout.

MSYS2 (khuyên dùng nếu có Python trong MSYS2):
```bash
python mpi/benchmark_table1_mpi.py --exe ./mpi/mpi_conv --mpiexec mpiexec
```

PowerShell (dùng Windows Python):
```powershell
py -3 mpi\benchmark_table1_mpi.py --exe .\mpi\mpi_conv --mpiexec mpiexec
```

Kết quả:
- CSV: `mpi/table1_mpi_times.csv`
- Log lỗi (nếu có): `mpi/table1_mpi_errors.log`

## 8) Benchmark Table 2 (MPI+OpenMP runtimes, loops=20)
Script: `mpi_omp/benchmark_table2_mpi_omp.py` tự tạo dữ liệu nhị phân trong `data/`, chạy tất cả case và in LaTeX table ra stdout.

MSYS2:
```bash
python mpi_omp/benchmark_table2_mpi_omp.py --exe ./mpi_omp/mpi_omp_conv --mpiexec mpiexec
```

PowerShell:
```powershell
py -3 mpi_omp\benchmark_table2_mpi_omp.py --exe .\mpi_omp\mpi_omp_conv --mpiexec mpiexec
```

Kết quả:
- CSV: `mpi_omp/table2_mpi_omp_times.csv`
- Log lỗi (nếu có): `mpi_omp/table2_mpi_omp_errors.log`

## 9) Cài Python deps để vẽ biểu đồ
```bash
python -m pip install -r requirements.txt
```
Nếu bạn chỉ cần matplotlib:
```bash
python -m pip install matplotlib
```

## 10) Tạo hình Figure 1 (grey 0 vs 20 iterations)
Script: `tools/make_fig1_grey_0_20.py` (cần Pillow).

PowerShell:
```powershell
py -3 tools\make_fig1_grey_0_20.py --input waterfall_grey_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 20 --outdir figures
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 tools/make_fig1_grey_0_20.py --input waterfall_grey_1920_2520.raw --width 1920 --height 2520 --exe ./seq/seq_conv.exe --loops 20 --outdir figures
```

Kết quả:
- `figures/grey_0.png`
- `figures/grey_20.png`
- `figures/grey_0_20.png` (ảnh ghép tùy chọn)

## 11) Tạo hình Figure 2 (grey 40 vs 60 iterations)
Script: `tools/make_fig2_grey_40_60.py` (cần Pillow).

PowerShell:
```powershell
py -3 tools\make_fig2_grey_40_60.py --input waterfall_grey_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 40 60 --outdir figures
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 tools/make_fig2_grey_40_60.py --input waterfall_grey_1920_2520.raw --width 1920 --height 2520 --exe ./seq/seq_conv.exe --loops 40 60 --outdir figures
```

Kết quả:
- `figures/grey_40.png`
- `figures/grey_60.png`
- `figures/grey_40_60.png`

## 12) Tạo hình Figure 3 (RGB 0 vs 20 iterations)
Script: `tools/make_fig3_rgb_0_20.py` (cần Pillow).

PowerShell:
```powershell
py -3 tools\make_fig3_rgb_0_20.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 20 --outdir figures
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 tools/make_fig3_rgb_0_20.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe ./seq/seq_conv.exe --loops 20 --outdir figures
```

Kết quả:
- `figures/rgb_0.png`
- `figures/rgb_20.png`
- `figures/rgb_0_20.png`

## 13) Tạo hình Figure 4 (RGB 40 vs 60 iterations)
Script: `tools/make_fig4_rgb_40_60.py` (cần Pillow).

PowerShell:
```powershell
py -3 tools\make_fig4_rgb_40_60.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 40 60 --outdir figures
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 tools/make_fig4_rgb_40_60.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe ./seq/seq_conv.exe --loops 40 60 --outdir figures
```

Kết quả:
- `figures/rgb_40.png`
- `figures/rgb_60.png`
- `figures/rgb_40_60.png`

## 14) Tạo hình MPI runtime
Script: `mpi/plot_mpi_runtime.py` (cần matplotlib).

PowerShell:
```powershell
py -3 mpi\plot_mpi_runtime.py
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 mpi/plot_mpi_runtime.py
```

Kết quả:
- `mpi/mpi_runtime.png`
- `mpi/mpi_runtime.pdf`

## 15) Tạo hình MPI speedup & efficiency
Script: `mpi/plot_mpi_speedup_efficiency.py` (cần matplotlib).

PowerShell:
```powershell
py -3 mpi\plot_mpi_speedup_efficiency.py
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 mpi/plot_mpi_speedup_efficiency.py
```

Kết quả:
- `mpi/mpi_speedup.png`
- `mpi/mpi_speedup.pdf`
- `mpi/mpi_efficiency.png`
- `mpi/mpi_efficiency.pdf`

## 16) Tạo hình MPI+OpenMP runtime
Script: `mpi_omp/plot_mpi_omp_runtime.py` (cần matplotlib, đọc `mpi_omp/table2_mpi_omp_times.csv`).

PowerShell:
```powershell
py -3 mpi_omp\plot_mpi_omp_runtime.py
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 mpi_omp/plot_mpi_omp_runtime.py
```

Kết quả:
- `mpi_omp/mpi_omp_runtime.png`
- `mpi_omp/mpi_omp_runtime.pdf`

## 17) Tạo hình MPI+OpenMP speedup & efficiency
Script: `mpi_omp/plot_mpi_omp_speedup_efficiency.py` (cần matplotlib, đọc `mpi_omp/table2_mpi_omp_times.csv`).

PowerShell:
```powershell
py -3 mpi_omp\plot_mpi_omp_speedup_efficiency.py
```

MSYS2 MINGW64 / Git Bash:
```bash
py -3 mpi_omp/plot_mpi_omp_speedup_efficiency.py
```

Kết quả:
- `mpi_omp/mpi_omp_speedup.png`
- `mpi_omp/mpi_omp_speedup.pdf`
- `mpi_omp/mpi_omp_efficiency.png`
- `mpi_omp/mpi_omp_efficiency.pdf`

## 18) Ghi chú về OpenMP
Bạn có thể đặt số lượng thread bằng biến môi trường.

MSYS2:
```bash
export OMP_NUM_THREADS=4
```

PowerShell:
```powershell
$env:OMP_NUM_THREADS = 4
```

## 19) Lỗi thường gặp
- `mpicc: command not found`:
  - Bạn đang ở Git Bash hoặc MSYS2 MSYS, hãy mở "MSYS2 MINGW64".
  - Kiểm tra: `which mpicc`.
- `mpiexec: command not found`:
  - Chưa cài MS-MPI Runtime hoặc PATH chưa đúng.
- `python.exe: can't open file ... toolsmake_fig4_rgb_40_60.py`:
  - Bạn đang chạy trong `MINGW64/Git Bash` nhưng dùng dấu `\`.
  - Dùng lại lệnh với dấu `/`:
    `py -3 tools/make_fig4_rgb_40_60.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe ./seq/seq_conv.exe --loops 40 60 --outdir figures`
- `Cannot divide to processes`:
  - Thay đổi `-n` sao cho width và height chia hết cho lưới process.
- `Error Input!`:
  - Dùng cú pháp: `<exe> <image> <width> <height> <loops> <rgb|grey>`.

## 20) CUDA trên Windows (tùy chọn)
CUDA code trong thư mục `cuda` dùng header POSIX và Makefile kiểu Unix, nên không build trực tiếp trên Windows native.
Nếu cần chạy CUDA, nên dùng WSL2 (Ubuntu) hoặc sửa code/Makefile để build bằng MSVC + nvcc.

## 21) Chuyển .raw/.pgm/.ppm sang PNG
Sau khi dùng `convert_raw.py` để tạo `.pgm` (grey) hoặc `.ppm` (rgb), bạn có thể đổi sang PNG bằng ImageMagick.

1. Cài ImageMagick cho Windows: https://imagemagick.org/
2. Chạy lệnh:
```bash
magick waterfall_grey_1920_2520.pgm waterfall_grey_1920_2520.png
magick waterfall_1920_2520.ppm waterfall_1920_2520.png
```

Nếu muốn đổi trực tiếp từ `.raw` sang PNG (không qua PGM/PPM), dùng:
```bash
magick -size 1920x2520 -depth 8 gray:waterfall_grey_1920_2520.raw waterfall_grey_1920_2520.png
magick -size 1920x2520 -depth 8 rgb:waterfall_1920_2520.raw waterfall_1920_2520.png
```

## 22) Tài liệu tham khảo
- MSYS2 Updating: https://www.msys2.org/docs/updating/
- MSYS2 package `mingw-w64-x86_64-msmpi`: https://packages.msys2.org/package/mingw-w64-x86_64-msmpi
- Microsoft MPI: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

## Appendix: CUDA on Google Colab (recommended for CUDA)
Notebook: `cuda/cuda_colab.ipynb`

1) Open the notebook in Colab
- `File -> Open notebook` and upload `cuda/cuda_colab.ipynb`.
- `Runtime -> Change runtime type` -> select `GPU`.

2) Run cells from top to bottom
The notebook regenerates source files using `%%writefile` and compiles inside Colab.

3) Upload input data (if your notebook expects upload)
- Grey: `waterfall_grey_1920_2520.raw`
- RGB: `waterfall_1920_2520.raw`

4) Run the program (example)
```python
image = "waterfall_grey_1920_2520.raw"
width = 1920
height = 2520
loops = 50
mode = "grey"

!./cuda_conv $image $width $height $loops $mode
```

5) Get results
Output file: `blur_<input>` (e.g. `blur_waterfall_grey_1920_2520.raw`).
Use the download cell inside the notebook to save it.
