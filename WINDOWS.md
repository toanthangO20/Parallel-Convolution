# Hướng dẫn chạy trên Windows

Tài liệu này hướng dẫn build và chạy phiên bản MPI và MPI+OpenMP trên Windows (native).
Phiên bản CUDA hiện không build trực tiếp trên Windows vì code dùng POSIX headers (unistd.h, fcntl.h) và Makefile kiểu Unix.
Nếu muốn chạy CUDA, khuyến nghị dùng WSL2 (Ubuntu) hoặc port lại code sang Win32.

## Yêu cầu
- Windows 10/11 64-bit
- MSYS2 (MINGW64)
- MS-MPI Runtime (Windows)
- Repo đã clone

## Bước 1: Cài MSYS2
1) Tải installer từ: https://www.msys2.org/
2) Cài vào đường dẫn ASCII (ví dụ: C:\\msys64)
3) Mở "MSYS2 MSYS" và cập nhật (MSYS2 là rolling release, cần full system upgrade):
   - pacman -Syu
   - khi được yêu cầu, đóng tất cả cửa sổ MSYS2
   - mở lại "MSYS2 MSYS"
   - pacman -Su
4) Mở "MSYS2 MINGW64" (không dùng Git Bash)

## Bước 2: Cài toolchain + MPI cho MSYS2
Chạy trong "MSYS2 MINGW64":

```
pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-msmpi make
```
Gói `mingw-w64-x86_64-msmpi` cung cấp `mpicc.exe` để build MPI trong MSYS2.

## Bước 3: Cài MS-MPI Runtime (Windows)
1) Tải bộ cài đặt MS-MPI từ Microsoft:
   https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
2) Cài đặt, đảm bảo `mpiexec.exe` nằm trong PATH.
   Thường ở: C:\\Program Files\\Microsoft MPI\\Bin
   Nếu bạn cài MS-MPI ở `C:\MPI\Bin`, trong MSYS2 có thể thêm PATH vĩnh viễn bằng:

   ```
   echo 'export PATH="/c/MPI/Bin:$PATH"' >> ~/.bashrc
   ```
   (đóng/mở lại MSYS2 sau đó)
3) Mở lại terminal nếu vừa thêm PATH.

## Bước 4: Build và chạy
Chạy trong "MSYS2 MINGW64" tại thư mục repo:

```
cd /e/Master/20251/HighPerformanceComputing/Parallel-Convolution

# MPI
mpicc -O3 -o mpi/mpi_conv mpi/mpi_conv.c
mpiexec -n 4 ./mpi/mpi_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI + OpenMP
mpicc -O3 -fopenmp -o mpi_omp/mpi_omp_conv mpi_omp/mpi_omp_conv.c
OMP_NUM_THREADS=4 mpiexec -n 4 ./mpi_omp/mpi_omp_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
 
# Sequential
gcc -O2 -o seq/seq_conv seq/seq_conv.c
./seq/seq_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
```

Ví dụ chạy RGB:

```
mpiexec -n 4 ./mpi/mpi_conv waterfall_1920_2520.raw 1920 2520 50 rgb
```

Kết quả sẽ tạo file: `blur_<tên_ảnh_gốc>` tại thư mục đang chạy.

## Benchmark (chạy nhiều lần, lấy trung bình)
Script `scripts/benchmark.sh` chạy lệnh nhiều lần và tính trung bình thời gian (chương trình in ra thời gian ở dòng cuối).

```
# Sequential
bash scripts/benchmark.sh 5 ./seq/seq_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI
bash scripts/benchmark.sh 5 mpiexec -n 4 ./mpi/mpi_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey

# MPI + OpenMP (dùng env để set biến môi trường)
bash scripts/benchmark.sh 5 env OMP_NUM_THREADS=4 mpiexec -n 4 ./mpi_omp/mpi_omp_conv waterfall_grey_1920_2520.raw 1920 2520 50 grey
```

## Benchmark Table 1 (MPI runtimes, loops=20)
Script: `benchmark_table1_mpi.py` tự tạo dữ liệu nhị phân trong `data/`, chạy tất cả case và in LaTeX table ra stdout.

Chạy trong "MSYS2 MINGW64":

```
cd /c/Users/Administrator/Downloads/Parallel-Convolution

# build MPI
mpicc -O3 -o mpi/mpi_conv mpi/mpi_conv.c

# chạy benchmark
python benchmark_table1_mpi.py --exe ./mpi/mpi_conv --mpiexec mpiexec
```

Nếu MSYS2 không có python, gọi python Windows từ MSYS2:

```
/c/Users/Administrator/AppData/Local/Programs/Python/Python312/python.exe \
  benchmark_table1_mpi.py --exe ./mpi/mpi_conv --mpiexec mpiexec
```

Kết quả:
- CSV: `table1_mpi_times.csv`
- Log lỗi (nếu có): `table1_mpi_errors.log`

## Tạo hình Figure 1 (grey 0 vs 20 iterations)
Script Python: `tools/make_fig1_grey_0_20.py` (cần Pillow).

1) Cài Pillow:
```
py -m pip install -r requirements.txt
```

2) Chạy trong PowerShell (khuyên dùng):
```
& "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe" tools\make_fig1_grey_0_20.py --input waterfall_grey_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 20 --outdir figures
```

Script sẽ tạo:
- `figures/grey_0.png`
- `figures/grey_20.png`
- `figures/grey_0_20.png` (ảnh ghép tùy chọn)

## Tạo hình RGB (40 vs 60 iterations)
Script Python: `tools/make_fig4_rgb_40_60.py` (cần Pillow).

Ví dụ chạy trong PowerShell:
```
& "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe" tools\make_fig4_rgb_40_60.py --input waterfall_1920_2520.raw --width 1920 --height 2520 --exe .\seq\seq_conv.exe --loops 40 60 --outdir Figures
```

Script sẽ tạo:
- `Figures/rgb_40.png`
- `Figures/rgb_60.png`
- `Figures/rgb_40_60.png`

## Ghi chú về OpenMP
Số lượng thread có thể điều chỉnh bằng biến môi trường. Ví dụ:

```
export OMP_NUM_THREADS=4
```

## Lỗi thường gặp
- `mpicc: command not found`:
  - Bạn đang ở Git Bash hoặc MSYS2 MSYS, hãy mở "MSYS2 MINGW64".
  - Kiểm tra: `which mpicc`.
- `mpiexec: command not found`:
  - Chưa cài MS-MPI Runtime hoặc PATH chưa đúng.
- `Cannot divide to processes`:
  - Thay đổi `-n` sao cho width và height chia hết cho lưới process.
  - Ví dụ với 1920x2520, `-n 4` chạy được.
- `Error Input!`:
  - Dùng cú pháp: `<exe> <image> <width> <height> <loops> <rgb|grey>`.

## CUDA trên Windows (tùy chọn)
CUDA code trong thư mục `cuda` dùng header POSIX và Makefile kiểu Unix, nên không build trực tiếp trên Windows native.
Nếu cần chạy CUDA, nên dùng WSL2 (Ubuntu) hoặc sửa code/Makefile để build bằng MSVC + nvcc.

## Chuyển .raw/.pgm/.ppm sang PNG
Sau khi dùng `convert_raw.py` để tạo `.pgm` (grey) hoặc `.ppm` (rgb), bạn có thể đổi sang PNG bằng ImageMagick.

1) Cài ImageMagick cho Windows: https://imagemagick.org/
2) Chạy lệnh:

```
magick waterfall_grey_1920_2520.pgm waterfall_grey_1920_2520.png
magick waterfall_1920_2520.ppm waterfall_1920_2520.png
```

Nếu muốn đổi trực tiếp từ `.raw` sang PNG (không qua PGM/PPM), dùng:

```
magick -size 1920x2520 -depth 8 gray:waterfall_grey_1920_2520.raw waterfall_grey_1920_2520.png
magick -size 1920x2520 -depth 8 rgb:waterfall_1920_2520.raw waterfall_1920_2520.png
```

## Tài liệu tham khảo
- MSYS2 Updating: https://www.msys2.org/docs/updating/
- MSYS2 package `mingw-w64-x86_64-msmpi`: https://packages.msys2.org/package/mingw-w64-x86_64-msmpi
- Microsoft MPI: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
