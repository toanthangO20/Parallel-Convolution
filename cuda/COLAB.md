# Hướng dẫn chạy CUDA trên Google Colab

Tệp notebook: `cuda_colab.ipynb`

## 1) Mở notebook trên Colab
- Vào Colab, chọn `File -> Open notebook` và tải lên `cuda/cuda_colab.ipynb`.
- Vào `Runtime -> Change runtime type` và chọn `GPU`.

## 2) Chạy các cell theo thứ tự
Chạy lần lượt từ trên xuống dưới. Notebook sẽ tự tạo lại toàn bộ source code trong các cell `%%writefile`.

## 3) Upload dữ liệu đầu vào
- Grey: `waterfall_grey_1920_2520.raw`
- RGB: `waterfall_1920_2520.raw`

Các file này sẽ nằm cùng thư mục làm việc của Colab sau khi upload.

## 4) Chạy chương trình
Ví dụ grey:
```python
image = "waterfall_grey_1920_2520.raw"
width = 1920
height = 2520
loops = 50
mode = "grey"

!./cuda_conv $image $width $height $loops $mode
```

Ví dụ rgb:
```python
image = "waterfall_1920_2520.raw"
width = 1920
height = 2520
loops = 50
mode = "rgb"

!./cuda_conv $image $width $height $loops $mode
```

## 5) Lấy kết quả
File đầu ra: `blur_<input>` (ví dụ `blur_waterfall_grey_1920_2520.raw`).
Dùng cell download trong notebook để tải về máy.
