#include <stdio.h>
#include <stdlib.h>
#include "cuda_convolute.h"
#include "funcs.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16

__device__ __constant__ int c_kernel[9];

__device__ __forceinline__ uint8_t div16_u8(int sum) {
    return (uint8_t)(sum >> 4);
}

__global__ void kernel_conv_grey(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width, int height) {
    __shared__ uint8_t tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < height && y < width) {
        tile[tx][ty] = src[width * x + y];
    } else {
        tile[tx][ty] = 0;
    }

    if (threadIdx.x == 0) {
        tile[0][ty] = (x > 0 && y < width) ? src[width * (x - 1) + y] : 0;
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        tile[BLOCK_SIZE + 1][ty] = (x + 1 < height && y < width) ? src[width * (x + 1) + y] : 0;
    }
    if (threadIdx.y == 0) {
        tile[tx][0] = (y > 0 && x < height) ? src[width * x + (y - 1)] : 0;
    }
    if (threadIdx.y == BLOCK_SIZE - 1) {
        tile[tx][BLOCK_SIZE + 1] = (y + 1 < width && x < height) ? src[width * x + (y + 1)] : 0;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile[0][0] = (x > 0 && y > 0) ? src[width * (x - 1) + (y - 1)] : 0;
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        tile[0][BLOCK_SIZE + 1] = (x > 0 && y + 1 < width) ? src[width * (x - 1) + (y + 1)] : 0;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        tile[BLOCK_SIZE + 1][0] = (x + 1 < height && y > 0) ? src[width * (x + 1) + (y - 1)] : 0;
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] =
            (x + 1 < height && y + 1 < width) ? src[width * (x + 1) + (y + 1)] : 0;
    }

    __syncthreads();

    if (x > 0 && x < height - 1 && y > 0 && y < width - 1) {
        int sum = 0;
        sum += tile[tx - 1][ty - 1] * c_kernel[0];
        sum += tile[tx - 1][ty] * c_kernel[1];
        sum += tile[tx - 1][ty + 1] * c_kernel[2];
        sum += tile[tx][ty - 1] * c_kernel[3];
        sum += tile[tx][ty] * c_kernel[4];
        sum += tile[tx][ty + 1] * c_kernel[5];
        sum += tile[tx + 1][ty - 1] * c_kernel[6];
        sum += tile[tx + 1][ty] * c_kernel[7];
        sum += tile[tx + 1][ty + 1] * c_kernel[8];
        dst[width * x + y] = div16_u8(sum);
    }
}

__global__ void kernel_conv_rgb(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width, int height) {
    __shared__ uchar3 tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < height && y < width) {
        int idx = (x * width + y) * 3;
        tile[tx][ty] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
    } else {
        tile[tx][ty] = make_uchar3(0, 0, 0);
    }

    if (threadIdx.x == 0) {
        if (x > 0 && y < width) {
            int idx = ((x - 1) * width + y) * 3;
            tile[0][ty] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[0][ty] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        if (x + 1 < height && y < width) {
            int idx = ((x + 1) * width + y) * 3;
            tile[BLOCK_SIZE + 1][ty] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[BLOCK_SIZE + 1][ty] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.y == 0) {
        if (y > 0 && x < height) {
            int idx = (x * width + (y - 1)) * 3;
            tile[tx][0] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[tx][0] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.y == BLOCK_SIZE - 1) {
        if (y + 1 < width && x < height) {
            int idx = (x * width + (y + 1)) * 3;
            tile[tx][BLOCK_SIZE + 1] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[tx][BLOCK_SIZE + 1] = make_uchar3(0, 0, 0);
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (x > 0 && y > 0) {
            int idx = ((x - 1) * width + (y - 1)) * 3;
            tile[0][0] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[0][0] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        if (x > 0 && y + 1 < width) {
            int idx = ((x - 1) * width + (y + 1)) * 3;
            tile[0][BLOCK_SIZE + 1] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[0][BLOCK_SIZE + 1] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        if (x + 1 < height && y > 0) {
            int idx = ((x + 1) * width + (y - 1)) * 3;
            tile[BLOCK_SIZE + 1][0] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[BLOCK_SIZE + 1][0] = make_uchar3(0, 0, 0);
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        if (x + 1 < height && y + 1 < width) {
            int idx = ((x + 1) * width + (y + 1)) * 3;
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = make_uchar3(src[idx], src[idx + 1], src[idx + 2]);
        } else {
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = make_uchar3(0, 0, 0);
        }
    }

    __syncthreads();

    if (x > 0 && x < height - 1 && y > 0 && y < width - 1) {
        int sum_r = 0, sum_g = 0, sum_b = 0;
        const int *k = c_kernel;
        uchar3 v;

        v = tile[tx - 1][ty - 1]; sum_r += v.x * k[0]; sum_g += v.y * k[0]; sum_b += v.z * k[0];
        v = tile[tx - 1][ty];     sum_r += v.x * k[1]; sum_g += v.y * k[1]; sum_b += v.z * k[1];
        v = tile[tx - 1][ty + 1]; sum_r += v.x * k[2]; sum_g += v.y * k[2]; sum_b += v.z * k[2];
        v = tile[tx][ty - 1];     sum_r += v.x * k[3]; sum_g += v.y * k[3]; sum_b += v.z * k[3];
        v = tile[tx][ty];         sum_r += v.x * k[4]; sum_g += v.y * k[4]; sum_b += v.z * k[4];
        v = tile[tx][ty + 1];     sum_r += v.x * k[5]; sum_g += v.y * k[5]; sum_b += v.z * k[5];
        v = tile[tx + 1][ty - 1]; sum_r += v.x * k[6]; sum_g += v.y * k[6]; sum_b += v.z * k[6];
        v = tile[tx + 1][ty];     sum_r += v.x * k[7]; sum_g += v.y * k[7]; sum_b += v.z * k[7];
        v = tile[tx + 1][ty + 1]; sum_r += v.x * k[8]; sum_g += v.y * k[8]; sum_b += v.z * k[8];

        int out_idx = (x * width + y) * 3;
        dst[out_idx] = div16_u8(sum_r);
        dst[out_idx + 1] = div16_u8(sum_g);
        dst[out_idx + 2] = div16_u8(sum_b);
    }
}

extern "C" void gpuConvolute(uint8_t *src, int width, int height, int loops, color_t imageType) {
    uint8_t *d_src, *d_dst, *tmp;
    size_t bytes = (imageType == GREY) ? (size_t)height * width : (size_t)height * width * 3;

    static const int h_kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_kernel, h_kernel, sizeof(h_kernel)));

    CUDA_SAFE_CALL(cudaMalloc(&d_src, bytes * sizeof(uint8_t)));
    CUDA_SAFE_CALL(cudaMalloc(&d_dst, bytes * sizeof(uint8_t)));

    CUDA_SAFE_CALL(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_dst, 0, bytes));

    const int blockSize = BLOCK_SIZE;
    dim3 block(blockSize, blockSize);
    dim3 grid_grey(FRACTION_CEILING(height, blockSize), FRACTION_CEILING(width, blockSize));
    dim3 grid_rgb(FRACTION_CEILING(height, blockSize), FRACTION_CEILING(width, blockSize));

    for (int t = 0; t < loops; t++) {
        if (imageType == GREY) {
            kernel_conv_grey<<<grid_grey, block>>>(d_src, d_dst, width, height);
        } else if (imageType == RGB) {
            kernel_conv_rgb<<<grid_rgb, block>>>(d_src, d_dst, width, height);
        }

        tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    if (loops % 2 == 0) {
        CUDA_SAFE_CALL(cudaMemcpy(src, d_src, bytes, cudaMemcpyDeviceToHost));
    } else {
        CUDA_SAFE_CALL(cudaMemcpy(src, d_dst, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_SAFE_CALL(cudaFree(d_src));
    CUDA_SAFE_CALL(cudaFree(d_dst));
}
