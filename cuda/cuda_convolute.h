#ifndef CUDA_CONVOLUTE_H
#define CUDA_CONVOLUTE_H

#include "funcs.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpuConvolute(uint8_t *src, int width, int height, int loops, color_t imageType);

#ifdef __cplusplus
}
#endif

#endif
