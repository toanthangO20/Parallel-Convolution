#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef enum {RGB, GREY} color_t;

void convolute(uint8_t *, uint8_t *, int, int, int, int, int, int, float**, color_t);
static inline void convolute_grey(uint8_t *, uint8_t *, int, int, int, int, float **);
static inline void convolute_rgb(uint8_t *, uint8_t *, int, int, int, int, float **);
void Usage(int, char **, char **, int *, int *, int *, color_t *);
uint8_t *offset(uint8_t *, int, int, int);

int main(int argc, char** argv) {
	int i, j, width, height, loops, t;
	double timer;
	char *image = NULL;
	color_t imageType;

	Usage(argc, argv, &image, &width, &height, &loops, &imageType);

	/* Init filters */
	int box_blur[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
	int gaussian_blur[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	int edge_detection[3][3] = {{1, 4, 1}, {4, 8, 4}, {1, 4, 1}};
	float **h = malloc(3 * sizeof(float *));
	if (h == NULL) {
		fprintf(stderr, "%s: Not enough memory\n", argv[0]);
		return EXIT_FAILURE;
	}
	for (i = 0 ; i < 3 ; i++) {
		h[i] = malloc(3 * sizeof(float));
		if (h[i] == NULL) {
			fprintf(stderr, "%s: Not enough memory\n", argv[0]);
			return EXIT_FAILURE;
		}
	}
	for (i = 0 ; i < 3 ; i++) {
		for (j = 0 ; j < 3 ; j++){
			/* h[i][j] = box_blur[i][j] / 9.0f; */
			h[i][j] = gaussian_blur[i][j] / 16.0f;
			/* h[i][j] = edge_detection[i][j] / 28.0f; */
		}
	}

	/* Init arrays */
	uint8_t *src = NULL, *dst = NULL, *tmp = NULL;
	int row_stride = 0;
	if (imageType == GREY) {
		row_stride = width + 2;
		src = calloc((size_t)(height + 2) * (size_t)row_stride, sizeof(uint8_t));
		dst = calloc((size_t)(height + 2) * (size_t)row_stride, sizeof(uint8_t));
	} else if (imageType == RGB) {
		row_stride = width * 3 + 6;
		src = calloc((size_t)(height + 2) * (size_t)row_stride, sizeof(uint8_t));
		dst = calloc((size_t)(height + 2) * (size_t)row_stride, sizeof(uint8_t));
	}
	if (src == NULL || dst == NULL) {
		fprintf(stderr, "%s: Not enough memory\n", argv[0]);
		return EXIT_FAILURE;
	}

	/* Read input file */
	FILE *fh = fopen(image, "rb");
	if (fh == NULL) {
		fprintf(stderr, "%s: Cannot open input file %s\n", argv[0], image);
		return EXIT_FAILURE;
	}
	if (imageType == GREY) {
		for (i = 1 ; i <= height ; i++) {
			uint8_t *row = offset(src, i, 1, row_stride);
			if (fread(row, 1, (size_t)width, fh) != (size_t)width) {
				fprintf(stderr, "%s: Read error\n", argv[0]);
				fclose(fh);
				return EXIT_FAILURE;
			}
		}
	} else if (imageType == RGB) {
		for (i = 1 ; i <= height ; i++) {
			uint8_t *row = offset(src, i, 3, row_stride);
			if (fread(row, 1, (size_t)width * 3, fh) != (size_t)width * 3) {
				fprintf(stderr, "%s: Read error\n", argv[0]);
				fclose(fh);
				return EXIT_FAILURE;
			}
		}
	}
	fclose(fh);

	/* Convolute "loops" times */
	clock_t start = clock();
	for (t = 0 ; t < loops ; t++) {
		convolute(src, dst, 1, height, 1, width, width, height, h, imageType);
		tmp = src;
		src = dst;
		dst = tmp;
	}
	timer = (double)(clock() - start) / CLOCKS_PER_SEC;

	/* Write output file */
	size_t out_len = strlen(image) + 6;
	char *outImage = malloc(out_len);
	if (outImage == NULL) {
		fprintf(stderr, "%s: Not enough memory\n", argv[0]);
		return EXIT_FAILURE;
	}
	snprintf(outImage, out_len, "blur_%s", image);
	FILE *outFile = fopen(outImage, "wb");
	if (outFile == NULL) {
		fprintf(stderr, "%s: Cannot open output file %s\n", argv[0], outImage);
		return EXIT_FAILURE;
	}
	if (imageType == GREY) {
		for (i = 1 ; i <= height ; i++) {
			uint8_t *row = offset(src, i, 1, row_stride);
			if (fwrite(row, 1, (size_t)width, outFile) != (size_t)width) {
				fprintf(stderr, "%s: Write error\n", argv[0]);
				fclose(outFile);
				return EXIT_FAILURE;
			}
		}
	} else if (imageType == RGB) {
		for (i = 1 ; i <= height ; i++) {
			uint8_t *row = offset(src, i, 3, row_stride);
			if (fwrite(row, 1, (size_t)width * 3, outFile) != (size_t)width * 3) {
				fprintf(stderr, "%s: Write error\n", argv[0]);
				fclose(outFile);
				return EXIT_FAILURE;
			}
		}
	}
	fclose(outFile);

	printf("%f\n", timer);

	/* De-allocate space */
	free(src);
	free(dst);
	for (i = 0 ; i < 3 ; i++)
		free(h[i]);
	free(h);
	free(image);
	free(outImage);

	return EXIT_SUCCESS;
}

void convolute(uint8_t *src, uint8_t *dst, int row_from, int row_to, int col_from, int col_to, int width, int height, float** h, color_t imageType) {
	int i, j;
	if (imageType == GREY) {
		for (i = row_from ; i <= row_to ; i++)
			for (j = col_from ; j <= col_to ; j++)
				convolute_grey(src, dst, i, j, width+2, height, h);
	} else if (imageType == RGB) {
		for (i = row_from ; i <= row_to ; i++)
			for (j = col_from ; j <= col_to ; j++)
				convolute_rgb(src, dst, i, j*3, width*3+6, height, h);
	}
}

static inline void convolute_grey(uint8_t *src, uint8_t *dst, int x, int y, int width, int height, float** h) {
	const uint8_t *row0 = src + (x - 1) * width + (y - 1);
	const uint8_t *row1 = row0 + width;
	const uint8_t *row2 = row1 + width;
	const float *h0 = h[0];
	const float *h1 = h[1];
	const float *h2 = h[2];
	float val =
		row0[0] * h0[0] + row0[1] * h0[1] + row0[2] * h0[2] +
		row1[0] * h1[0] + row1[1] * h1[1] + row1[2] * h1[2] +
		row2[0] * h2[0] + row2[1] * h2[1] + row2[2] * h2[2];
	dst[width * x + y] = (uint8_t)val;
}

static inline void convolute_rgb(uint8_t *src, uint8_t *dst, int x, int y, int width, int height, float** h) {
	const uint8_t *row0 = src + (x - 1) * width + (y - 3);
	const uint8_t *row1 = row0 + width;
	const uint8_t *row2 = row1 + width;
	const float *h0 = h[0];
	const float *h1 = h[1];
	const float *h2 = h[2];
	float redval =
		row0[0] * h0[0] + row0[3] * h0[1] + row0[6] * h0[2] +
		row1[0] * h1[0] + row1[3] * h1[1] + row1[6] * h1[2] +
		row2[0] * h2[0] + row2[3] * h2[1] + row2[6] * h2[2];
	float greenval =
		row0[1] * h0[0] + row0[4] * h0[1] + row0[7] * h0[2] +
		row1[1] * h1[0] + row1[4] * h1[1] + row1[7] * h1[2] +
		row2[1] * h2[0] + row2[4] * h2[1] + row2[7] * h2[2];
	float blueval =
		row0[2] * h0[0] + row0[5] * h0[1] + row0[8] * h0[2] +
		row1[2] * h1[0] + row1[5] * h1[1] + row1[8] * h1[2] +
		row2[2] * h2[0] + row2[5] * h2[1] + row2[8] * h2[2];
	dst[width * x + y] = (uint8_t)redval;
	dst[width * x + y+1] = (uint8_t)greenval;
	dst[width * x + y+2] = (uint8_t)blueval;
}

/* Get pointer to internal array position */
uint8_t *offset(uint8_t *array, int i, int j, int width) {
	return &array[width * i + j];
}

void Usage(int argc, char **argv, char **image, int *width, int *height, int *loops, color_t *imageType) {
	if (argc == 6 && !strcmp(argv[5], "grey")) {
		*image = malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*loops = atoi(argv[4]);
		*imageType = GREY;
	} else if (argc == 6 && !strcmp(argv[5], "rgb")) {
		*image = malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(*image, argv[1]);
		*width = atoi(argv[2]);
		*height = atoi(argv[3]);
		*loops = atoi(argv[4]);
		*imageType = RGB;
	} else {
		fprintf(stderr, "\nError Input!\n%s image_name width height loops [rgb/grey].\n\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}
