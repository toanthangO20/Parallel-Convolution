#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include "mpi.h"
#include "omp.h"

typedef enum {RGB, GREY} color_t;

void convolute(uint8_t *, uint8_t *, int, int, int, int, int, int, float**, color_t);
static inline void convolute_grey(uint8_t *, uint8_t *, int, int, int, int, float **);
static inline void convolute_rgb(uint8_t *, uint8_t *, int, int, int, int, float **);
void Usage(int, char **, char **, int *, int *, int *, color_t *);
uint8_t *offset(uint8_t *, int, int, int);
int divide_rows(int, int, int);


int main(int argc, char** argv) {
	int thread_count = 4;
	int fd, i, j, k, width, height, loops, t, row_div, col_div, rows, cols;
	double timer, remote_time;
	char *image;
	color_t imageType;
	/* MPI world topology */
    int process_id, num_processes;
	/* Find current task id */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	omp_set_dynamic(0);
	omp_set_num_threads(thread_count);
	/* MPI status */
    MPI_Status status;
	/* MPI data types */
    MPI_Datatype grey_col_type;
    MPI_Datatype rgb_col_type;
    MPI_Datatype grey_row_type;
    MPI_Datatype rgb_row_type;
	/* MPI requests */
    MPI_Request send_north_req;
    MPI_Request send_south_req;
    MPI_Request send_west_req;
    MPI_Request send_east_req;
    MPI_Request recv_north_req;
    MPI_Request recv_south_req;
    MPI_Request recv_west_req;
    MPI_Request recv_east_req;
    MPI_Request send_nw_req;
    MPI_Request send_ne_req;
    MPI_Request send_sw_req;
    MPI_Request send_se_req;
    MPI_Request recv_nw_req;
    MPI_Request recv_ne_req;
    MPI_Request recv_sw_req;
    MPI_Request recv_se_req;
	enum { TAG_N = 10, TAG_S = 11, TAG_W = 12, TAG_E = 13,
		   TAG_NW = 20, TAG_NE = 21, TAG_SW = 22, TAG_SE = 23 };
	
	/* Neighbours */
	int north = -1;
	int south = -1;
    int west = -1;
    int east = -1;

    /* Check arguments */
    if (process_id == 0) {
		Usage(argc, argv, &image, &width, &height, &loops, &imageType);
		/* Division of data in each process */
		row_div = divide_rows(height, width, num_processes);
		if (row_div <= 0 || height % row_div || num_processes % row_div || width % (col_div = num_processes / row_div)) {
				fprintf(stderr, "%s: Cannot divide to processes\n", argv[0]);
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				return EXIT_FAILURE;
		}
	}
	if (process_id != 0) {
		image = malloc((strlen(argv[1])+1) * sizeof(char));
		strcpy(image, argv[1]);
	}
	/* Broadcast parameters */
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageType, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&row_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	/* Compute number of rows per process */
	rows = height / row_div;
	cols = width / col_div;

	/* Create column data type for grey & rgb */
	MPI_Type_vector(rows, 1, cols+2, MPI_BYTE, &grey_col_type);
	MPI_Type_commit(&grey_col_type);
	MPI_Type_vector(rows, 3, 3*cols+6, MPI_BYTE, &rgb_col_type);
	MPI_Type_commit(&rgb_col_type);
	/* Create row data type */
	MPI_Type_contiguous(cols, MPI_BYTE, &grey_row_type);
	MPI_Type_commit(&grey_row_type);
	MPI_Type_contiguous(3*cols, MPI_BYTE, &rgb_row_type);
	MPI_Type_commit(&rgb_row_type);

	 /* Compute starting row and column */
    int start_row = (process_id / col_div) * rows;
    int start_col = (process_id % col_div) * cols;
	
	/* Init filters */
	int box_blur[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
	int gaussian_blur[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	int edge_detection[3][3] = {{1, 4, 1}, {4, 8, 4}, {1, 4, 1}};
	float **h = malloc(3 * sizeof(float *));
	for (i = 0 ; i < 3 ; i++)
		h[i] = malloc(3 * sizeof(float));
	for (i = 0 ; i < 3 ; i++) {
		for (j = 0 ; j < 3 ; j++){
			// h[i][j] = box_blur[i][j] / 9.0;
			h[i][j] = gaussian_blur[i][j] / 16.0;
			// h[i][j] = edge_detection[i][j] / 28.0;
		}
	}

	/* Init arrays */
	uint8_t *src = NULL, *dst = NULL, *tmpbuf = NULL, *tmp = NULL;
	MPI_File fh;
	int filesize, bufsize, nbytes;
	if (imageType == GREY) {
		filesize = width * height;
		bufsize = filesize / num_processes;
		nbytes = bufsize / sizeof(uint8_t);
		src = calloc((rows+2) * (cols+2), sizeof(uint8_t));
		dst = calloc((rows+2) * (cols+2), sizeof(uint8_t));
	} else if (imageType == RGB) {
		filesize = width*3 * height;
		bufsize = filesize / num_processes;
		nbytes = bufsize / sizeof(uint8_t);
		src = calloc((rows+2) * (cols*3+6), sizeof(uint8_t));
		dst = calloc((rows+2) * (cols*3+6), sizeof(uint8_t));
	}
	if (src == NULL || dst == NULL) {
        fprintf(stderr, "%s: Not enough memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
	}

	/* Parallel read */
	MPI_File_open(MPI_COMM_WORLD, image, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	if (imageType == GREY) {
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(fh, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			tmpbuf = offset(src, i, 1, cols+2);
			MPI_File_read(fh, tmpbuf, cols, MPI_BYTE, &status);
		}
	} else if (imageType == RGB) {
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(fh, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
			tmpbuf = offset(src, i, 3, cols*3+6);
			MPI_File_read(fh, tmpbuf, cols*3, MPI_BYTE, &status);
		}
	}
	MPI_File_close(&fh);

	/* Compute neighbours */
    if (start_row != 0)
        north = process_id - col_div;
    if (start_row + rows != height)
        south = process_id + col_div;
    if (start_col != 0)
        west = process_id - 1;
    if (start_col + cols != width)
        east = process_id + 1;
	int nw = (north != -1 && west != -1) ? process_id - col_div - 1 : -1;
	int ne = (north != -1 && east != -1) ? process_id - col_div + 1 : -1;
	int sw = (south != -1 && west != -1) ? process_id + col_div - 1 : -1;
	int se = (south != -1 && east != -1) ? process_id + col_div + 1 : -1;
	
	/* Get time before */
	MPI_Barrier(MPI_COMM_WORLD);
    timer = MPI_Wtime();
	/* Convolute "loops" times */
	for (t = 0 ; t < loops ; t++) {
        /* Send and request borders */
		if (imageType == GREY) {
			if (north != -1) {
				MPI_Isend(offset(src, 1, 1, cols+2), 1, grey_row_type, north, TAG_S, MPI_COMM_WORLD, &send_north_req);
				MPI_Irecv(offset(src, 0, 1, cols+2), 1, grey_row_type, north, TAG_N, MPI_COMM_WORLD, &recv_north_req);
			}
			if (west != -1) {
				MPI_Isend(offset(src, 1, 1, cols+2), 1, grey_col_type,  west, TAG_E, MPI_COMM_WORLD, &send_west_req);
				MPI_Irecv(offset(src, 1, 0, cols+2), 1, grey_col_type,  west, TAG_W, MPI_COMM_WORLD, &recv_west_req);
			}
			if (south != -1) {
				MPI_Isend(offset(src, rows, 1, cols+2), 1, grey_row_type, south, TAG_N, MPI_COMM_WORLD, &send_south_req);
				MPI_Irecv(offset(src, rows+1, 1, cols+2), 1, grey_row_type, south, TAG_S, MPI_COMM_WORLD, &recv_south_req);
			}
			if (east != -1) {
				MPI_Isend(offset(src, 1, cols, cols+2), 1, grey_col_type,  east, TAG_W, MPI_COMM_WORLD, &send_east_req);
				MPI_Irecv(offset(src, 1, cols+1, cols+2), 1, grey_col_type,  east, TAG_E, MPI_COMM_WORLD, &recv_east_req);
			}
			if (nw != -1) {
				MPI_Isend(offset(src, 1, 1, cols+2), 1, MPI_BYTE, nw, TAG_SE, MPI_COMM_WORLD, &send_nw_req);
				MPI_Irecv(offset(src, 0, 0, cols+2), 1, MPI_BYTE, nw, TAG_NW, MPI_COMM_WORLD, &recv_nw_req);
			}
			if (ne != -1) {
				MPI_Isend(offset(src, 1, cols, cols+2), 1, MPI_BYTE, ne, TAG_SW, MPI_COMM_WORLD, &send_ne_req);
				MPI_Irecv(offset(src, 0, cols+1, cols+2), 1, MPI_BYTE, ne, TAG_NE, MPI_COMM_WORLD, &recv_ne_req);
			}
			if (sw != -1) {
				MPI_Isend(offset(src, rows, 1, cols+2), 1, MPI_BYTE, sw, TAG_NE, MPI_COMM_WORLD, &send_sw_req);
				MPI_Irecv(offset(src, rows+1, 0, cols+2), 1, MPI_BYTE, sw, TAG_SW, MPI_COMM_WORLD, &recv_sw_req);
			}
			if (se != -1) {
				MPI_Isend(offset(src, rows, cols, cols+2), 1, MPI_BYTE, se, TAG_NW, MPI_COMM_WORLD, &send_se_req);
				MPI_Irecv(offset(src, rows+1, cols+1, cols+2), 1, MPI_BYTE, se, TAG_SE, MPI_COMM_WORLD, &recv_se_req);
			}
		} else if (imageType == RGB) {
			if (north != -1) {
				MPI_Isend(offset(src, 1, 3, 3*cols+6), 1, rgb_row_type, north, TAG_S, MPI_COMM_WORLD, &send_north_req);
				MPI_Irecv(offset(src, 0, 3, 3*cols+6), 1, rgb_row_type, north, TAG_N, MPI_COMM_WORLD, &recv_north_req);
			}
			if (west != -1) {
				MPI_Isend(offset(src, 1, 3, 3*cols+6), 1, rgb_col_type,  west, TAG_E, MPI_COMM_WORLD, &send_west_req);
				MPI_Irecv(offset(src, 1, 0, 3*cols+6), 1, rgb_col_type,  west, TAG_W, MPI_COMM_WORLD, &recv_west_req);
			}
			if (south != -1) {
				MPI_Isend(offset(src, rows, 3, 3*cols+6), 1, rgb_row_type, south, TAG_N, MPI_COMM_WORLD, &send_south_req);
				MPI_Irecv(offset(src, rows+1, 3, 3*cols+6), 1, rgb_row_type, south, TAG_S, MPI_COMM_WORLD, &recv_south_req);
			}
			if (east != -1) {
				MPI_Isend(offset(src, 1, 3*cols, 3*cols+6), 1, rgb_col_type,  east, TAG_W, MPI_COMM_WORLD, &send_east_req);
				MPI_Irecv(offset(src, 1, 3*cols+3, 3*cols+6), 1, rgb_col_type,  east, TAG_E, MPI_COMM_WORLD, &recv_east_req);
			}
			if (nw != -1) {
				MPI_Isend(offset(src, 1, 3, 3*cols+6), 3, MPI_BYTE, nw, TAG_SE, MPI_COMM_WORLD, &send_nw_req);
				MPI_Irecv(offset(src, 0, 0, 3*cols+6), 3, MPI_BYTE, nw, TAG_NW, MPI_COMM_WORLD, &recv_nw_req);
			}
			if (ne != -1) {
				MPI_Isend(offset(src, 1, 3*cols, 3*cols+6), 3, MPI_BYTE, ne, TAG_SW, MPI_COMM_WORLD, &send_ne_req);
				MPI_Irecv(offset(src, 0, 3*cols+3, 3*cols+6), 3, MPI_BYTE, ne, TAG_NE, MPI_COMM_WORLD, &recv_ne_req);
			}
			if (sw != -1) {
				MPI_Isend(offset(src, rows, 3, 3*cols+6), 3, MPI_BYTE, sw, TAG_NE, MPI_COMM_WORLD, &send_sw_req);
				MPI_Irecv(offset(src, rows+1, 0, 3*cols+6), 3, MPI_BYTE, sw, TAG_SW, MPI_COMM_WORLD, &recv_sw_req);
			}
			if (se != -1) {
				MPI_Isend(offset(src, rows, 3*cols, 3*cols+6), 3, MPI_BYTE, se, TAG_NW, MPI_COMM_WORLD, &send_se_req);
				MPI_Irecv(offset(src, rows+1, 3*cols+3, 3*cols+6), 3, MPI_BYTE, se, TAG_SE, MPI_COMM_WORLD, &recv_se_req);
			}
		}

		/* Inner Data Convolute */
		if (rows >= 3 && cols >= 3)
			convolute(src, dst, 2, rows-1, 2, cols-1, cols, rows, h, imageType);

		/* Wait for all receives, then compute boundary */
		{
			MPI_Request recv_reqs[8];
			MPI_Status recv_stats[8];
			int recv_count = 0;
			if (north != -1) recv_reqs[recv_count++] = recv_north_req;
			if (south != -1) recv_reqs[recv_count++] = recv_south_req;
			if (west != -1)  recv_reqs[recv_count++] = recv_west_req;
			if (east != -1)  recv_reqs[recv_count++] = recv_east_req;
			if (nw != -1)    recv_reqs[recv_count++] = recv_nw_req;
			if (ne != -1)    recv_reqs[recv_count++] = recv_ne_req;
			if (sw != -1)    recv_reqs[recv_count++] = recv_sw_req;
			if (se != -1)    recv_reqs[recv_count++] = recv_se_req;
			MPI_Waitall(recv_count, recv_reqs, recv_stats);
		}

		if (cols > 0 && rows > 0)
			convolute(src, dst, 1, 1, 1, cols, cols, rows, h, imageType);
		if (cols > 0 && rows > 1)
			convolute(src, dst, rows, rows, 1, cols, cols, rows, h, imageType);
		if (cols > 0 && rows > 2)
			convolute(src, dst, 2, rows-1, 1, 1, cols, rows, h, imageType);
		if (cols > 1 && rows > 2)
			convolute(src, dst, 2, rows-1, cols, cols, cols, rows, h, imageType);

		/* Wait to have sent all borders */
		{
			MPI_Request send_reqs[8];
			MPI_Status send_stats[8];
			int send_count = 0;
			if (north != -1) send_reqs[send_count++] = send_north_req;
			if (south != -1) send_reqs[send_count++] = send_south_req;
			if (west != -1)  send_reqs[send_count++] = send_west_req;
			if (east != -1)  send_reqs[send_count++] = send_east_req;
			if (nw != -1)    send_reqs[send_count++] = send_nw_req;
			if (ne != -1)    send_reqs[send_count++] = send_ne_req;
			if (sw != -1)    send_reqs[send_count++] = send_sw_req;
			if (se != -1)    send_reqs[send_count++] = send_se_req;
			MPI_Waitall(send_count, send_reqs, send_stats);
		}

		/* swap arrays */
		tmp = src;
	    src = dst;
	    dst = tmp;
	}
	/* Get time elapsed */
    timer = MPI_Wtime() - timer;

	/* Parallel write */
	char *outImage = malloc((strlen(image) + 9) * sizeof(char));
	strcpy(outImage, "blur_");
	strcat(outImage, image);
	MPI_File outFile;
	MPI_File_open(MPI_COMM_WORLD, outImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);
	if (imageType == GREY) {
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			tmpbuf = offset(src, i, 1, cols+2);
			MPI_File_write(outFile, tmpbuf, cols, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	} else if (imageType == RGB) {
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
			tmpbuf = offset(src, i, 3, cols*3+6);
			MPI_File_write(outFile, tmpbuf, cols*3, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	}
	MPI_File_close(&outFile);

	/* Get times from other processes and print maximum */
    if (process_id != 0)
        MPI_Send(&timer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else {
        for (i = 1 ; i != num_processes ; ++i) {
            MPI_Recv(&remote_time, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            if (remote_time > timer)
                timer = remote_time;
        }
        printf("%f\n", timer);
    }

    /* De-allocate space */
    free(src);
    free(dst);
    MPI_Type_free(&rgb_col_type);
    MPI_Type_free(&rgb_row_type);
    MPI_Type_free(&grey_col_type);
    MPI_Type_free(&grey_row_type);

	/* Finalize and exit */
    MPI_Finalize();
	return EXIT_SUCCESS;
}

void convolute(uint8_t *src, uint8_t *dst, int row_from, int row_to, int col_from, int col_to, int width, int height, float** h, color_t imageType) {
	int i, j;
	if (imageType == GREY) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(2)
		for (i = row_from ; i <= row_to ; i++)
			for (j = col_from ; j <= col_to ; j++)
				convolute_grey(src, dst, i, j, width+2, height, h);
	} else if (imageType == RGB) {
#pragma omp parallel for shared(src, dst) schedule(static) collapse(2)
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
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		fprintf(stderr, "Error Input!\n%s image_name width height loops [rgb/grey].\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}

/* Divide rows and columns in a way to minimize perimeter of blocks */
int divide_rows(int rows, int cols, int workers) {
    int per, rows_to, cols_to, best = 0;
    int per_min = rows + cols + 1;
    for (rows_to = 1 ; rows_to <= workers ; ++rows_to) {
        if (workers % rows_to || rows % rows_to) continue;
        cols_to = workers / rows_to;
        if (cols % cols_to) continue;
        per = rows / rows_to + cols / cols_to;
        if (per < per_min) {
            per_min = per;
            best = rows_to;
        }
    }
    return best;
}
