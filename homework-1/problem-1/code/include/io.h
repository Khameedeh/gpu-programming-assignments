#ifndef IO_H
#define IO_H

#include <stddef.h>
#include <stdint.h>
#include "sort.h"

/* File readers (first number is N, then N values) */
int read_int_array_from_file(const char *path, int32_t **arr_out, size_t *n_out, int64_t *checksum_out);
int read_double_array_from_file(const char *path, double **arr_out, size_t *n_out, double *checksum_out);

NodeInt*    read_int_list_from_file(const char *path, size_t *n_out, int64_t *checksum_out);
NodeDouble* read_double_list_from_file(const char *path, size_t *n_out, double *checksum_out);

/* Verify helpers */
int is_sorted_int_array(const int32_t *a, size_t n);
int is_sorted_double_array(const double *a, size_t n);
int is_sorted_int_list(const NodeInt *h);
int is_sorted_double_list(const NodeDouble *h);

/* Cleanup */
void free_int_list(NodeInt *h);
void free_double_list(NodeDouble *h);

#endif /* IO_H */
