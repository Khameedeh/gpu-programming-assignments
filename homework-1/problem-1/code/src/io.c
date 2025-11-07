#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "io.h"

static int read_size_from_file(FILE *fp, size_t *n_out) {
    unsigned long long n = 0ULL;
    if (fscanf(fp, "%llu", &n) != 1) return 1;
    if (n == 0ULL) return 1;
    *n_out = (size_t)n;
    return 0;
}

/* Arrays */

int read_int_array_from_file(const char *path, int32_t **arr_out, size_t *n_out, int64_t *checksum_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open '%s'\n", path); return 1; }

    size_t n = 0;
    if (read_size_from_file(fp, &n) != 0) { fprintf(stderr, "Error: failed to read N from '%s'\n", path); fclose(fp); return 1; }

    int32_t *a = (int32_t*)malloc(n * sizeof(int32_t));
    if (!a) { fprintf(stderr, "Error: out of memory for %zu ints\n", n); fclose(fp); return 1; }

    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        long long tmp;
        if (fscanf(fp, "%lld", &tmp) != 1) {
            fprintf(stderr, "Error: insufficient int values in '%s' (expected %zu)\n", path, n);
            free(a); fclose(fp); return 1;
        }
        a[i] = (int32_t)tmp;
        sum += a[i];
    }

    fclose(fp);
    *arr_out = a; *n_out = n; *checksum_out = sum;
    return 0;
}

int read_double_array_from_file(const char *path, double **arr_out, size_t *n_out, double *checksum_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open '%s'\n", path); return 1; }

    size_t n = 0;
    if (read_size_from_file(fp, &n) != 0) { fprintf(stderr, "Error: failed to read N from '%s'\n", path); fclose(fp); return 1; }

    double *a = (double*)malloc(n * sizeof(double));
    if (!a) { fprintf(stderr, "Error: out of memory for %zu doubles\n", n); fclose(fp); return 1; }

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double val;
        if (fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "Error: insufficient double values in '%s' (expected %zu)\n", path, n);
            free(a); fclose(fp); return 1;
        }
        a[i] = val;
        sum += val;
    }

    fclose(fp);
    *arr_out = a; *n_out = n; *checksum_out = sum;
    return 0;
}

/* Lists */

NodeInt* read_int_list_from_file(const char *path, size_t *n_out, int64_t *checksum_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open '%s'\n", path); return NULL; }

    size_t n = 0;
    if (read_size_from_file(fp, &n) != 0) { fprintf(stderr, "Error: failed to read N from '%s'\n", path); fclose(fp); return NULL; }

    NodeInt *head = NULL, *tail = NULL;
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        long long tmp;
        if (fscanf(fp, "%lld", &tmp) != 1) {
            fprintf(stderr, "Error: insufficient int values in '%s' (expected %zu)\n", path, n);
            NodeInt *p = head; while (p) { NodeInt *nx = p->next; free(p); p = nx; }
            fclose(fp); return NULL;
        }
        NodeInt *node = (NodeInt*)malloc(sizeof(NodeInt));
        if (!node) {
            fprintf(stderr, "Error: out of memory for list node\n");
            NodeInt *p = head; while (p) { NodeInt *nx = p->next; free(p); p = nx; }
            fclose(fp); return NULL;
        }
        node->v = (int32_t)tmp; node->next = NULL;
        sum += node->v;
        if (!head) head = tail = node; else { tail->next = node; tail = node; }
    }

    fclose(fp);
    *n_out = n; *checksum_out = sum;
    return head;
}

NodeDouble* read_double_list_from_file(const char *path, size_t *n_out, double *checksum_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open '%s'\n", path); return NULL; }

    size_t n = 0;
    if (read_size_from_file(fp, &n) != 0) { fprintf(stderr, "Error: failed to read N from '%s'\n", path); fclose(fp); return NULL; }

    NodeDouble *head = NULL, *tail = NULL;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double val;
        if (fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "Error: insufficient double values in '%s' (expected %zu)\n", path, n);
            NodeDouble *p = head; while (p) { NodeDouble *nx = p->next; free(p); p = nx; }
            fclose(fp); return NULL;
        }
        NodeDouble *node = (NodeDouble*)malloc(sizeof(NodeDouble));
        if (!node) {
            fprintf(stderr, "Error: out of memory for list node\n");
            NodeDouble *p = head; while (p) { NodeDouble *nx = p->next; free(p); p = nx; }
            fclose(fp); return NULL;
        }
        node->v = val; node->next = NULL;
        sum += val;
        if (!head) head = tail = node; else { tail->next = node; tail = node; }
    }

    fclose(fp);
    *n_out = n; *checksum_out = sum;
    return head;
}

/* Verify & cleanup */

int is_sorted_int_array(const int32_t *a, size_t n) {
    for (size_t i = 1; i < n; ++i) if (a[i-1] > a[i]) return 0; return 1;
}
int is_sorted_double_array(const double *a, size_t n) {
    for (size_t i = 1; i < n; ++i) if (a[i-1] > a[i]) return 0; return 1;
}
int is_sorted_int_list(const NodeInt *h) {
    if (!h) return 1; for (; h->next; h=h->next) if (h->v > h->next->v) return 0; return 1;
}
int is_sorted_double_list(const NodeDouble *h) {
    if (!h) return 1; for (; h->next; h=h->next) if (h->v > h->next->v) return 0; return 1;
}

void free_int_list(NodeInt *h) { while (h) { NodeInt *n = h->next; free(h); h = n; } }
void free_double_list(NodeDouble *h) { while (h) { NodeDouble *n = h->next; free(h); h = n; } }
