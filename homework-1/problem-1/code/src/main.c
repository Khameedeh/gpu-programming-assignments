#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "options.h"
#include "io.h"
#include "sort.h"

/* ---------- helpers to write sequences to a FILE* ---------- */

static void fprint_int_array_line(FILE *fp, const int32_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        fprintf(fp, "%d", a[i]);
        fputc((i + 1 < n) ? ' ' : '\n', fp);
    }
}

static void fprint_double_array_line(FILE *fp, const double *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        fprintf(fp, "%.17g", a[i]);
        fputc((i + 1 < n) ? ' ' : '\n', fp);
    }
}

static void fprint_int_list_line(FILE *fp, const NodeInt *head) {
    const NodeInt *p = head;
    while (p) {
        fprintf(fp, "%d", p->v);
        p = p->next;
        fputc(p ? ' ' : '\n', fp);
    }
}

static void fprint_double_list_line(FILE *fp, const NodeDouble *head) {
    const NodeDouble *p = head;
    while (p) {
        fprintf(fp, "%.17g", p->v);
        p = p->next;
        fputc(p ? ' ' : '\n', fp);
    }
}

/* ---------- file output: metadata + sequence ---------- */

static void write_file_header_and_sequence_int_array(
    const Options *opt, const char *meta, const int32_t *a, size_t n)
{
    if (!opt->write_output) return;
    const char *path = (opt->out_path && opt->out_path[0]) ? opt->out_path : "output.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open '%s' for writing; skipping output file.\n", path);
        return;
    }
    fprintf(fp, "%s\n", meta);            /* first line: metadata */
    fprint_int_array_line(fp, a, n);      /* second line: sorted sequence */
    fclose(fp);
}

static void write_file_header_and_sequence_double_array(
    const Options *opt, const char *meta, const double *a, size_t n)
{
    if (!opt->write_output) return;
    const char *path = (opt->out_path && opt->out_path[0]) ? opt->out_path : "output.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open '%s' for writing; skipping output file.\n", path);
        return;
    }
    fprintf(fp, "%s\n", meta);
    fprint_double_array_line(fp, a, n);
    fclose(fp);
}

static void write_file_header_and_sequence_int_list(
    const Options *opt, const char *meta, const NodeInt *head)
{
    if (!opt->write_output) return;
    const char *path = (opt->out_path && opt->out_path[0]) ? opt->out_path : "output.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open '%s' for writing; skipping output file.\n", path);
        return;
    }
    fprintf(fp, "%s\n", meta);
    fprint_int_list_line(fp, head);
    fclose(fp);
}

static void write_file_header_and_sequence_double_list(
    const Options *opt, const char *meta, const NodeDouble *head)
{
    if (!opt->write_output) return;
    const char *path = (opt->out_path && opt->out_path[0]) ? opt->out_path : "output.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open '%s' for writing; skipping output file.\n", path);
        return;
    }
    fprintf(fp, "%s\n", meta);
    fprint_double_list_line(fp, head);
    fclose(fp);
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    Options opt;
    if (parse_options(argc, argv, &opt) != 0) return 1;

    const char *layout_s = (opt.layout==LAYOUT_ARRAY?"array":"list");
    const char *alg_s    = (opt.alg==ALG_INSERTION?"insertion":opt.alg==ALG_BUBBLE?"bubble":"merge");
    const char *type_s   = (opt.type==TYPE_INT?"int":"double");

    if (opt.type == TYPE_INT) {
        if (opt.layout == LAYOUT_ARRAY) {
            int32_t *a = NULL; size_t n = 0; int64_t checksum = 0;
            if (read_int_array_from_file(opt.file_path, &a, &n, &checksum) != 0) return 1;

            if (opt.alg == ALG_INSERTION)      insertion_sort_int(a, n);
            else if (opt.alg == ALG_BUBBLE)    bubble_sort_int(a, n);
            else                               mergesort_int(a, n);

            int sorted_flag = opt.verify ? is_sorted_int_array(a, n) : 0;
            if (opt.verify && !sorted_flag) {
                fprintf(stderr, "Warning: verify failed; sequence is not sorted.\n");
            }

            int32_t first = (n ? a[0] : 0);
            char meta[512];
            if (opt.verify)
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%lld first=%d sorted=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, (long long)checksum, first, sorted_flag);
            else
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%lld first=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, (long long)checksum, first);

            /* stdout: metadata only */
            puts(meta);

            /* file (if requested): metadata + sequence */
            write_file_header_and_sequence_int_array(&opt, meta, a, n);

            free(a);
        } else { /* LIST + INT */
            size_t n = 0; int64_t checksum = 0;
            NodeInt *head = read_int_list_from_file(opt.file_path, &n, &checksum);
            if (!head && n == 0) return 1;

            if (opt.alg == ALG_INSERTION)      head = list_insertion_sort_int(head);
            else if (opt.alg == ALG_BUBBLE)    list_bubble_sort_int(head);
            else                               head = list_mergesort_int(head);

            int sorted_flag = opt.verify ? is_sorted_int_list(head) : 0;
            if (opt.verify && !sorted_flag) {
                fprintf(stderr, "Warning: verify failed; sequence is not sorted.\n");
            }

            int32_t first = head ? head->v : 0;
            char meta[512];
            if (opt.verify)
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%lld first=%d sorted=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, (long long)checksum, first, sorted_flag);
            else
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%lld first=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, (long long)checksum, first);

            puts(meta);
            write_file_header_and_sequence_int_list(&opt, meta, head);

            free_int_list(head);
        }
    } else { /* TYPE_DOUBLE */
        if (opt.layout == LAYOUT_ARRAY) {
            double *a = NULL; size_t n = 0; double checksum = 0.0;
            if (read_double_array_from_file(opt.file_path, &a, &n, &checksum) != 0) return 1;

            if (opt.alg == ALG_INSERTION)      insertion_sort_double(a, n);
            else if (opt.alg == ALG_BUBBLE)    bubble_sort_double(a, n);
            else                               mergesort_double(a, n);

            int sorted_flag = opt.verify ? is_sorted_double_array(a, n) : 0;
            if (opt.verify && !sorted_flag) {
                fprintf(stderr, "Warning: verify failed; sequence is not sorted.\n");
            }

            double first = (n ? a[0] : 0.0);
            char meta[512];
            if (opt.verify)
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%.17g first=%.17g sorted=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, checksum, first, sorted_flag);
            else
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%.17g first=%.17g",
                    opt.file_path, layout_s, alg_s, type_s, n, checksum, first);

            puts(meta);
            write_file_header_and_sequence_double_array(&opt, meta, a, n);

            free(a);
        } else { /* LIST + DOUBLE */
            size_t n = 0; double checksum = 0.0;
            NodeDouble *head = read_double_list_from_file(opt.file_path, &n, &checksum);
            if (!head && n == 0) return 1;

            if (opt.alg == ALG_INSERTION)      head = list_insertion_sort_double(head);
            else if (opt.alg == ALG_BUBBLE)    list_bubble_sort_double(head);
            else                               head = list_mergesort_double(head);

            int sorted_flag = opt.verify ? is_sorted_double_list(head) : 0;
            if (opt.verify && !sorted_flag) {
                fprintf(stderr, "Warning: verify failed; sequence is not sorted.\n");
            }

            double first = head ? head->v : 0.0;
            char meta[512];
            if (opt.verify)
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%.17g first=%.17g sorted=%d",
                    opt.file_path, layout_s, alg_s, type_s, n, checksum, first, sorted_flag);
            else
                snprintf(meta, sizeof meta,
                    "file=%s layout=%s alg=%s type=%s n=%zu checksum=%.17g first=%.17g",
                    opt.file_path, layout_s, alg_s, type_s, n, checksum, first);

            puts(meta);
            write_file_header_and_sequence_double_list(&opt, meta, head);

            free_double_list(head);
        }
    }

    return 0;
}
