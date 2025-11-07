#ifndef OPTIONS_H
#define OPTIONS_H

typedef enum { LAYOUT_ARRAY = 0, LAYOUT_LIST = 1 } Layout;
typedef enum { ALG_INSERTION = 0, ALG_BUBBLE = 1, ALG_MERGE = 2 } Algorithm;
typedef enum { TYPE_INT = 0, TYPE_DOUBLE = 1 } ValueType;

typedef struct {
    Layout    layout;
    Algorithm alg;
    ValueType type;
    const char *file_path;  /* defaults to "list.txt" */
    int        verify;      /* 0 off; 1 on */

    /* Output control */
    const char *out_path;   /* defaults to "output.txt" */
    int        write_output;/* 0 off; 1 on */
} Options;

/* Parse CLI + optional YAML (subset). Returns 0 on success, nonzero on error. */
int parse_options(int argc, char **argv, Options *o);

/* Emit the single summary line to stdout and (optionally) append to file. */
void emit_line(const Options *opt, const char *line);

#endif /* OPTIONS_H */
