#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "options.h"

/* --------- small helpers --------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -layout {array|list} -alg {insertion|bubble|merge} -t {int|double}\n"
        "           [-file <path>] [--verify]\n"
        "           [-config <config.yaml>] [-o <output.txt>] [--write-output]\n",
        prog);
}

static const char* lc_copy(char *dst, size_t dstsz, const char *src) {
    size_t len = strlen(src);
    if (len >= dstsz) len = dstsz - 1;
    for (size_t i = 0; i < len; ++i) dst[i] = (char)tolower((unsigned char)src[i]);
    dst[len] = '\0';
    return dst;
}

static int parse_layout(const char *s, Layout *out) {
    if (strcmp(s, "array") == 0) { *out = LAYOUT_ARRAY; return 0; }
    if (strcmp(s, "list")  == 0) { *out = LAYOUT_LIST;  return 0; }
    return 1;
}
static int parse_alg(const char *s, Algorithm *out) {
    if (strcmp(s, "insertion") == 0) { *out = ALG_INSERTION; return 0; }
    if (strcmp(s, "bubble")    == 0) { *out = ALG_BUBBLE;    return 0; }
    if (strcmp(s, "merge")     == 0) { *out = ALG_MERGE;     return 0; }
    return 1;
}
static int parse_type(const char *s, ValueType *out) {
    if (strcmp(s, "int")    == 0) { *out = TYPE_INT;    return 0; }
    if (strcmp(s, "double") == 0) { *out = TYPE_DOUBLE; return 0; }
    return 1;
}

static char *dup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char*)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static char *ltrim(char *s) { while (*s && isspace((unsigned char)*s)) ++s; return s; }
static void rtrim_inplace(char *s) { size_t n = strlen(s); while (n && isspace((unsigned char)s[n-1])) s[--n] = '\0'; }
static void trim_inplace(char *s) { char *p = ltrim(s); if (p != s) memmove(s, p, strlen(p)+1); rtrim_inplace(s); }

static void strip_inline_comment(char *s) {
    int in_single = 0, in_double = 0;
    for (size_t i = 0; s[i]; ++i) {
        if (s[i] == '\'' && !in_double) in_single = !in_single;
        else if (s[i] == '\"' && !in_single) in_double = !in_double;
        else if (s[i] == '#' && !in_single && !in_double) { s[i] = '\0'; break; }
    }
}
static void unquote_inplace(char *s) {
    size_t n = strlen(s);
    if (n >= 2) {
        if ((s[0] == '\"' && s[n-1] == '\"') || (s[0] == '\'' && s[n-1] == '\'')) {
            memmove(s, s+1, n-2);
            s[n-2] = '\0';
        }
    }
}
static int str_ieq(const char *a, const char *b) {
    for (; *a && *b; ++a, ++b) {
        int ca = tolower((unsigned char)*a);
        int cb = tolower((unsigned char)*b);
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}
static int parse_bool(const char *s, int *out) {
    if (str_ieq(s, "true") || str_ieq(s, "yes") || str_ieq(s, "on")  || str_ieq(s, "1")) { *out = 1; return 0; }
    if (str_ieq(s, "false")|| str_ieq(s, "no")  || str_ieq(s, "off") || str_ieq(s, "0")) { *out = 0; return 0; }
    return 1;
}

/* --------- YAML (tiny subset) --------- */

typedef struct {
    int has_layout, has_alg, has_type, has_file, has_verify, has_out, has_write;
} ConfigMask;

static int load_config_yaml(const char *path, Options *o, ConfigMask *mask, char *err, size_t errsz) {
    if (!path) return 1;
    FILE *fp = fopen(path, "r");
    if (!fp) { if (err) snprintf(err, errsz, "cannot open '%s'", path); return 1; }

    char line[4096];
    int ln = 0;
    while (fgets(line, sizeof line, fp)) {
        ++ln;
        strip_inline_comment(line);
        trim_inplace(line);
        if (line[0] == '\0') continue;
        if (strncmp(line, "---", 3) == 0) continue;

        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = '\0';
        char *k = line; trim_inplace(k);
        char *v = colon + 1; trim_inplace(v);
        unquote_inplace(v);

        char kbuf[64]; lc_copy(kbuf, sizeof kbuf, k);
        char vbuf[256]; lc_copy(vbuf, sizeof vbuf, v);

        if (strcmp(kbuf, "layout") == 0) {
            if (parse_layout(vbuf, &o->layout) == 0) mask->has_layout = 1;
            else if (err) snprintf(err, errsz, "line %d: invalid layout '%s'", ln, v);
        } else if (strcmp(kbuf, "alg") == 0 || strcmp(kbuf, "algorithm") == 0) {
            if (parse_alg(vbuf, &o->alg) == 0) mask->has_alg = 1;
            else if (err) snprintf(err, errsz, "line %d: invalid alg '%s'", ln, v);
        } else if (strcmp(kbuf, "t") == 0 || strcmp(kbuf, "type") == 0) {
            if (parse_type(vbuf, &o->type) == 0) mask->has_type = 1;
            else if (err) snprintf(err, errsz, "line %d: invalid type '%s'", ln, v);
        } else if (strcmp(kbuf, "file") == 0 || strcmp(kbuf, "path") == 0) {
            o->file_path = dup_cstr(v);
            mask->has_file = 1;
        } else if (strcmp(kbuf, "verify") == 0) {
            int b = 0; if (parse_bool(vbuf, &b) == 0) { o->verify = b; mask->has_verify = 1; }
            else if (err) snprintf(err, errsz, "line %d: invalid verify '%s'", ln, v);
        } else if (strcmp(kbuf, "output") == 0 || strcmp(kbuf, "out") == 0 || strcmp(kbuf, "output_path") == 0) {
            o->out_path = dup_cstr(v);
            mask->has_out = 1;
        } else if (strcmp(kbuf, "write_output") == 0) {
            int b = 0; if (parse_bool(vbuf, &b) == 0) { o->write_output = b; mask->has_write = 1; }
            else if (err) snprintf(err, errsz, "line %d: invalid write_output '%s'", ln, v);
        } else {
            /* ignore unknown keys */
        }
    }
    fclose(fp);
    return 0;
}

/* --------- public API --------- */

static void set_defaults(Options *o) {
    o->layout = LAYOUT_ARRAY;
    o->alg    = ALG_INSERTION;
    o->type   = TYPE_INT;
    o->file_path = "list.txt";
    o->verify = 0;
    o->out_path = "output.txt";
    o->write_output = 0;
}

int parse_options(int argc, char **argv, Options *o) {
    set_defaults(o);

    /* First pass: optional YAML */
    const char *config_path = NULL;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-config") == 0 || strcmp(argv[i], "--config") == 0) && i + 1 < argc) {
            config_path = argv[i+1];
            break;
        }
    }
    ConfigMask mask = {0,0,0,0,0,0,0};
    char err[128] = {0};
    if (config_path) {
        if (load_config_yaml(config_path, o, &mask, err, sizeof err) != 0 && err[0]) {
            fprintf(stderr, "Warning: config load failed: %s\n", err);
        }
    }

    /* Second pass: CLI overrides */
    int saw_layout = mask.has_layout;
    int saw_alg    = mask.has_alg;
    int saw_type   = mask.has_type;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if ((strcmp(arg, "-layout") == 0) && i + 1 < argc) {
            char buf[16]; lc_copy(buf, sizeof buf, argv[++i]);
            if (parse_layout(buf, &o->layout) != 0) { fprintf(stderr, "Error: bad -layout.\n"); return 1; }
            saw_layout = 1;
        } else if ((strcmp(arg, "-alg") == 0) && i + 1 < argc) {
            char buf[16]; lc_copy(buf, sizeof buf, argv[++i]);
            if (parse_alg(buf, &o->alg) != 0) { fprintf(stderr, "Error: bad -alg.\n"); return 1; }
            saw_alg = 1;
        } else if ((strcmp(arg, "-t") == 0 || strcmp(arg, "--type") == 0) && i + 1 < argc) {
            char buf[16]; lc_copy(buf, sizeof buf, argv[++i]);
            if (parse_type(buf, &o->type) != 0) { fprintf(stderr, "Error: bad -t.\n"); return 1; }
            saw_type = 1;
        } else if ((strcmp(arg, "-file") == 0) && i + 1 < argc) {
            const char *p = argv[++i];
            o->file_path = dup_cstr(p);
        } else if (strcmp(arg, "--verify") == 0) {
            o->verify = 1;
        } else if ((strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) && i + 1 < argc) {
            const char *p = argv[++i];
            o->out_path = dup_cstr(p);
        } else if (strcmp(arg, "--write-output") == 0) {
            o->write_output = 1;
        } else if ((strcmp(arg, "-config") == 0 || strcmp(arg, "--config") == 0) && i + 1 < argc) {
            ++i; /* already handled */
        } else {
            /* ignore unknowns to keep interface minimal */
        }
    }

    if (!saw_layout || !saw_alg || !saw_type) {
        usage(argv[0]);
        return 1;
    }
    return 0;
}

void emit_line(const Options *opt, const char *line) {
    fputs(line, stdout);
    fputc('\n', stdout);

    if (opt->write_output) {
        const char *path = (opt->out_path && opt->out_path[0]) ? opt->out_path : "output.txt";
        FILE *fp = fopen(path, "a");
        if (!fp) {
            fprintf(stderr, "Warning: cannot open '%s' for append; skipping output file.\n", path);
            return;
        }
        fputs(line, fp);
        fputc('\n', fp);
        fclose(fp);
    }
}
