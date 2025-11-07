#include <stdio.h>
#include <stdlib.h>
#include "sort.h"

/* ---------------- Array sorts ---------------- */

void insertion_sort_int(int32_t *a, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        int32_t key = a[i];
        size_t j = i;
        while (j > 0 && a[j-1] > key) { a[j] = a[j-1]; --j; }
        a[j] = key;
    }
}

void insertion_sort_double(double *a, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        double key = a[i];
        size_t j = i;
        while (j > 0 && a[j-1] > key) { a[j] = a[j-1]; --j; }
        a[j] = key;
    }
}

void bubble_sort_int(int32_t *a, size_t n) {
    if (n < 2) return;
    int swapped;
    do {
        swapped = 0;
        for (size_t i = 1; i < n; ++i) {
            if (a[i-1] > a[i]) { int32_t t=a[i]; a[i]=a[i-1]; a[i-1]=t; swapped=1; }
        }
        --n;
    } while (swapped);
}

void bubble_sort_double(double *a, size_t n) {
    if (n < 2) return;
    int swapped;
    do {
        swapped = 0;
        for (size_t i = 1; i < n; ++i) {
            if (a[i-1] > a[i]) { double t=a[i]; a[i]=a[i-1]; a[i-1]=t; swapped=1; }
        }
        --n;
    } while (swapped);
}

/* mergesort (int) */

static void merge_int(int32_t *a, int32_t *tmp, long lo, long mid, long hi) {
    long i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) tmp[k++] = (a[i] <= a[j] ? a[i++] : a[j++]);
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= hi)  tmp[k++] = a[j++];
    for (long t = lo; t <= hi; ++t) a[t] = tmp[t];
}

static void mergesort_int_rec(int32_t *a, int32_t *tmp, long lo, long hi) {
    if (lo >= hi) return;
    long mid = lo + (hi - lo) / 2;
    mergesort_int_rec(a, tmp, lo, mid);
    mergesort_int_rec(a, tmp, mid+1, hi);
    merge_int(a, tmp, lo, mid, hi);
}

void mergesort_int(int32_t *a, size_t n) {
    if (n < 2) return;
    int32_t *tmp = (int32_t*)malloc(n * sizeof(int32_t));
    if (!tmp) { fprintf(stderr, "Error: out of memory.\n"); exit(1); }
    mergesort_int_rec(a, tmp, 0, (long)n - 1);
    free(tmp);
}

/* mergesort (double) */

static void merge_double(double *a, double *tmp, long lo, long mid, long hi) {
    long i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) tmp[k++] = (a[i] <= a[j] ? a[i++] : a[j++]);
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= hi)  tmp[k++] = a[j++];
    for (long t = lo; t <= hi; ++t) a[t] = tmp[t];
}

static void mergesort_double_rec(double *a, double *tmp, long lo, long hi) {
    if (lo >= hi) return;
    long mid = lo + (hi - lo) / 2;
    mergesort_double_rec(a, tmp, lo, mid);
    mergesort_double_rec(a, tmp, mid+1, hi);
    merge_double(a, tmp, lo, mid, hi);
}

void mergesort_double(double *a, size_t n) {
    if (n < 2) return;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) { fprintf(stderr, "Error: out of memory.\n"); exit(1); }
    mergesort_double_rec(a, tmp, 0, (long)n - 1);
    free(tmp);
}

/* ---------------- Linked-list sorts ---------------- */

NodeInt* list_insertion_sort_int(NodeInt *head) {
    NodeInt *sorted = NULL;
    while (head) {
        NodeInt *cur = head; head = head->next;
        if (!sorted || cur->v <= sorted->v) { cur->next = sorted; sorted = cur; }
        else {
            NodeInt *p = sorted;
            while (p->next && p->next->v < cur->v) p = p->next;
            cur->next = p->next; p->next = cur;
        }
    }
    return sorted;
}

NodeDouble* list_insertion_sort_double(NodeDouble *head) {
    NodeDouble *sorted = NULL;
    while (head) {
        NodeDouble *cur = head; head = head->next;
        if (!sorted || cur->v <= sorted->v) { cur->next = sorted; sorted = cur; }
        else {
            NodeDouble *p = sorted;
            while (p->next && p->next->v < cur->v) p = p->next;
            cur->next = p->next; p->next = cur;
        }
    }
    return sorted;
}

void list_bubble_sort_int(NodeInt *head) {
    if (!head) return;
    int swapped;
    do {
        swapped = 0;
        for (NodeInt *p = head; p && p->next; p = p->next) {
            if (p->v > p->next->v) { int32_t t=p->v; p->v=p->next->v; p->next->v=t; swapped=1; }
        }
    } while (swapped);
}

void list_bubble_sort_double(NodeDouble *head) {
    if (!head) return;
    int swapped;
    do {
        swapped = 0;
        for (NodeDouble *p = head; p && p->next; p = p->next) {
            if (p->v > p->next->v) { double t=p->v; p->v=p->next->v; p->next->v=t; swapped=1; }
        }
    } while (swapped);
}

/* mergesort (lists) */

static void list_split_int(NodeInt *head, NodeInt **a, NodeInt **b) {
    NodeInt *slow = head, *fast = head ? head->next : NULL;
    while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
    *a = head; *b = slow ? slow->next : NULL; if (slow) slow->next = NULL;
}
static NodeInt* list_merge_int(NodeInt *a, NodeInt *b) {
    if (!a) return b; if (!b) return a;
    if (a->v <= b->v) { a->next = list_merge_int(a->next, b); return a; }
    else              { b->next = list_merge_int(a, b->next); return b; }
}
NodeInt* list_mergesort_int(NodeInt *head) {
    if (!head || !head->next) return head;
    NodeInt *a, *b; list_split_int(head, &a, &b);
    a = list_mergesort_int(a); b = list_mergesort_int(b);
    return list_merge_int(a, b);
}

static void list_split_double(NodeDouble *head, NodeDouble **a, NodeDouble **b) {
    NodeDouble *slow = head, *fast = head ? head->next : NULL;
    while (fast && fast->next) { slow = slow->next; fast = fast->next->next; }
    *a = head; *b = slow ? slow->next : NULL; if (slow) slow->next = NULL;
}
static NodeDouble* list_merge_double(NodeDouble *a, NodeDouble *b) {
    if (!a) return b; if (!b) return a;
    if (a->v <= b->v) { a->next = list_merge_double(a->next, b); return a; }
    else              { b->next = list_merge_double(a, b->next); return b; }
}
NodeDouble* list_mergesort_double(NodeDouble *head) {
    if (!head || !head->next) return head;
    NodeDouble *a, *b; list_split_double(head, &a, &b);
    a = list_mergesort_double(a); b = list_mergesort_double(b);
    return list_merge_double(a, b);
}
