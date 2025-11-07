#ifndef SORT_H
#define SORT_H

#include <stdint.h>
#include <stddef.h>

/* Linked-list node types */
typedef struct NodeInt    { int32_t v; struct NodeInt *next; } NodeInt;
typedef struct NodeDouble { double  v; struct NodeDouble *next; } NodeDouble;

/* Array sorts */
void insertion_sort_int(int32_t *a, size_t n);
void insertion_sort_double(double *a, size_t n);
void bubble_sort_int(int32_t *a, size_t n);
void bubble_sort_double(double *a, size_t n);
void mergesort_int(int32_t *a, size_t n);
void mergesort_double(double *a, size_t n);

/* Linked-list sorts */
NodeInt*    list_insertion_sort_int(NodeInt *head);
NodeDouble* list_insertion_sort_double(NodeDouble *head);
void        list_bubble_sort_int(NodeInt *head);
void        list_bubble_sort_double(NodeDouble *head);
NodeInt*    list_mergesort_int(NodeInt *head);
NodeDouble* list_mergesort_double(NodeDouble *head);

#endif /* SORT_H */
