// main.c
// Tiny test harness for draw_shape: draws a diagonal line.

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void draw_shape(int *arr, int N);

int main(void) {
    int N = 512;
    int *pixels = malloc(N * N * sizeof(int));
    if (!pixels) {
        perror("malloc");
        return 1;
    }

    memset(pixels, 0, N * N * sizeof(int));

    // Simple test: diagonal line
    // x + y = 0
    for (int i = 0; i < N; ++i) {
        pixels[i * N + i] = 1;
    }

    draw_shape(pixels, N);

    free(pixels);
    printf("Open output.ppm with any image viewer.\n");
    return 0;
}
