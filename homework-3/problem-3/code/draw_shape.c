// draw_shape.c
// Dependency-free visualization for HW2 – Parallel Rasterization
// Writes the N x N array as a PPM image file (output.ppm).

#include <stdio.h>

void draw_shape(int *arr, int N) {
    const char *filename = "output.ppm";
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        return;
    }

    // PPM header (P6 = binary RGB)
    fprintf(f, "P6\n%d %d\n255\n", N, N);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = y * N + x;
            unsigned char pixel[3];

            if (arr[idx] != 0) {
                // white
                pixel[0] = 255;
                pixel[1] = 255;
                pixel[2] = 255;
            } else {
                // black
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            }

            fwrite(pixel, 1, 3, f);
        }
    }

    fclose(f);
    printf("Wrote image to %s\n", filename);
}
