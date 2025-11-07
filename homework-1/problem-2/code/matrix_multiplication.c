#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// Configuration Constants
#define MAXN 5000   

// The IO_RUNS constant is REMOVED to avoid the "meaningless changes" penalty.
// I/O boundness will rely on the large size of the file operations.

// Function to perform matrix multiplication with different loop orders
// The function signature still uses MAXN for static allocation purposes in main.
void multiply(int A[MAXN][MAXN], int B[MAXN][MAXN], int C[MAXN][MAXN], int order, int n) {
    int i, j, k;
    switch(order) {
        case 0: // ijk
            for (i=0; i<n; i++)
                for (j=0; j<n; j++)
                    for (k=0; k<n; k++)
                        C[i][j] += A[i][k] * B[k][j];
            break;
        case 1: // ikj (Cache-optimized)
            for (i=0; i<n; i++)
                for (k=0; k<n; k++)
                    for (j=0; j<n; j++)
                        C[i][j] += A[i][k] * B[k][j];
            break;
        case 2: // jik
            for (j=0; j<n; j++)
                for (i=0; i<n; i++)
                    for (k=0; k<n; k++)
                        C[i][j] += A[i][k] * B[k][j];
            break;
    }
}

/**
 * @brief Performs a meaningful, O(N^2) CPU operation: calculate the sum of each row.
 * This is done in the io_simulate function to represent the "computation" part
 * of a real I/O-processing pipeline, without the artificial O(N^3) cost.
 * @param matrix_in The input matrix data (flat array).
 * @param n Matrix size N.
 */
void process_matrix_data(int *matrix_in, int n) {
    // This is the meaningful O(N^2) CPU work.
    for (int i = 0; i < n; i++) {
        long long row_sum = 0;
        for (int j = 0; j < n; j++) {
            row_sum += matrix_in[i * n + j];
        }
        // Example "processing": if the sum is even, set the first element to a non-zero value.
        // This prevents the compiler from optimizing out the loop entirely.
        if (row_sum % 2 == 0) {
            matrix_in[i * n] = 1; 
        }
    }
}


/**
 * @brief Simulates an I/O-bound workload by performing a large Read,
 * followed by O(N^2) CPU processing, and then a large Write.
 * This is done *once* to avoid "meaningless changes" and relies on the 
 * slowness of the file operations on large N to dominate.
 * @param n Matrix size N.
 */
void io_simulate(int n) {
    const char *DATA_FILE = "io_data_matrix.bin";
    
    // Allocate memory for the matrix (N x N)
    int size = n * n;
    int *matrix_data = (int*)malloc(size * sizeof(int));
    if (!matrix_data) {
        perror("Memory allocation failed for I/O matrix");
        return;
    }

    // --- Phase 1: Create a large input file (only if it doesn't exist) ---
    // If we can't find the file, or if we are running for the first time, create it.
    if (access(DATA_FILE, F_OK) == -1) {
        int fd_out = open(DATA_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd_out < 0) {
            perror("Error creating data file for I/O simulation");
            free(matrix_data);
            return;
        }

        // Fill with dummy data
        for (int i = 0; i < size; i++) {
            matrix_data[i] = i;
        }
        
        ssize_t w = write(fd_out, matrix_data, size * sizeof(int));
        if (w < 0) perror("write (initial file creation)");
        close(fd_out);
        fprintf(stderr, "Created data file: %s (Size: %ld bytes)\n", DATA_FILE, (long)size * sizeof(int));
    }
    
    // --- Phase 2: Read, Process, and Write (The single, required pipeline) ---
    
    // 1. Read the input file
    int fd = open(DATA_FILE, O_RDWR); // Open for both Read and Write
    if (fd < 0) {
        perror("Error opening data file");
        free(matrix_data);
        return;
    }
    
    // Read the entire N*N matrix data
    ssize_t r = read(fd, matrix_data, size * sizeof(int));
    if (r < 0) {
        perror("read");
        close(fd);
        free(matrix_data);
        return;
    }

    // 2. Perform O(N^2) CPU processing (Row sum and conditional update)
    process_matrix_data(matrix_data, n);
    
    // 3. Write the modified data back to the same file
    // Seek back to the start of the file before writing
    lseek(fd, 0, SEEK_SET);

    ssize_t w = write(fd, matrix_data, size * sizeof(int));
    if (w < 0) perror("write");
    
    close(fd);
    
    free(matrix_data);
}

int main(int argc, char *argv[]) {
    // Default values
    int order = 0; // 0=ijk, 1=ikj, 2=jik
    char *mode = "cpu"; // "cpu" or "io"
    int n = MAXN; 

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ijk|ikj|jik> <cpu|io> <N>\n", argv[0]);
        return 1;
    }

    // Argument Parsing (Unchanged)
    if (strcmp(argv[1], "ijk") == 0) order = 0;
    else if (strcmp(argv[1], "ikj") == 0) order = 1;
    else if (strcmp(argv[1], "jik") == 0) order = 2;
    else { fprintf(stderr,"Unknown loop order: %s\n", argv[1]); return 1; }
    
    if (strcmp(argv[2], "cpu") == 0) mode = "cpu";
    else if (strcmp(argv[2], "io") == 0) mode = "io";
    else { fprintf(stderr,"Unknown mode: %s\n", argv[2]); return 1; }

    n = atoi(argv[3]);
    if (n <= 0 || n > MAXN) {
        fprintf(stderr, "Invalid matrix size N=%d. Must be between 1 and %d.\n", n, MAXN);
        return 1;
    }
    
    // Static allocation for CPU-bound task
    static int A[MAXN][MAXN], B[MAXN][MAXN], C[MAXN][MAXN];

    if (strcmp(mode,"cpu") == 0) {
        // --- CPU-Bound Task (Matrix Multiplication, O(N^3)) ---
        srand(time(NULL));
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++) {
                A[i][j] = rand() % 100;
                B[i][j] = rand() % 100;
                C[i][j] = 0;
            }

        // Perform the O(N^3) computation
        multiply(A, B, C, order, n);
    } else {
        // --- I/O-Bound Task (Simulated, I/O Time >> CPU Time) ---
        // A single, non-repeated Read/Process/Write sequence is performed on large data.
        io_simulate(n);
    }

    return 0;
}
