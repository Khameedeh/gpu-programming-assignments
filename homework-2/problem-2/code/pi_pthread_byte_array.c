#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

/**
 * Advanced Optimization: Byte-Array Result Storage with Partitioning
 * 
 * Strategy: Instead of counting inside the critical path, store the result
 * of each sample (0 or 1) in a byte array. Each thread writes to its own
 * partition of the array (no false sharing). A final aggregation pass
 * counts all points.
 * 
 * Benefits:
 * - Eliminates any synchronization during sampling phase
 * - Avoids false sharing (each thread writes to different cache lines)
 * - Enables different load-balancing strategies (static vs dynamic)
 * 
 * Trade-off: Uses O(n) memory instead of O(1)
 */

typedef struct {
    long long start_idx;
    long long end_idx;
    unsigned char* result_array;
    int thread_id;
} PartitionData;

/**
 * Thread function: Generate samples and store results in array partition
 * Static partitioning: each thread writes to a fixed region
 */
void* thread_monte_carlo_byte_array(void* args_ptr) {
    PartitionData* data = (PartitionData*)args_ptr;
    
    // Initialize thread-local RNG
    unsigned int seed = time(NULL) + data->thread_id;
    struct drand48_data rng_state;
    srand48_r(seed, &rng_state);
    
    // Each thread fills its assigned partition
    for (long long i = data->start_idx; i < data->end_idx; i++) {
        double x, y;
        drand48_r(&rng_state, &x);
        drand48_r(&rng_state, &y);
        
        // Store 1 if inside circle, 0 if outside
        // This avoids any shared counter entirely
        data->result_array[i] = (x * x + y * y <= 1.0) ? 1 : 0;
    }
    
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <number_of_samples> <number_of_threads>\n", argv[0]);
        fprintf(stderr, "Example: %s 1000000 4\n", argv[0]);
        return 1;
    }
    
    long long total_samples = atoll(argv[1]);
    int num_threads = atoi(argv[2]);
    
    if (total_samples <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error: samples and threads must be positive\n");
        return 1;
    }
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Allocate result array: 1 byte per sample
    unsigned char* result_array = malloc(total_samples * sizeof(unsigned char));
    if (!result_array) {
        fprintf(stderr, "Error: Failed to allocate result array\n");
        return 1;
    }
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    PartitionData* partition_data = malloc(num_threads * sizeof(PartitionData));
    
    long long samples_per_thread = total_samples / num_threads;
    
    // Create threads with static partitioning
    for (int i = 0; i < num_threads; i++) {
        partition_data[i].start_idx = i * samples_per_thread;
        partition_data[i].end_idx = (i == num_threads - 1) ? 
                                     total_samples : (i + 1) * samples_per_thread;
        partition_data[i].result_array = result_array;
        partition_data[i].thread_id = i;
        
        pthread_create(&threads[i], NULL, thread_monte_carlo_byte_array, &partition_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Aggregation phase: count all 1s in the array
    long long points_inside = 0;
    for (long long i = 0; i < total_samples; i++) {
        points_inside += result_array[i];
    }
    
    gettimeofday(&end, NULL);
    
    // Clean up
    free(result_array);
    free(threads);
    free(partition_data);
    
    // Calculate results
    double pi_estimate = 4.0 * points_inside / total_samples;
    double execution_time = (double)(end.tv_sec - start.tv_sec) + 
                           (double)(end.tv_usec - start.tv_usec) / 1e6;
    
    // Print results in CSV format
    printf("BYTE_ARRAY_RESULTS\n");
    printf("samples,%lld\n", total_samples);
    printf("threads,%d\n", num_threads);
    printf("points_inside,%lld\n", points_inside);
    printf("pi_estimate,%.15f\n", pi_estimate);
    printf("error,%.15f\n", fabs(pi_estimate - M_PI));
    printf("execution_time_sec,%.6f\n", execution_time);
    printf("speedup,N/A\n");
    
    return 0;
}