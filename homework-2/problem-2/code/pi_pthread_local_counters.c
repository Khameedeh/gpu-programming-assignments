#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

/**
 * Optimized Monte Carlo with Thread-Local Counters
 * 
 * Optimization: Each thread maintains its own local counter (no synchronization).
 * Only at the end do threads add their local counts to a shared total.
 * 
 * Benefit: Eliminates mutex contention on the critical path, allowing threads
 * to run in parallel without synchronization overhead.
 * 
 * This is the preferred pattern for work-aggregate algorithms.
 */

typedef struct {
    long long total_samples;
    long long points_in_circle;
} SharedData;

typedef struct {
    long long samples_per_thread;
    long long local_count;  // Thread-local counter (no synchronization)
    int thread_id;
} ThreadLocalData;

/**
 * Thread function with thread-local counter
 * No synchronization during sampling phase
 */
void* thread_monte_carlo_local(void* args_ptr) {
    ThreadLocalData* data = (ThreadLocalData*)args_ptr;
    
    // Initialize random seed per thread
    unsigned int seed = time(NULL) + data->thread_id;
    struct drand48_data rng_state;
    srand48_r(seed, &rng_state);
    
    // Each thread maintains its own local counter (no mutex needed)
    long long local_count = 0;
    
    for (long long i = 0; i < data->samples_per_thread; i++) {
        double x, y;
        drand48_r(&rng_state, &x);
        drand48_r(&rng_state, &y);
        
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }
    
    // Store result in thread's local data
    data->local_count = local_count;
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
    
    // Allocate thread-local data (no mutex needed in main structures)
    ThreadLocalData* thread_data = malloc(num_threads * sizeof(ThreadLocalData));
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    
    long long samples_per_thread = total_samples / num_threads;
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].samples_per_thread = samples_per_thread;
        thread_data[i].local_count = 0;
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], NULL, thread_monte_carlo_local, &thread_data[i]);
    }
    
    // Wait for all threads and aggregate results
    long long total_inside = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_inside += thread_data[i].local_count;
    }
    
    gettimeofday(&end, NULL);
    
    // Clean up
    free(thread_data);
    free(threads);
    
    // Calculate results
    double pi_estimate = 4.0 * total_inside / total_samples;
    double execution_time = (double)(end.tv_sec - start.tv_sec) + 
                           (double)(end.tv_usec - start.tv_usec) / 1e6;
    
    // Print results in CSV format
    printf("PTHREAD_LOCAL_COUNTERS_RESULTS\n");
    printf("samples,%lld\n", total_samples);
    printf("threads,%d\n", num_threads);
    printf("points_inside,%lld\n", total_inside);
    printf("pi_estimate,%.15f\n", pi_estimate);
    printf("error,%.15f\n", fabs(pi_estimate - M_PI));
    printf("execution_time_sec,%.6f\n", execution_time);
    printf("speedup,N/A\n");
    
    return 0;
}