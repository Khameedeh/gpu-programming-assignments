#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

/**
 * Multithreaded Monte Carlo estimation of π
 * 
 * Basic implementation: Each thread generates its own subset of random samples
 * and counts points inside the circle. Results are protected by a mutex.
 * 
 * Race Condition Risk: Without proper synchronization, concurrent access to
 * the global counter could cause lost updates.
 */

typedef struct {
    long long total_samples;
    long long points_in_circle;
    pthread_mutex_t mutex;
} SharedData;

typedef struct {
    long long samples_per_thread;
    SharedData* shared;
    int thread_id;
} ThreadArgs;

/**
 * Thread function: Each thread generates its own set of random samples
 * and updates the shared counter (with mutex protection)
 */
void* thread_monte_carlo(void* args_ptr) {
    ThreadArgs* args = (ThreadArgs*)args_ptr;
    
    // Initialize random seed per thread with thread-safe version
    long int seed = time(NULL) + args->thread_id;
    struct drand48_data rng_state;
    srand48_r(seed, &rng_state);
    
    long long local_count = 0;
    
    // Generate samples for this thread
    for (long long i = 0; i < args->samples_per_thread; i++) {
        double x, y;
        drand48_r(&rng_state, &x);
        drand48_r(&rng_state, &y);
        
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }
    
    // Update shared counter with mutex protection
    pthread_mutex_lock(&args->shared->mutex);
    args->shared->points_in_circle += local_count;
    pthread_mutex_unlock(&args->shared->mutex);
    
    free(args);
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
    
    // Initialize shared data and mutex
    SharedData shared = {0};
    shared.total_samples = total_samples;
    pthread_mutex_init(&shared.mutex, NULL);
    
    // Create threads
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    long long samples_per_thread = total_samples / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        ThreadArgs* args = malloc(sizeof(ThreadArgs));
        args->samples_per_thread = samples_per_thread;
        args->shared = &shared;
        args->thread_id = i;
        
        pthread_create(&threads[i], NULL, thread_monte_carlo, args);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    gettimeofday(&end, NULL);
    
    // Clean up
    pthread_mutex_destroy(&shared.mutex);
    free(threads);
    
    // Calculate results
    double pi_estimate = 4.0 * shared.points_in_circle / total_samples;
    double execution_time = (double)(end.tv_sec - start.tv_sec) + 
                           (double)(end.tv_usec - start.tv_usec) / 1e6;
    
    // Print results in CSV format
    printf("PTHREAD_RESULTS\n");
    printf("samples,%lld\n", total_samples);
    printf("threads,%d\n", num_threads);
    printf("points_inside,%lld\n", shared.points_in_circle);
    printf("pi_estimate,%.15f\n", pi_estimate);
    printf("error,%.15f\n", fabs(pi_estimate - M_PI));
    printf("execution_time_sec,%.6f\n", execution_time);
    printf("speedup,N/A\n");
    
    return 0;
}