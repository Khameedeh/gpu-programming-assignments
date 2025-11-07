#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/**
 * Sequential Monte Carlo estimation of π
 * 
 * Algorithm: Generate random points in [0,1] x [0,1] and count how many
 * fall within the unit circle (x^2 + y^2 <= 1). The ratio of points inside
 * the circle to total points approximates π/4.
 */

typedef struct {
    long long total_samples;
    long long points_in_circle;
    double pi_estimate;
    double execution_time;
} PiResult;

/**
 * Sequential Monte Carlo implementation
 * @param samples: Number of random samples to generate
 * @return: PiResult containing estimate and timing
 */
PiResult monte_carlo_sequential(long long samples) {
    PiResult result = {0};
    result.total_samples = samples;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Use drand48 for better random number quality than rand()
    srand48(time(NULL));
    
    long long count_inside = 0;
    for (long long i = 0; i < samples; i++) {
        // Generate random point in [0,1] x [0,1]
        double x = drand48();
        double y = drand48();
        
        // Check if point falls within unit circle
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            count_inside++;
        }
    }
    
    gettimeofday(&end, NULL);
    
    result.points_in_circle = count_inside;
    
    // Estimate π: (points_inside / total_points) ≈ π/4
    // Therefore: π ≈ 4 * (points_inside / total_points)
    result.pi_estimate = 4.0 * count_inside / samples;
    
    // Calculate execution time in seconds
    result.execution_time = (double)(end.tv_sec - start.tv_sec) + 
                           (double)(end.tv_usec - start.tv_usec) / 1e6;
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <number_of_samples>\n", argv[0]);
        fprintf(stderr, "Example: %s 1000000\n", argv[0]);
        return 1;
    }
    
    long long samples = atoll(argv[1]);
    
    if (samples <= 0) {
        fprintf(stderr, "Error: number_of_samples must be positive\n");
        return 1;
    }
    
    PiResult result = monte_carlo_sequential(samples);
    
    // Print results in CSV format for easy parsing
    printf("SEQUENTIAL_RESULTS\n");
    printf("samples,%lld\n", result.total_samples);
    printf("points_inside,%lld\n", result.points_in_circle);
    printf("pi_estimate,%.15f\n", result.pi_estimate);
    printf("error,%.15f\n", fabs(result.pi_estimate - M_PI));
    printf("execution_time_sec,%.6f\n", result.execution_time);
    
    return 0;
}