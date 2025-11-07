#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define BUFFER_SIZE 5
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 3
#define ITEMS_TO_PRODUCE 5

int buffer[BUFFER_SIZE];
int count = 0;       // number of items currently in buffer
int in = 0;          // next position to produce
int out = 0;         // next position to consume

// Shutdown flag for graceful termination
volatile bool producers_done = false;

pthread_mutex_t mutex;
pthread_cond_t not_full;
pthread_cond_t not_empty;

// Statistics tracking
int items_produced = 0;
int items_consumed = 0;
int max_buffer_size = 0;

// --- Producer thread function ---
void* producer(void* arg) {
    int id = *(int*)arg;
    free(arg);

    for (int i = 0; i < ITEMS_TO_PRODUCE; i++) {
        int item = i + (id * 100); // unique item per producer

        pthread_mutex_lock(&mutex);

        // Handle spurious wakeups - wait while buffer is full
        while (count == BUFFER_SIZE) {
            pthread_cond_wait(&not_full, &mutex);
        }

        // Add item to buffer
        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        count++;
        items_produced++;
        
        // Track max buffer size for statistics
        if (count > max_buffer_size) {
            max_buffer_size = count;
        }

        printf("[%ld] Producer %d produced %d (buffer: %d/%d)\n", 
               pthread_self(), id, item, count, BUFFER_SIZE);

        pthread_mutex_unlock(&mutex);
        
        // Wake all waiting consumers
        pthread_cond_broadcast(&not_empty);

        // Simulate processing time
        usleep((rand() % 500) * 1000); 
    }
    
    printf("[%ld] Producer %d finished producing items\n", pthread_self(), id);
    return NULL;
}

// --- Consumer thread function ---
void* consumer(void* arg) {
    int id = *(int*)arg;
    free(arg);

    while (1) {
        pthread_mutex_lock(&mutex);

        // Wait while buffer is empty AND producers are still working
        while (count == 0 && !producers_done) {
            pthread_cond_wait(&not_empty, &mutex);
        }
        
        // Exit gracefully if no more items and producers finished
        if (count == 0 && producers_done) {
            printf("[%ld] Consumer %d exiting gracefully\n", pthread_self(), id);
            pthread_mutex_unlock(&mutex);
            break;
        }

        // Consume item from buffer
        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;
        items_consumed++;

        printf("[%ld] Consumer %d consumed %d (buffer: %d/%d)\n", 
               pthread_self(), id, item, count, BUFFER_SIZE);

        pthread_mutex_unlock(&mutex);
        
        // Wake all waiting producers
        pthread_cond_broadcast(&not_full);

        // Simulate item processing time
        sleep(1);
        usleep((rand() % 700) * 1000);
    }

    return NULL;
}

int main() {
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];

    srand(time(NULL));

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&not_full, NULL);
    pthread_cond_init(&not_empty, NULL);

    printf("=== Producer-Consumer System Started ===\n");
    printf("Configuration: %d producers, %d consumers, buffer size %d, %d items per producer\n\n",
           NUM_PRODUCERS, NUM_CONSUMERS, BUFFER_SIZE, ITEMS_TO_PRODUCE);

    // Create producer threads
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        pthread_create(&producers[i], NULL, producer, id);
    }

    // Create consumer threads
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        pthread_create(&consumers[i], NULL, consumer, id);
    }

    // Wait for all producers to complete
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }

    // Signal consumers that all producers are done
    pthread_mutex_lock(&mutex);
    producers_done = true;
    printf("\n=== All producers finished. Signaling consumers... ===\n\n");
    pthread_mutex_unlock(&mutex);
    
    // Broadcast to wake all waiting consumers for graceful shutdown
    pthread_cond_broadcast(&not_empty);

    // Wait for all consumers to finish gracefully
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }
    
    // Print statistics
    printf("\n=== System Statistics ===\n");
    printf("Total items produced: %d\n", items_produced);
    printf("Total items consumed: %d\n", items_consumed);
    printf("Maximum buffer size reached: %d/%d\n", max_buffer_size, BUFFER_SIZE);
    printf("All threads finished successfully.\n");

    // Cleanup
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    return 0;
}