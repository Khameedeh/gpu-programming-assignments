#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define BUFFER_SIZE 5
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 3
#define ITEMS_TO_PRODUCE 5

int buffer[BUFFER_SIZE];
int count = 0;       // number of items currently in buffer
int in = 0;          // next position to consume
int out = 0;         // next position to produce

// FIX: Added shutdown flag for graceful termination
volatile bool producers_done = false;

pthread_mutex_t mutex;
pthread_cond_t not_full;
pthread_cond_t not_empty;

// --- Producer thread function ---
void* producer(void* arg) {
    int id = *(int*)arg;
    free(arg);

    for (int i = 0; i < ITEMS_TO_PRODUCE; i++) {
        int item = i + (id * 100); // unique item per producer

        pthread_mutex_lock(&mutex);

        // Handle spurious wakeups with while loop
        while (count == BUFFER_SIZE)
            pthread_cond_wait(&not_full, &mutex);

        // Add item to buffer
        buffer[in] = item;
        in = (in + 1) % BUFFER_SIZE;
        count++;
        printf("Producer %d produced %d (buffer size: %d)\n", id, item, count);

        pthread_mutex_unlock(&mutex);
        
        // Wake all waiting consumers
        pthread_cond_broadcast(&not_empty);

        // A random delay to simulate processing time
        usleep((rand() % 500) * 1000); 
    }
    
    // FIX: Signal all consumers after producer is done
    pthread_mutex_lock(&mutex);
    printf("Producer %d finished producing items\n", id);
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

// --- Consumer thread function ---
void* consumer(void* arg) {
    int id = *(int*)arg;
    free(arg);

    while (1) {
        pthread_mutex_lock(&mutex);

        // FIX: Changed condition - exit if producers done AND buffer is empty
        while (count == 0 && !producers_done) {
            pthread_cond_wait(&not_empty, &mutex);
        }
        
        // FIX: Exit gracefully if producers done and buffer empty
        if (count == 0 && producers_done) {
            printf("Consumer %d exiting gracefully\n", id);
            pthread_mutex_unlock(&mutex);
            break;
        }

        // Remove item from buffer
        int item = buffer[out];
        out = (out + 1) % BUFFER_SIZE;
        count--;
        sleep(1);  // A small delay to simulate processing time for the item
        printf("Consumer %d consumed %d (buffer size: %d)\n", id, item, count);

        pthread_mutex_unlock(&mutex);
        
        // Wake all waiting producers
        pthread_cond_broadcast(&not_full);

        // A random delay to simulate processing time
        usleep((rand() % 700) * 1000);
    }

    return NULL;
}

int main() {
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&not_full, NULL);
    pthread_cond_init(&not_empty, NULL);

    for (int i = 0; i < NUM_PRODUCERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        pthread_create(&producers[i], NULL, producer, id);
    }

    for (int i = 0; i < NUM_CONSUMERS; i++) {
        int* id = malloc(sizeof(int));
        *id = i + 1;
        pthread_create(&consumers[i], NULL, consumer, id);
    }

    // Wait for all producers to finish
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }

    // FIX: Signal consumers that producers are done
    pthread_mutex_lock(&mutex);
    producers_done = true;
    printf("All producers finished. Waiting for consumers to finish remaining items...\n");
    pthread_mutex_unlock(&mutex);
    
    // FIX: Broadcast to wake all waiting consumers so they check the shutdown condition
    pthread_cond_broadcast(&not_empty);

    // Wait for all consumers to finish gracefully (no pthread_cancel needed!)
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }
    
    printf("All threads finished. Cleaning up...\n");
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    return 0;
}