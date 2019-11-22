
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <sys/msg.h>
#include <sys/queue.h>

static const long Num_To_Add = 1000000000;
static const double Scale = 10.0 / RAND_MAX;

struct Queue {
    omp_lock_t lock;
    LIST_HEAD(message_list, Message) messages;
};

struct Message {
    long value;
    LIST_ENTRY(Message) pointers;
};

// Sends a message to the given queue;
void send_msg(struct Queue *queue, long message) {
    omp_set_lock(&queue->lock);
    struct Message *mess = malloc(sizeof(struct Message));
    mess->value = message;
    LIST_INSERT_HEAD(&queue->messages, mess, pointers);
    omp_unset_lock(&queue->lock);
}

// Attempts to get the newest message in the queue, if there is no message, -1 is returned.
long try_recv(struct Queue *queue) {
    long result = -1;
    omp_set_lock(&queue->lock);
    if (!LIST_EMPTY(&queue->messages)) {
        struct Message *head = LIST_FIRST(&queue->messages);
        result = head->value;
        LIST_REMOVE(head, pointers);
    }
    omp_unset_lock(&queue->lock);
    return result;
}

// Blocks until a message is received.
long block_recv(struct Queue *queue) {
    long result = -1;
    while (result == -1) {
        result = try_recv(queue);
    }
    return result;
}

long add_serial(const char *numbers) {
    long sum = 0;
    for (long i = 0; i < Num_To_Add; i++) {
        sum += numbers[i];
    }
    return sum;
}

long add_parallel(const char *numbers) {
    long sum = 0;
    int thread_count = omp_get_max_threads();

    // Setup queues for each thread
    struct Queue *queues = malloc(thread_count * sizeof(struct Queue));
    for (int i = 0; i < thread_count; i++) {
        struct Queue queue;
        omp_init_lock(&queue.lock);
        LIST_INIT(&queue.messages);
        queues[i] = queue;
    }

#pragma omp parallel num_threads(thread_count)
    {
        long my_sum = 0;
        int my_rank = omp_get_thread_num();

        // Each thread sums up their own portion of the array
#pragma omp for
        for (int i = 0; i < Num_To_Add; i++) {
            my_sum += numbers[i];
        }

        // Use a tree sum to find total sum of the elements in the array.
        int divisor = 2;
        int rank_difference = 1;

        while(1) {
            if (my_rank % divisor == 0) {
                // This thread is a receiver on this round.
                int sender = my_rank + rank_difference;
                if (sender >= thread_count)
                    break;
                long value = block_recv(&queues[my_rank]);
                my_sum += value;
            } else {
                // This thread is a sender on this round.
                int recipient = my_rank - rank_difference;
                if (recipient < 0)
                    break;
                send_msg(&queues[recipient], my_sum);
            }
            divisor *= 2;
            rank_difference *= 2;
        }

        // The final sum is located in the thread with rank 0
        if (my_rank == 0)
            sum = my_sum;
    }

    free(queues);

    return sum;
}

int main() {
    // Due to limitations with how rand() works, we cannot parallelize this generation
    char *numbers = malloc(sizeof(long) * Num_To_Add);
    for (long i = 0; i < Num_To_Add; i++) {
        numbers[i] = (char) (rand() * Scale);
    }

    struct timeval start, end;

    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    long sum_s = add_serial(numbers);
    gettimeofday(&end, NULL);
    double time_s = end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000;
    printf("Took %f seconds\n\n", time_s);

    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    long sum_p = add_parallel(numbers);
    gettimeofday(&end, NULL);
    double time_p = end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000;
    printf("Took %f seconds\n\n", time_p);

    printf("Sum serial: %ld\nSum parallel: %ld\n", sum_s, sum_p);

    printf("Speedup factor of %.4f\n", time_s/time_p);

    free(numbers);
    return 0;
}
