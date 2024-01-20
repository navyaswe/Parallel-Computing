#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);
    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }


   
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }


    
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();


    
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));


    
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * n);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    
    memcpy(input_array1, input_array, sizeof(int) * n);
    memset(prefix_array1, 0, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);


    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

// Funtion to calculate the prefix sum using NlogN algorithm
void prefix_sum_p1(int *src, int *prefix, int n) {

// Allocate memory for a temporary array to store intermediate values
    int *temp = (int *)malloc(sizeof(int) * n);
// Allocate memory for a new source array to preserve the original values

    int *new_src = (int *)malloc(sizeof(int) * n);
    memcpy(new_src, src, sizeof(int) * n);
    int stride = 1;
// Perform a tree-based reduction and distribution for the parallel prefix sum
    while (stride < n) {
#pragma omp parallel for
        for (int i = stride; i < n; i++) {
            temp[i] = new_src[i] + new_src[i - stride]; // Update 'new_src' with the computed prefix sums from 'temp'
        }

#pragma omp parallel for
        for (int i = stride; i < n; i++) {
            new_src[i] = temp[i];
        }

        stride *= 2;
    }
// Copy the computed prefix sums from 'new_src' to the 'prefix' array

    memcpy(prefix, new_src, sizeof(int) * n);
    free(temp);
    free(new_src);
}


// Funtion to calculate the prefix sum using 2(N-1) algorithm
void prefix_sum_p2(int* src, int* prefix, int n) {
    int num_threads, *partial_sums, *prefix_sums = prefix;

   // Determining the number of threads available
    #pragma omp parallel
    {
        int i;
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
           // Allocate memory for partial sums with an extra element
            partial_sums = (int *)malloc(sizeof(int) * (num_threads + 1));
             
 	   // Initialize the first element of partial_sums to 0
            partial_sums[0] = 0;
        }

        int thread_id = omp_get_thread_num();
        int local_sum = 0;

        // Compute the local prefix sum for the thread's portion of the input
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
            local_sum += src[i];
            prefix_sums[i] = local_sum;
        }

       // Store the local sum in the partial_sums array
        partial_sums[thread_id + 1] = local_sum;

       // Synchronize all threads to ensure the partial sums are complete
        #pragma omp barrier

        int offset = 0;

       // Calculate the offset for the current thread
        for (i = 0; i < (thread_id + 1); i++) {
            offset += partial_sums[i];
        }

       // Update the output array with the final prefix sums
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++) {
            prefix_sums[i] += offset;
        }
    }

     // Free the allocated memory for partial_sums
    free(partial_sums);
}
