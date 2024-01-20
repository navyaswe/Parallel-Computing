#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>

#define PI 3.1415926535

void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1(int num_steps);
double calcPi_P1A(int num_steps);
double calcPi_P2(int num_steps);

int main(int argc, char** argv)
{
  //get input values
  uint32_t num_steps = 100000;
<<<<<<< HEAD
  //    if (argc > 1) {
  //           num_steps = atoi(argv[1]);
  //             } else {
  //                    //usage(argc, argv);
  //                          // printf("using %"PRIu32"\n", num_steps);
  //                            }
  //                               // fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);
  //
  //                                 // set up timer
  //                                   uint64_t start_t;
  //                                     uint64_t end_t;
  //                                       InitTSC();
  //
  //                                         // calculate in serial
  //                                           //start_t = ReadTSC();
  //                                             //double Pi0 = calcPi_Serial(num_steps);
  //                                               //end_t = ReadTSC();
  //                                                 //printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n", num_steps, ElapsedTime(end_t - start_t));
  //                                                   //printf("Pi is %0.10f\n", Pi0);
  //                                                     /*
  //                                                       // calculate in parallel with integration using critical directive
  //                                                           start_t = ReadTSC();
  //                                                               double Pi1 = calcPi_P1(num_steps);
  //                                                                   end_t = ReadTSC();
  //                                                                       printf("Time to calculate Pi using OpenMP critical section  with %"PRIu32" steps is: %g\n",
  //                                                                       -- INSERT --                                                                                          
=======
  if (argc > 1) {
       num_steps = atoi(argv[1]);
  } else {
       usage(argc, argv);
       printf("using %"PRIu32"\n", num_steps);
  }
    fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);
  
  // set up timer
  uint64_t start_t;
  uint64_t end_t;
  InitTSC();

  // calculate in serial
  start_t = ReadTSC();
  double Pi0 = calcPi_Serial(num_steps);
  end_t = ReadTSC();
  printf("Time to calculate Pi serially with %"PRIu32" steps is: %g\n", num_steps, ElapsedTime(end_t - start_t));
  printf("Pi is %0.10f\n", Pi0);
  
  // calculate in parallel with integration using critical directive
    start_t = ReadTSC();
    double Pi1 = calcPi_P1(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi using OpenMP critical section  with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi1);

  // calculate the parallel with integration using  atomic directive   
    start_t = ReadTSC();
    double Pi2 = calcPi_P1A(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi using OpenMP atomic  with %"PRIu32" steps is: %g\n",
           num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi2);
 

  // calculate in parallel with Monte Carlo
    start_t = ReadTSC();
    double Pi3 = calcPi_P2(num_steps);
    end_t = ReadTSC();
    printf("Time to calculate Pi in using Monte Carlo Estimates  with %"PRIu32" guesses is: %g\n",
          num_steps, ElapsedTime(end_t - start_t));
    printf("Pi is %0.10f\n", Pi3);

    
    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

// Calculating pi based on formula dx(4 / 1+x^2) in interval [0, 1]
double calcPi_Serial(int num_steps) {
    double pi = 0.0;
    // calculating each subinterval size by  taking rectangle width(step) 
    double step = 1.0 / (double)num_steps; 

    for (int i = 0; i < num_steps; i++) {
	//Divides x co-ordinates into small intervals
        double x = (i + 0.5) * step;
    // quarter of a circle with a radius of 1, which approximates the area under the curve within the subinterval.
        pi += 4.0 / (1.0 + x * x);
    }

    pi *= step;
    return pi;
}

//parallel integration with OpenMP critical section
double calcPi_P1(int num_steps) {
    double pi = 0.0;
    double step = 1.0 / (double)num_steps;

    #pragma omp parallel for
    for (int i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        double local_sum = 4.0 / (1.0 + x * x);

        #pragma omp critical
        pi += local_sum;
    }

    pi *= step;
    return pi;
}

//parallel with integration using  atomic

double calcPi_P1A(int num_steps) {
   double pi = 0.0;
   double step = 1.0 / (double)num_steps;

   #pragma omp parallel for
   for (int i = 0; i < num_steps; i++) {
       double x = (i + 0.5) * step;
       double local_sum = 4.0 / (1.0 + x * x);

       #pragma omp atomic
       pi += local_sum;
   }

   pi *= step;
   return pi;
}


// Monte Carlo estimation of π based on the probability of a random point falling inside a quarter circle.
double calcPi_P2(int num_steps) {
 int points = 0;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() + time(NULL));//Random number generator uniquely to a thread

        #pragma omp for reduction(+:points)//aggregation of points from multiple threads.
        for (int i = 0; i < num_steps; i++) {
     //generates random double-precision floating-point number
            double x = (double)rand_r(&seed) / (double)RAND_MAX;
            double y = (double)rand_r(&seed) / (double)RAND_MAX;

            if (x * x + y * y <= 1.0) {
                points++;
            }
        }
    }

    double pi = 4.0 * (double)points / (double)num_steps;
    return pi;
}


>>>>>>> dc6aa2a4c80b2010ae3a90d6f199c7281ac00e45
