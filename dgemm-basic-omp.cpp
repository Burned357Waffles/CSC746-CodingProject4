#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values.
 * C[i*n + j] += A[i*n + k] * B[k*n + j];
 */
void square_dgemm(int n, double* A, double* B, double* C) 
{
   #pragma omp parallel 
   {
      #ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
      #endif
      
      #pragma omp for
      for(int i = 0; i < n; i++)
      {
         double* aptr = A + i * n;
         double* cptr = C + i * n;
         for(int j = 0; j < n; j++)
         {
            double p_dot = 0;
            for(int k = 0; k < n; k++)
            {
               p_dot += aptr[k] * B[k*n + j];
            }
            cptr[j] += p_dot;
         }
      }  
      #ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
      #endif
   }
   
}
