#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. 
 */


void 
basic_mm(int n, double* A, double* B, double* C) 
{
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
}

void
copy_matrix(int n, int i, int k, int block_size, double* M, double* tile_M, bool to_tile)
{
   for(int x = 0; x < block_size; x++)
   {
      if(to_tile)
         memcpy(tile_M + x * block_size, M + (i + x) * n + k, block_size * sizeof(double));
      else
         memcpy(M + (i + x) * n + k, tile_M + x * block_size, block_size * sizeof(double));
   }
}


void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{   
   std::vector<double> buf(3 * block_size * block_size);
   double* tile_A = buf.data();
   double* tile_B = tile_A + block_size * block_size;
   double* tile_C = tile_B + block_size * block_size;

   #pragma omp parallel
   {
      #ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
      #endif

      #pragma omp for collapse(2)
      for(int i = 0; i < n; i += block_size)
      {
         for(int j = 0; j < n; j += block_size)
         {   
            // copy tile of C into cache
            copy_matrix(n, i, j, block_size, C, tile_C, true);

            for(int k = 0; k < n; k += block_size)
            {
               // copy tile of A and B into cache
               copy_matrix(n, i, k, block_size, A, tile_A, true);
               copy_matrix(n, k, j, block_size, B, tile_B, true);  
               // perform basic matrix multiplication
               basic_mm(block_size, tile_A, tile_B, tile_C);
            }
            
            // copy tile of C back into C
            copy_matrix(n, i, j, block_size, C, tile_C, false);
         }
      }

      #ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
      #endif
   }
}