#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "f_eval2.h"

void jacobian(double *input, double *output, const int m, const int n, const double h, const double epsilon)
{
   #pragma omp parallel for num_threads(16) collapse(2)
   for(int i = 0; i < n; i++){
      for(int j = 0; j < m; j++) { 
         double val1 = f_eval(&input[i*m], m, j, h);

         double val2 = f_eval(&input[i*m], m, j, -h);

         double result = (val1 - val2) / (2*epsilon);
         output[i*m + j] = result;
      } 
   }  
}

int main(int argc, char* argv[]) {
   if(argc == 4){
       FILE* fpIn = fopen(argv[1], "r");
       FILE* fpOut = fopen(argv[2], "w");
       double epsilon = atof(argv[3]);
       const double h = 1e-2;

       // n different input points, m variables ( x1,x2,.....xm) at each point
       int n, m;
       fscanf(fpIn, "%d", &n);
       fscanf(fpIn, "%d", &m);

       double *input = (double*) malloc(m * n * sizeof(double));
       double *output = (double*) malloc(m * n * sizeof(double));
      
       // Read all the input points from the file
       for (int i = 0; i < n; i++) {
          for(int j = 0; j < m; j++) {
             fscanf(fpIn, "%lf,", &input[i*m+j]);
          }
       }

        // Start the timer
       double start = omp_get_wtime();
          
       jacobian(input, output, m, n, h, epsilon);

       // Stop the timer
       double end = omp_get_wtime();
* 
       // Write the result into the file
       for(int i = 0; i < n; i++) {
           for(int j = 0; j < m; j++) {
               if(j != m-1)
                   fprintf(fpOut, "%.6lf,", output[i*m + j] );
               else
                   fprintf(fpOut, "%.6lf\n", output[i*m + j] );
           }
       }
      
       fclose(fpIn);
       fclose(fpOut);

       double time = (end - start) * 1000;    // time in ms
      
       // Create a new file to log the execution time    
       FILE* fpLog = fopen("ompLog", "a");
       fprintf(fpLog, "%d %d %lf\n", n, m, time);
       fclose(fpLog);
       
       free(input);
       free(output);
    }
    else{
       printf("Insufficient arguments");
    }

    return 0;
}
