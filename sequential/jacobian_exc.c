#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

double f_eval(double* p_x, int m, int i){
   double value =  p_x[i] + 2*p_x[i+1] + 3*p_x[i+2] + 4*p_x[i+3] + 5*p_x[i+4] + 6*p_x[i+5] + 7*p_x[i+6] + 8*p_x[i+7] + 9*p_x[i+8] + 0*p_x[i+9];
   printf("%lf\t", value);
   return value;
}

int main(int argc, char* argv[]){
   if(argc == 4){

      FILE* fpIn = fopen(argv[1], "r");
      FILE* fpOut = fopen(argv[2], "w");
      double epsilon = atof(argv[3]);
      
      // n different input points, m variables ( x1,x2,.....xm) at each point
      int n, m;
      fscanf(fpIn, "%d", &n); 
      fscanf(fpIn, "%d", &m); 
      
      double h = 0.000001;
      double* input = (double*)malloc(m*n*sizeof(double));
      double* output = (double*)malloc(m*n*sizeof(double));
     
      // Loop over all the n input points
      for(int i = 0; i < n; i++){
         // Fill the input buffer 
         for(int j = i*m; j < m*(i+1); j++) {
            fscanf(fpIn, "%lf,", &input[j]);
         }
      }

      // Start the timer
      double start = omp_get_wtime();

      // Loop over all the n input points
      for(int i = 0; i < n; i++){
         // Evaluate the differential w.r.t to each variable
         for(int j = i*m; j < m*(i+1); j++) {
            input[j] += h;
            double val1 = f_eval(input, m, i*m);
            input[j] -= 2*h;
            double val2 = f_eval(input, m, i*m);
            double pdiff = (val1 - val2)/(2*epsilon);
            output[j] = pdiff;
            input[j] += h;
         } 
      }
      // Stop the timer
      double end = omp_get_wtime();

      for(int i = 0; i < n; i++){
         // Evaluate the differential w.r.t to each variable
         for(int j = i*m; j < m*(i+1); j++) {
            fprintf(fpOut, "%lf\t", output[j] );
         } 
         fprintf(fpOut, "\n");
      }

      double time = (end - start) * 1000;    // time in ms

      fprintf(fpOut, "%lf\n", time);

      fclose(fpIn);
      fclose(fpOut);
   }
   else{
      printf("Insufficient arguments");   
   }
}
