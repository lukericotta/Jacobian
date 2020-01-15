#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "f_eval.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/host_vector.h>

struct func
{

    double m,n,h,epsilon;
    func(int m, int n, double h, double epsilon) : m(m),n(n),h(h),epsilon(epsilon) {}

  __host__ __device__
  double operator()(double& input)
  {

    for(int j = 0; j <m; j++){
	double val1 = f_eval(&input, m, j, h);
	double val2 = f_eval(&input, m, j, -h);
	double result = (val1 - val2) / (2*epsilon);
	return result;
    }
   } 
};


struct func2
{

    double m,n,h,epsilon;
    func2(int m, int n, double h, double epsilon) : m(m),n(n),h(h),epsilon(epsilon) {}

  __host__ __device__
  double operator()(double& input)
  {
         input += h;
         double val1 = f_eval(&input, m);

         input -= 2*h;
         double val2 = f_eval(&input, m);

         double result = (val1 - val2) / (2*epsilon);
         return result;

         // input[i*m + j] += h;

   }
};


struct func3
{

    double m,n,h,epsilon;
    func3(int m, int start_addr, double h, double epsilon) : m(m),start_addr(start_addr),h(h),epsilon(epsilon) {}

  __host__ __device__
  double operator()(double& input)
  {
         input += h;
         double val1 = f_eval(start_addr, m);

         input -= 2*h;
         double val2 = f_eval(start_addr, m);

         double result = (val1 - val2) / (2*epsilon);
         return result;

         // input[i*m + j] += h;

   }
};

struct func4
{

  __host__ __device__
  double operator(double& input)()
  {

	return -1.7;

  }

};


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
      
       thrust::host_vector<double> hX(n*m);

       // Read all the input points from the file
       for (int i = 0; i < n; i++) {
          for(int j = 0; j < m; j++) {
             fscanf(fpIn, "%lf,", &input[i*m+j]);
	     hX[i*m+j] = input[i*m+j];
          }
       }

        // Start the timer
//       double start = omp_get_wtime();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	cudaEventRecord(start, NULL);



       thrust::device_vector<double> dX = hX;
       thrust::device_vector<double> dout(n*m);

//       thrust::transform(dX.begin(),dX.end(),dout.begin(), func(m,n,h,epsilon));

//       for(int i = 0; i < n; i++){

//		int start_addr = i*m;

//		thrust::transform(&dX[start_addr],&dX[start_addr+m],&dout[start_addr], func3(m,&dX[start_addr],h,epsilon));

//	}


	thrust::transform(dX.begin(),dX.end(),dout.begin(),func4())

       thrust::copy(dout.begin(), dout.end(), &output[0]);

       // Stop the timer
//       double end = omp_get_wtime();
  
	cudaEventRecord(stop,NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);




       // Write the result into the file
       for(int i = 0; i < n; i++) {
           for(int j = 0; j < m; j++) {
               if(j != m-1)
                   fprintf(fpOut, "%.4lf,", output[i*m + j] );
               else
                   fprintf(fpOut, "%.4lf\n", output[i*m + j] );
           }
       }
      
       fclose(fpIn);
       fclose(fpOut);

       double time = msecTotal;    // time in ms
      
       // Create a new file to log the execution time    
       FILE* fpLog = fopen("sequentialLog", "a");
       fprintf(fpLog, "%d\t%d\t%lf\n", n, m, time);
       fclose(fpLog);
       
       free(input);
       free(output);
    }
    else{
       printf("Insufficient arguments");
    }

    return 0;
}
