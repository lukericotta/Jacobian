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
#include <thrust/device_ptr.h>


// struct func
// {

//     int m,n;
//     double h,epsilon;
//     func(int m, int n, double h, double epsilon) : m(m),n(n),h(h),epsilon(epsilon) {}

//   __host__ __device__
//   double operator()(double& input)
//   {
//          input += h;
//          double val1 = f_eval(&input, m);

//          input -= 2*h;
//          double val2 = f_eval(&input, m);

//          double result = (val1 - val2) / (2*epsilon);
//          return result;

//          // input[i*m + j] += h;

//    }
// };


struct func2
{

    int m;
    double h,epsilon;
    thrust::device_ptr<double> addr;
    func2(int m, double h, double epsilon,  thrust::device_ptr<double> addr) : m(m),h(h),epsilon(epsilon),addr(addr) {}

  __host__ __device__
  double operator()(double& input)
  {
         input += h;
         double val1 = f_eval(addr, m);
	for(int i =0; i < m; i++)
	 printf("%lf \t", addr[i]);
/*
         input -= 2*h;
         double val2 = f_eval(addr, m);

         double result = (val1 - val2) / (2*epsilon);
         return result;

         input += h;
*/
	return val1;
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

       for(int i = 0; i < n; i++){

	//	double* addr = thrust::raw_pointer_cast(&dX[i*m]);

		thrust::device_vector<double>::iterator begin_iter = dX.begin();
		thrust::advance(begin_iter, i * m);
		thrust::device_vector<double>::iterator end_iter = begin_iter;
		thrust::advance(end_iter, m);

		thrust::device_vector<double>::iterator dout_iter = dout.begin();
		thrust::advance(dout_iter, i * m);
	        
		thrust::device_ptr<double> addr = dX.data();
		addr += (i * m);	
		double* ptr = thrust::raw_pointer_cast(addr);
		thrust::transform(begin_iter, end_iter, dout_iter, func2(m,h,epsilon,addr));

	}

	printf("%lf\n", *dout.begin());
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
