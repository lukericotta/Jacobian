#include <iomanip>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/host_vector.h>
#include <fstream>
using namespace std;

struct integrate
{
  __host__ __device__
   double operator()(double x)
   {
       return exp(sin(x))*cos(x/40);
   }
};

double f(double x){
	return exp(sin(x))*cos(x/40);
}


int main(int argc, char *argv[]){

	double n = atoi(argv[1]);
	double x0 = 0;
	double xn = 100;
	double h = (xn-x0)/n;
	double sum;
	double res;
	double ref = 32.121040666358;
	ofstream ofile;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	cudaEventRecord(start, NULL);

	thrust::host_vector<double> hX(n);

	for(int i = 0; i < (n-4); i++){
		hX[i] = (xn-x0)*(i+4)/n;
	}


        thrust::device_vector<double> dX = hX;
//	thrust::device_vector<double> dOUT;
	thrust::plus<float> binary_op;
	cudaEventRecord(start,NULL);
	double init = 0.0;

	sum = thrust::transform_reduce(dX.begin(), dX.end(), integrate(), init, binary_op);
//	sum = thrust::reduce(dOUT.begin(), dOUT.end(), 1, plus<double>());
	sum = sum*48;
	sum += 17*f(x0) + 59*f((xn-x0)/n) + 43*f((xn-x0)*2/n) + 49*f((xn-x0)*3/n);
	sum += 17*f((xn-x0)) + 59*f((xn-x0)*(n-1)/n) + 43*f((xn-x0)*(n-2)/n) + 49*f((xn-x0)*(n-3)/n);
	res = sum*h/48;

	cudaEventRecord(stop,NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

//Use this for scaling analysis
/*        cout << setprecision(12) << abs(res - ref);
        printf("\t%f\t",(msecTotal));
        cout << log2(n) << endl;
*/


	ofile.open ("problem3.out");
        ofile << setprecision(12) << abs(res-ref);
        ofile << endl;
        ofile << msecTotal;
        ofile.close();

	return 0;
}
