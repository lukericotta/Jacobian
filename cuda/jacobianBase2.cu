#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "nvToolsExt.h"
#include <cuda.h>
#include <cmath>
#include "f_eval2.cuh"

using namespace std;

// x - m-dimensional point
// n - total number of points
__global__ void jacobian(double *deviceInput, double *deviceResult, const int m, const int n, const double h, const double epsilon)
{
    int currentThread = blockIdx.x * blockDim.x + threadIdx.x;

    if(currentThread < n) {
        for(int j = 0; j < m; j++) { 
           double val1 = f_eval(&deviceInput[currentThread*m], m, j, h);
           double val2 = f_eval(&deviceInput[currentThread*m], m, j, -h);

           double result = (val1 - val2) / (2*epsilon);
           deviceResult[currentThread*m + j] = result;
        } 
    }
}

int main(int argc, char* argv[]) {
    string ipFile(argv[1]);
    string opFile(argv[2]);
    double epsilon = atof(argv[3]);

    const double h = 1e-2;

    int m, n;
    string line;

    ifstream ifile;
    ifile.open(ipFile);

    getline(ifile, line);
    n = stoi(line);

    getline(ifile, line);
    m = stoi(line);

    int nElements = m * n;
    double *hostInput = new double[m * n];
    double *hostResult = new double[m * n];
   
    // Read all the input points from the file
    for (int i = 0; getline(ifile, line);) {
        string number;
        stringstream s(line);

        while(getline(s, number, ',')) {
            hostInput[i] = stod(number);
            i++;
        }
    }
    
    ifile.close();

    // Setting up timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    double *deviceInput, *deviceResult;

    cudaError_t error;

    error = cudaMalloc((void**) &deviceInput, nElements * sizeof(double));
    if (error != cudaSuccess)
    {
         printf("cudaMalloc returned error code %d: %s, line(%d)\n", error, cudaGetErrorString(error), __LINE__);
         exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &deviceResult, nElements * sizeof(double));
    if (error != cudaSuccess)
    {
         printf("cudaMalloc returned error code %d: %s, line(%d)\n", error, cudaGetErrorString(error),  __LINE__);
         exit(EXIT_FAILURE);
    }

    cudaMemcpy(deviceInput, hostInput, nElements * sizeof(double), cudaMemcpyHostToDevice);

    int totalBlocks = (n + 1023)/1024;
    int threadsPerBlock = min(n, 1024);

    jacobian<<<totalBlocks, threadsPerBlock>>>(deviceInput, deviceResult, m, n, h, epsilon);

    cudaMemcpy(hostResult, deviceResult, nElements * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    ofstream ofile;
    ofile.open(opFile);
//    if( (m == 4096) && (n == 16384) ){
       for(int i = 0; i < n; i++) {
           for(int j = 0; j < m; j++) {
               if(j != m-1)
                   ofile << fixed << setprecision(6) << hostResult[i*m + j] << ",";
               else
                   ofile << fixed << setprecision(6) << hostResult[i*m + j] << endl;
           }
       }
//    }
    ofile.close();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
   
    // Create a new file to log the execution time    
    ofstream ologfile;
    ologfile.open("cudaBase2Log", ios_base::app);
    ologfile <<  n << " " <<  m << " " << fixed << setprecision(6) << elapsedTime << endl;
    ologfile.close();

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(deviceInput);
    cudaFree(deviceResult);
    delete[] hostInput, hostResult;

    return 0;
}
