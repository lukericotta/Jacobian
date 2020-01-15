#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "nvToolsExt.h"
#include <cuda.h>

#include "f_eval.cuh"

using namespace std;

__global__ void jacobian(double *deviceInput, double *deviceResult, const int m, const int n, const double h, const double epsilon)
{
	// number of blocks requried to process each point
    int blocksPerPoint = (m + 1023) / 1024;

    // idx in global deviceInput to start copying data from into local memory
    int startBlockIdx = (blockIdx.x / blocksPerPoint) * m;

    // idx in global deviceInput till where to copy data into local memory
    int endBlockIdx = startBlockIdx + m;

    // int elementsToCopyPerThread = (m + 1023) / 1024;

    // int threadStartCopyIdx = startBlockIdx + (threadIdx.x * elementsToCopyPerThread);

    // int threadEndCopyIdx = threadStartCopyIdx + elementsToCopyPerThread;

    int currentThread = ((blockIdx.x % blocksPerPoint) * blockDim.x) + threadIdx.x;

    

    

    

    

    // double *localPlus = (double*) malloc (m * sizeof(double));
    // double *localMinus = (double*) malloc (m * sizeof(double));

    

    if(currentThread < m) {

        // TODO: fix this - malloc() returning NULL pointer
        double *localX = (double*) malloc (m * sizeof(double));

        // if(localX == 0) {
        //     printf("You're screwed!\n");
        // }

        for(int i = startBlockIdx, idx = 0; i < endBlockIdx; i++, idx++) {
            localX[idx] = deviceInput[i];
        }

        localX[currentThread] += h;
        double val1 = f_eval(localX, m);

        localX[currentThread] -= 2*h;
        double val2 = f_eval(localX, m);

        double result = (val1 - val2) / (2*epsilon);

        deviceResult[startBlockIdx + currentThread] = result;

        free(localX);
    }

    
}

int main(int argc, char* argv[]) {

    size_t freeDeviceMemory, totalDeviceMemory;
    cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);

    // currently allocating half of free device memory to the heap
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, freeDeviceMemory / 2);

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

    int blocksPerPoint = (m + 1023) / 1024;
    int totalBlocks = n * blocksPerPoint;
    int threadsPerBlock = min(m, 1024);

    jacobian<<<totalBlocks, threadsPerBlock>>>(deviceInput, deviceResult, m, n, h, epsilon);

    cudaDeviceSynchronize();

    cudaMemcpy(hostResult, deviceResult, nElements * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    ofstream ofile;
    ofile.open(opFile);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            if(j != m-1)
                ofile << hostResult[i*m + j] << ",";
            else
                ofile << hostResult[i*m + j];
        }
        ofile << endl;
    }
    
    ofile.close();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
   
    // Create a new file to log the execution time    
    ofstream ologfile;
    ologfile.open("cudaBaseLog", ios_base::app);
    ologfile << left << setw(10) << n << "\t" << left << setw(10) << m << "\t" << left << setw(10) << setprecision(10) << elapsedTime << endl;
    ologfile.close();

    cudaFree(deviceInput);
    cudaFree(deviceResult);
    delete[] hostInput, hostResult;

    return 0;
}
