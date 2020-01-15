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

// x - m-dimensional point
// n - total number of points
__global__ void jacobian(double *deviceInput, double *deviceResult, const int m, const int n, const double h, const double epsilon)
{

    // number of blocks requried to process each point
    int blocksPerPoint = (m + 1023) / 1024;
    
    // idx in global deviceInput to start copying data from into shared memory
    int startBlockIdx = (blockIdx.x / blocksPerPoint) * m;

    // idx in global deviceInput till where to copy data into shared memory
    int endBlockIdx = startBlockIdx + m;

    //  we have to copy from deviceInput[startBlockIdx] to deviceInput[endBlockIdx].

    // int elementsToCopy = m;
    // int threadsPerBlock = blockDim.x;

    int elementsToCopyPerThread = (m + 1023) / 1024;

    int threadStartCopyIdx = startBlockIdx + (threadIdx.x * elementsToCopyPerThread);

    int threadEndCopyIdx = threadStartCopyIdx + elementsToCopyPerThread;

    int i = threadStartCopyIdx;
    int j = (threadIdx.x * elementsToCopyPerThread);

    extern __shared__ double sharedX[];
    while(i < threadEndCopyIdx && i < endBlockIdx) {
        sharedX[j] = deviceInput[i];
        i++; j++;
    }

    __syncthreads();

    // if (m <= threadsPerBlock) {
    //     // bring in utmost one thing
    // } else {
    //     // bring in more than one thing
    // }

    // int startIdx = threadIdx.x * nEachThread;
    // int endIdx = startIdx + nEachThread;

    // for(int i = startIdx; i < endIdx; i++) {
	   // sharedX[i] = x[i];
    // }

    

    
    
    // TODO: fix this - malloc() returning NULL pointer
    // if(localX == 0) {
    //     printf("You're screwed!\n");
    // }

    // where to do the xi + h and xi - h? -- can't touch sharedX
    
    
    int currentThread = ((blockIdx.x % blocksPerPoint) * blockDim.x) + threadIdx.x;

    if(currentThread < m) {

        double *localX = (double*) malloc (m * sizeof(double));

        for(int i = 0; i < m; i++)
            localX[i] = sharedX[i];

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

    // Setting up timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

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

    for (int i = 0; getline(ifile, line);) {
        
        string number;
        stringstream s(line);

        while(getline(s, number, ',')) {
            hostInput[i] = stod(number);
            i++;
        }
    }

    ifile.close();

    // for(int i = 0; i < nElements; i++)
    //     cout << hostInput[i] << " ";
    
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

    // copy stuff
    cudaMemcpy(deviceInput, hostInput, nElements * sizeof(double), cudaMemcpyHostToDevice);

    int blocksPerPoint = (m + 1023) / 1024;
    int totalBlocks = n * blocksPerPoint;
    int threadsPerBlock = min(m, 1024);

    // call kernel
    jacobian<<<totalBlocks, threadsPerBlock, 48000>>>(deviceInput, deviceResult, m, n, h, epsilon);

    cudaDeviceSynchronize();
    // copy back
    cudaMemcpy(hostResult, deviceResult, nElements * sizeof(double), cudaMemcpyDeviceToHost);    

    // fix kernel


    // write to file
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

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    ofile.close();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
   
    // Create a new file to log the execution time
    ofstream ologfile;
    ologfile.open("cudaSharedLog", ios_base::app);
    ologfile << left << setw(10) << n << "\t" << left << setw(10) << m << "\t" << left << setw(10) << setprecision(10) << elapsedTime << endl;
    ologfile.close();

    cudaFree(deviceInput);
    cudaFree(deviceResult);
    delete[] hostInput, hostResult;

    return 0;
}
