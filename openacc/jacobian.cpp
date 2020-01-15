#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string.h>

#include <accelmath.h>
#include <openacc.h>

#include <chrono>
#include <ratio>

#include "f_eval2.h"

using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char* argv[]) {

    double epsilon = atof(argv[3]);
    
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> duration_msec;

    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(argv[1], "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    const double h = 0.01;
    int m, n;
    
    read = getline(&line, &len, fp);
    n = stoi(line);

    read = getline(&line, &len, fp);
    m = stoi(line);

    double *input = new double[n * m];

    int ctr = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
	
	char *ptr = NULL;
	if((ptr = strchr(line, '\n')) != NULL) {
	    *ptr = '\0';
	}
	
	char *n = strtok(line, ",");
	while(n != NULL) {
	    input[ctr++] = atof(n);
	    n = strtok(NULL, ",");
	}

	line = NULL;
    }
    fclose(fp);
    
    double *result = new double[n * m];

    start = high_resolution_clock::now();
    
#pragma acc parallel loop
    for(int i = 0; i < n; i++) {
	for(int j = 0; j < m; j++) {
	    double val1 = f_eval((input + i*m), m, j, h);
	    double val2 = f_eval((input + i*m), m, j, -h);
	    double r = (val1 - val2) / (2*epsilon);
	    result[i*m + j] = r;
	}
    }

    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    // fp = fopen("openaccTiming.log", "a+");
    // if (fp == NULL)
    // 	exit(EXIT_FAILURE);
    
    // fprintf(fp, "%d %d %.6lf\n", n, m, duration_msec.count());

    // fclose(fp);

    fp = fopen(argv[2], "w");
    if (fp == NULL)
    	exit(EXIT_FAILURE);

    for(int i = 0; i < n; i++) {
    	for(int j = 0; j < m; j++) {
    	    fprintf(fp, "%.6lf", result[i*m + j]); 
    	    if (j != m-1)
    		fprintf(fp, ",");
    	}
    	fprintf(fp, "\n");
    }
    
    fclose(fp);

    delete[] input;
    delete[] result;

    return 0;
}
