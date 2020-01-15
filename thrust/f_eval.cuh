#include <thrust/device_ptr.h>

inline __host__ __device__ double f_eval(thrust::device_ptr<double> p_x, int m)
{
	double result = 0.0f;
	for(int i = 0; i < m; i++) {
		result += ((i+1) * p_x[i]);
	}
	return result;
}
