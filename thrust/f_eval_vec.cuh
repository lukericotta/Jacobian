inline __host__ __device__ double f_eval_vec(thrust::device_vector<double> p_x, int m)
{
	double result = 0.0f;
	for(int i = 0; i < m; i++) {
		result += ((i+1) * p_x[i]);
	}
	return result;
}
