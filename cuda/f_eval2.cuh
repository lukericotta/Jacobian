inline __host__ __device__ double f_eval(double* p_x, int m, int index, double h)
{
	double result = 0.0f;
	for(int i = 0; i < m; i++){
      if(i == index){
   		result += ((i+1) * p_x[0] * pow((p_x[i] + h),2));
      }
      else{
   		result += ((i+1) * p_x[0] * pow((p_x[i]),2));
      }         
	}
	return result;
}
