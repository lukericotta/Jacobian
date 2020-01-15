double f_eval(double* p_x, int m)
{
	double result = 0.0f;
	for(int i = 0; i < m; i++) {
		result += ((i+1) * p_x[i]);
	}
	return result;
}
