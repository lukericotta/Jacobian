double f_eval(double* x, int m, int index, double h) {
    double result = 0.0;
    for(int i = 0; i < m; i++) {
	if(i == index) {
	    result += ((i+1) * x[0] * pow((x[i] + h), 2));
	} else {
	    result += ((i+1) * x[0] * pow(x[i], 2));
	}
    }
    return result;
}
