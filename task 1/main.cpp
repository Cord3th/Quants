#include <omp.h>
#include <iostream>
#include <complex>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <vector>

typedef std::complex<double> complexd;

using namespace std;

vector<complexd> single_qubit_transform(vector<complexd>& a, vector<vector<complexd>>& u,
								  size_t n, size_t k) {
	size_t num_qubits = 1 << n, temp = 1 << (n - k);
	vector<complexd> b(num_qubits);

	#pragma omp parallel
	{
		#pragma omp for
		for (size_t i = 0; i < num_qubits; ++i) {
			b[i] = u[(i & temp) >> (n - k)][0] * a[(i | temp) ^ temp]
			+ u[(i & temp) >> (n - k)][1] * a[i | temp];
		}
	}

	return b;
}

int main(int argc, char **argv) {
	if (argc != 4) {
		cerr << "Input: <n> <k> <numthreads>" << endl;
		return 0;
	}

	double a_start_time, b_start_time, a_end_time, b_end_time;
	size_t n = atoi(argv[1]), k = atoi(argv[2]), numthreads = atoi(argv[3]);
	omp_set_num_threads(numthreads);
	size_t num_qubits = 1 << n;

	vector<complexd> a(num_qubits);
	vector<vector<complexd>> u(2);
	for (size_t i = 0; i < 2; ++i) {
		u[i].resize(2);
	}
	for (size_t i = 0; i < 2; ++i) {
		for (size_t j = 0; j < 2; ++j) {
				u[i][j] = 1.0 / sqrt(2);
		}
	}
	u[1][1] *= -1;

	a_start_time = omp_get_wtime();

	double norm_length_ = 0;
	srand(omp_get_wtime());
	int temp_seed = rand();
	#pragma omp parallel
	{
		unsigned int seed = temp_seed * (omp_get_thread_num() + 1);
		#pragma omp for reduction(+ : norm_length_)
		for (size_t i = 0; i < num_qubits; ++i) {
			a[i] = complexd(((double) rand_r(&seed)) / RAND_MAX,
							((double) rand_r(&seed)) / RAND_MAX);
			norm_length_ += norm(a[i]);
		}
	}
	norm_length_ = sqrt(norm_length_);
	#pragma omp for
	for (size_t i = 0; i < num_qubits; ++i) {
		a[i] = a[i] / norm_length_;
	}

	a_end_time = omp_get_wtime();
	b_start_time = omp_get_wtime();

	vector<complexd> b = single_qubit_transform(a, u, n, k);
	b_end_time = omp_get_wtime();
	cout << "Work time: " << a_end_time - a_start_time + b_end_time - b_start_time << endl;

	ofstream out("out.txt");
	for (size_t i = 0; i < num_qubits; ++i) {
		out << i << ' ' << b[i] << ' ' << endl;
	}
	out.close();

	return 0;
}
