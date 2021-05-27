#include <iostream>
#include <complex>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include "mpi.h"
#include <omp.h>
#include <vector>

typedef std::complex<double> complexd;
using namespace std;

double NormalDisGen() {
    double S = 0.;

    for (int i = 0; i < 12; ++i) {
        S += (double) rand() / RAND_MAX;
    }

    return S - 6.;
}

void MakeNoisy(const vector<vector<complexd>>& u,
            vector<vector<complexd>>& u_noisy, double ksi) {
    u_noisy[0][0] = u[0][0] * cos(ksi) - u[0][1] * sin(ksi);
    u_noisy[0][1] = u[0][0] * sin(ksi) + u[0][1] * cos(ksi);
    u_noisy[1][0] = u[1][0] * cos(ksi) - u[1][1] * sin(ksi);
    u_noisy[1][1] = u[1][0] * sin(ksi) + u[1][1] * cos(ksi);
}

vector<complexd> QubitTransform(vector<complexd>& a, int n,
                                 vector<vector<complexd>>& u, int k,
                                 long proc_exp, int rank) {
    long vec_length = 1 << (n - proc_exp),
 		 shift = 1 << (n - k),
 		 start_idx = vec_length * rank;
	vector<complexd> b(vec_length);
    if (shift < vec_length) {
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
        b[i] = u[((i + start_idx) & shift) >> (n - k)][0]
               * a[(((i + start_idx) | shift) ^ shift) - start_idx]
             + u[((i + start_idx) & shift) >> (n - k)][1]
               * a[((i + start_idx) | shift) - start_idx];
        }
    } else {
        int dest_src_rank;
        if ((start_idx & shift) == 0) {
            dest_src_rank = (start_idx | shift) / vec_length;
        } else {
            dest_src_rank = (start_idx & ~shift) / vec_length;
        }
		vector<complexd> tmp(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, tmp.data(), vec_length * 2,
                     MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        vector<complexd> vec_0, vec_1;
        if (rank < dest_src_rank) {
            vec_0 = a;
            vec_1 = tmp;
        } else {
            vec_0 = tmp;
            vec_1 = a;
        }
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
            b[i] = u[((i + start_idx) & shift) >> (n - k)][0] * vec_0[i]
            + u[((i + start_idx) & shift) >> (n - k)][1] * vec_1[i];
        }
    }
    return b;
}

void FileRead(vector<complexd>& a, long vec_length, int rank) {
	MPI_File file;
	MPI_Status status;
	MPI_Datatype filetype;

	int s = 1, p = 0;
	MPI_Type_create_subarray(1, &s, &s, &p, MPI_ORDER_C, MPI_DOUBLE, &filetype);
	MPI_Type_commit(&filetype);
	int offset = sizeof(int) + vec_length * 2 * rank * sizeof(double);

	MPI_File_open(MPI_COMM_WORLD, "in.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	MPI_File_set_view(file, offset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
	MPI_File_read_all(file, a.data(), vec_length * 2, MPI_DOUBLE, &status);
	MPI_File_close(&file);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        cout << "Input: <n> <k> <mode> <numthreads> <eps>" << endl;
        return 0;
    }
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    double a_start_time, b_start_time, a_end_time, b_end_time;
    int n = atoi(argv[1]), k = atoi(argv[2]),
        mode = atoi(argv[3]), numthreads = atoi(argv[4]);
    double eps = atof(argv[5]), sum = 0;
    omp_set_num_threads(numthreads);

    vector<vector<complexd>> u(2);
	for (size_t i = 0; i < 2; ++i) {
		u[i].resize(2);
	}
    u[0][0] = u[0][1] = u[1][0] = 1.0 / sqrt(2.0);
    u[1][1] = -u[0][0];

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Datatype filetype;
    MPI_File file;

    int proc_exp = log2(size);

    long vec_length = 1 << (n - proc_exp);
    vector<complexd> a(vec_length);

    a_start_time = MPI_Wtime();

    if (mode == 2) {
        double norm_length_ = 0, temp;
        #pragma omp parallel
    	{
    		#pragma omp for reduction(+ : norm_length_)
            for (int i = 0; i < vec_length; ++i) {
                a[i] = complexd(((double) rand()) / RAND_MAX,
    							((double) rand()) / RAND_MAX);
                norm_length_ += norm(a[i]);
            }
        }
        MPI_Allreduce(&norm_length_, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm_length_ = sqrt(temp);
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
            a[i] = a[i] / norm_length_;
        }
    } else {
        FileRead(a, vec_length, rank);
        double norm_length_ = 0, temp;
        for (int i = 0; i < vec_length; ++i) {
            norm_length_ += norm(a[i]);
        }
        MPI_Allreduce(&norm_length_, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm_length_ = sqrt(temp);
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
            a[i] = a[i] / norm_length_;
        }
    }
	a_end_time = MPI_Wtime();

    vector<vector<complexd>> u_noisy(2);

	for (size_t i = 0; i < 2; ++i) {
		u_noisy[i].resize(2);
	}

    b_start_time = MPI_Wtime();
    vector<complexd> temp;
    /*vector<complexd> b = QubitTransform(a, n, u, 1, proc_exp, rank);

    for (int i = 2; i <= n; ++i) {
        temp = b;
        b = QubitTransform(temp, n, u, i, proc_exp, rank);
    }*/

    //for (int j = 0; j < 60; ++j) {
        double ksi[n];
        if (rank == 0) {
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                ksi[i] = eps * NormalDisGen();
            }
        }
        MPI_Bcast(ksi, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MakeNoisy(u, u_noisy, ksi[0]);

        vector<complexd> b_noisy = QubitTransform(a, n, u_noisy, 1, proc_exp, rank);

        for (int i = 2; i <= n; ++i) {
            temp = b_noisy;
            MakeNoisy(u, u_noisy, ksi[i - 1]);
            b_noisy = QubitTransform(temp, n, u_noisy, i, proc_exp, rank);
        }
/*
        complexd c = 0;
        for (int i = 0; i < vec_length; ++i) {
            c += abs(b_noisy[i] * conj(b[i]));
        }

        double sum_tmp = c.real(), res;
        sum += c.real();

        MPI_Allreduce(&sum_tmp, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            fstream fout("answer", ios::out | ios::app);
            fout << 1 -  res * res << endl;
            fout.close();
        }*/
    //}

    b_end_time = MPI_Wtime();

	double a_time = a_end_time - a_start_time,
		   b_time = b_end_time - b_start_time;
	double local_sum = a_time + b_time, max_time;
	MPI_Reduce(&local_sum, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		cout << max_time << endl;
	}

    MPI_Finalize();

    return 0;
}
