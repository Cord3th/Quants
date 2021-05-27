#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "./logic_gate.h"

typedef std::complex<double> complexd;
using namespace std;

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

void FileWrite(vector<complexd>& b, long vec_length,
               int rank, int n, char* file_name) {
	MPI_File file;
	MPI_Status status;
	MPI_Datatype filetype;

	int s = 1, p = 0;
	MPI_Type_create_subarray(1, &s, &s, &p, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
	int offset = sizeof(int) + vec_length * 2 * rank * sizeof(double);

	if (rank == 0) {
		MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
		MPI_File_write(file, &n, 1, MPI_INT, &status);
		MPI_File_close(&file);
	}

	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
	MPI_File_set_view(file, offset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(file, b.data(), vec_length * 2, MPI_DOUBLE, &status);
	MPI_File_close(&file);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        cout << "Input: <n> <q1> <q2> <mode> "
             << "<numthreads> <out.bin>" << endl;
        return 0;
    }
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int n = atoi(argv[1]), q1 = atoi(argv[2]), q2 = atoi(argv[3]),
        mode = atoi(argv[4]), numthreads = atoi(argv[5]);
    omp_set_num_threads(numthreads);

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

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int proc_exp = log2(size);

    long vec_length = 1 << (n - proc_exp);
    vector<complexd> a(vec_length);

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

    vector<complexd> b = N_hadamar(a, n, proc_exp, rank);

    FileWrite(b, vec_length, rank, n, argv[6]);

    MPI_Finalize();
    return 0;
}
