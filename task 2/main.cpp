#include <iostream>
#include <complex>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include "mpi.h"
#include <vector>

typedef std::complex<double> complexd;
using namespace std;

vector<complexd> QubitTransform(vector<complexd>& a, int n,
								vector<vector<complexd>>& u, int k,
								long proc_exp, int rank) {
	long vec_length = 1 << (n - proc_exp),
		 temp = 1 << (n - k),
		 start_idx = vec_length * rank;
	vector<complexd> b(vec_length);
	if (temp < vec_length) {
		for (long i = 0; i < vec_length; ++i) {
			b[i] = u[(i + start_idx) & temp >> (n - k)][0]
				   * a[(((i + start_idx) | temp) ^ temp) - start_idx]
				 + u[((i + start_idx) & temp) >> (n - k)][1]
				   * a[((i + start_idx) | temp) - start_idx];
		}
	} else {
		int dest_src_rank;
		if ((start_idx & temp) == 0) {
			dest_src_rank = (start_idx | temp) / vec_length;
		} else {
			dest_src_rank = (start_idx & ~temp) / vec_length;
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
		for (long i = 0; i < vec_length; ++i) {
			b[i] = u[(i + start_idx) & temp >> (n - k)][0] * vec_0[i]
				 + u[((i + start_idx) & temp) >> (n - k)][1] * vec_1[i];
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

void FileWrite(vector<complexd>& b, long vec_length, int rank, int n) {
	MPI_File file;
	MPI_Status status;
	MPI_Datatype filetype;

	int s = 1, p = 0;
	MPI_Type_create_subarray(1, &s, &s, &p, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
	int offset = sizeof(int) + vec_length * 2 * rank * sizeof(double);

	if (rank == 0) {
		MPI_File_open(MPI_COMM_SELF, "out.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
		MPI_File_write(file, &n, 1, MPI_INT, &status);
		MPI_File_close(&file);
	}

	MPI_File_open(MPI_COMM_WORLD, "out.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
	MPI_File_set_view(file, offset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(file, b.data(), vec_length * 2, MPI_DOUBLE, &status);
	MPI_File_close(&file);
}

int main(int argc, char **argv) {
	if (argc != 4) {
		cout << "Input: <n> <k> <mode> "
			 << "(1 - .bin extension file, 2 - random)"
			 << endl;
		return 0;
	}

	double a_start_time, b_start_time,
		   a_end_time, b_end_time;
	int n = atoi(argv[1]), k = atoi(argv[2]),
	 	   mode = atoi(argv[3]);

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
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int proc_exp = log2(size);

    long vec_length = 1 << (n - proc_exp);
	vector<complexd> a(vec_length);

	double time_tmp;
	if (rank == 0) {
		time_tmp = MPI_Wtime();
	}
	MPI_Bcast(&time_tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	a_start_time = MPI_Wtime();
	if (mode == 2) {
		double norm_length_ = 0, temp;
		srand(time_tmp * (rank + 1));

		for (long i = 0; i < vec_length; ++i) {
			a[i] = complexd(((double) rand()) / RAND_MAX,
							((double) rand()) / RAND_MAX);
			norm_length_ += norm(a[i]);
		}
		MPI_Allreduce(&norm_length_, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		norm_length_ = sqrt(temp);
		for (long i = 0; i < vec_length; ++i) {
			a[i] = a[i] / norm_length_;
		}

		a_end_time = MPI_Wtime();
	} else {
		a_end_time = MPI_Wtime();
		FileRead(a, vec_length, rank);
	}

	b_start_time = MPI_Wtime();
	vector<complexd> b = QubitTransform(a, n, u, k, proc_exp, rank);
	b_end_time = MPI_Wtime();


	double time1 = a_end_time - a_start_time;
	double time2 = b_end_time - b_start_time;
	double timelocal = time1 + time2;
	double sumtime1, sumtime2, maxtime;
	MPI_Reduce(&time1, &sumtime1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time2, &sumtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&timelocal, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		cout << "sumtime1: " << sumtime1 <<" sumtime2: " << sumtime2 << " maxtime: " << maxtime << endl;
	}

	FileWrite(b, vec_length, rank, n);

	MPI_Finalize();
	return 0;
}
