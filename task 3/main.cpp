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
 		 temp = 1 << (n - k),
 		 start_idx = vec_length * rank;
	vector<complexd> b(vec_length);
    if (temp < vec_length) {
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
            b[i] = u[((i + start_idx) & temp) >> (n - k)][0]
                   * a[(((i + start_idx) | temp) ^ temp) - start_idx]
                 + u[((i + start_idx) & temp) >> (n - k)][1]
                   * a[ ((i + start_idx) | temp) - start_idx];
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
        #pragma omp parallel for
        for (int i = 0; i < vec_length; ++i) {
            b[i] = u[((i + start_idx) & temp) >> (n - k)][0] * vec_0[i]
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


int main(int argc, char **argv) {
    if (argc != 6) {
        cout << "Input: <n> <k> <mode> "
			 << "(1 - .bin extension file, 2 - random)"
             << " <numthreads> <eps>" << endl;
        return 0;
    }
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    //double a_start_time, b_start_time, a_end_time, b_end_time;
    int n = atoi(argv[1]), k = atoi(argv[2]),
        numthreads = atoi(argv[3]), mode = atoi(argv[4]);
    double eps = atof(argv[5]);
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
    MPI_Status status;
    MPI_Datatype filetype;
    MPI_File file;

    int proc_exp = log2(size);

    long vec_length = 1 << (n - proc_exp);
    vector<complexd> a(vec_length);

    double time_tmp;
    if (rank == 0) {
        time_tmp = MPI_Wtime();
    }
    MPI_Bcast(&time_tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //a_start_time = MPI_Wtime();

    if (mode == 2) {
        double norm_length_ = 0, temp;
        #pragma omp parallel
        {
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
        double norm_length_ = 0;
        for (int i = 0; i < vec_length; ++i) {
            norm_length_ += norm(a[i]);
        }
        double temp;
        MPI_Allreduce(&norm_length_, &temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm_length_ = sqrt(temp);
        for (int i = 0; i < vec_length; ++i) {
            a[i] = a[i] / norm_length_;
        }
    }

    double sum = 0;

    vector<vector<complexd>> u_noisy(2);

	for (size_t i = 0; i < 2; ++i) {
		u_noisy[i].resize(2);
	}

    vector<complexd> b_norm = QubitTransform(a, n, u, 1, proc_exp, rank);

    vector<complexd> temp;
    for (int i = 2; i <= n; ++i) {
        temp = b_norm;
        b_norm = QubitTransform(temp, n, u, i, proc_exp, rank);
    }

    for (int j = 0; j < 60; ++j) {
        double ksi[n];
        if (rank == 0) {
            for (int i = 0; i < n; ++i) {
                ksi[i] = eps * NormalDisGen();
            }
        }
        MPI_Bcast(ksi, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //b_start_time = omp_get_wtime();
        MakeNoisy(u, u_noisy, ksi[0]);

        vector<complexd> b = QubitTransform(a, n, u_noisy, 1, proc_exp, rank);

        for (int i = 2; i <= n; ++i) {
            temp = b;
            MakeNoisy(u, u_noisy, ksi[i - 1]);
            b = QubitTransform(temp, n, u_noisy, i, proc_exp, rank);
        }

        complexd c = 0;
        for (int i = 0; i < vec_length; ++i) {
            c += abs(b[i] * conj(b_norm[i]));
        }
        c = c;
        double sum_tmp = c.real();
        sum += c.real();

        double res;
        MPI_Allreduce(&sum_tmp, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            fstream fout("answer", ios::out | ios::app);
            fout << 1 -  res * res << endl;
            fout.close();
        }
    }
    //cout << 1 -  sum / 90 << endl;
    //b_end_time = omp_get_wtime();

    /*double time1 = a_end_time - a_start_time;
    double time2 = b_end_time - b_start_time;
    double timelocal = time1 + time2;
    double sumtime1, sumtime2, maxtime;
    MPI_Reduce(&time1, &sumtime1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time2, &sumtime2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timelocal, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "sumtime1: " << sumtime1 <<" sumtime2: " << sumtime2 << " maxtime: " << maxtime << endl;
    }*/

    //file output
    /*
    int s = 1;
    int p = 0;
    MPI_Type_create_subarray(1,  &s, &s, &p, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    int offset = sizeof(int) + vec_length * 2 * rank * sizeof(double);
    if( rank == 0) {
        MPI_File_open(MPI_COMM_SELF, "out.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
        MPI_File_write(file, &n, 1, MPI_INT, &status);
        MPI_File_close(&file);
    }
    MPI_File_open(MPI_COMM_WORLD, "out.bin", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, offset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(file, b, vec_length * 2, MPI_DOUBLE, &status);
    MPI_File_close(&file);
    */

    MPI_Finalize();
    return 0;
}
