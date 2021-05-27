#include <omp.h>
#include <mpi.h>
#include <complex>
#include <vector>
#ifndef SEM6_TASK4_QUANTUM_LOGIC_GATE__LOGIC_GATE_H_
#define SEM6_TASK4_QUANTUM_LOGIC_GATE__LOGIC_GATE_H_
#endif   //  SEM6_TASK4_QUANTUM_LOGIC_GATE__LOGIC_GATE_H_

using namespace std;
typedef std::complex<double> complexd;

vector<complexd> QubitTransform(vector<complexd>& a, int n,
                                vector<vector<complexd>>& u,
                                int k, int proc_exp, int rank) {
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


vector<complexd> TwoQubitsEvolution(vector<complexd>& a, int n, int q1, int q2,
                                    vector<vector<complexd>>& u, int proc_exp, int rank) {
    int shift1 = 1 << (n - q1),
        shift2 = 1 << (n - q2);

    long vec_length = 1 << (n - proc_exp),
         start_idx = vec_length * rank;
    vector<complexd> b(vec_length);

    if (shift1 < vec_length && shift2 < vec_length) {
        #pragma omp parallel for
        for (int i1 = 0; i1 < vec_length; ++i1) {
            int i = i1 + start_idx,
                i00 = (i & ~shift1 & ~shift2) - start_idx,
                i01 = (i & ~shift1 | shift2) - start_idx,
                i10 = ((i | shift1) & ~shift2) - start_idx,
                i11 = (i | shift1 | shift2) - start_idx;

            int iq1 = (i & shift1) >> n - q1,
                iq2 = (i & shift2) >> n - q2;

            int iq = (iq1 << 1) + iq2;
            b[i1] = u[iq][(0 << 1) + 0] * a[i00] + u[iq][(0 << 1) + 1] * a[i01]
                  + u[iq][(1 << 1) + 0] * a[i10] + u[iq][(1 << 1) + 1] * a[i11];
        }
        return b;
    } else if (shift1 < vec_length && shift2 >= vec_length) {
        int dest_src_rank;
        if ((start_idx & shift2) == 0) {
            dest_src_rank = (start_idx | shift2) / vec_length;
        } else {
            dest_src_rank = (start_idx & ~shift2) / vec_length;
        }

        vector<complexd> tmp(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, tmp.data(), vec_length * 2,
                    MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<complexd> vecq2_0, vecq2_1;
        if (rank < dest_src_rank) {
            vecq2_0 = a;
            vecq2_1 = tmp;
        } else {
            vecq2_0 = tmp;
            vecq2_1 = a;
        }

        #pragma omp parallel for
        for (int i1 = 0; i1 < vec_length; ++i1) {
            int i = i1 + start_idx,
                i00 = (i & ~shift1) - start_idx,
                i01 = (i & ~shift1) - start_idx,
                i10 = (i | shift1) - start_idx,
                i11 = (i | shift1) - start_idx;

            int iq1 = (i & shift1) >> n - q1,
                iq2 = (i & shift2) >> n - q2;

            int iq = (iq1 << 1) + iq2;
            b[i1] = u[iq][(0 << 1) + 0] * vecq2_0[i00] + u[iq][(0 << 1) + 1] * vecq2_1[i01]
                  + u[iq][(1 << 1) + 0] * vecq2_0[i10] + u[iq][(1 << 1) + 1] * vecq2_1[i11];
        }
        return b;
    } else if (shift2 < vec_length && shift1 >= vec_length) {
        int dest_src_rank;
        if ((start_idx & shift1) == 0) {
            dest_src_rank = (start_idx | shift1) / vec_length;
        } else {
            dest_src_rank = (start_idx & ~shift1) / vec_length;
        }
        vector<complexd> tmp(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, tmp.data(), vec_length * 2,
                    MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<complexd> vecq1_0, vecq1_1;
        if (rank < dest_src_rank) {
            vecq1_0 = a;
            vecq1_1 = tmp;
        } else {
            vecq1_0 = tmp;
            vecq1_1 = a;
        }
        #pragma omp parallel for
        for (int i1 = 0; i1 < vec_length; ++i1) {
            int i = i1 + start_idx,
                i00 = (i & ~shift2) - start_idx,
                i01 = (i | shift2) - start_idx,
                i10 = (i & ~shift2) - start_idx,
                i11 = (i | shift2) - start_idx;

            int iq1 = (i & shift1) >> n - q1,
                iq2 = (i & shift2) >> n - q2;

            int iq = (iq1 << 1) + iq2;
            b[i1] = u[iq][(0 << 1) + 0] * vecq1_0[i00] + u[iq][(0 << 1) + 1] * vecq1_0[i01]
                  + u[iq][(1 << 1) + 0] * vecq1_1[i10] + u[iq][(1 << 1) + 1] * vecq1_1[i11];
        }
        return b;
    } else if (shift1 >= vec_length && shift2 >= vec_length) {
        int dest_src_rank,
            value00 = start_idx & ~shift1 & ~shift2;
        dest_src_rank = value00 / vec_length;
        vector<complexd> vec00(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, vec00.data(), vec_length * 2,
                     MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int value01 = start_idx & ~shift1 | shift2;
        dest_src_rank = value01 / vec_length;
        vector<complexd> vec01(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, vec01.data(), vec_length * 2,
            MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int value10 = (start_idx | shift1) & ~shift2;
        dest_src_rank = value10 / vec_length;
        vector<complexd> vec10(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, vec10.data(), vec_length * 2,
            MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int value11 = start_idx | shift1 | shift2;
        dest_src_rank = value11 / vec_length;
        vector<complexd> vec11(vec_length);
        MPI_Sendrecv(a.data(), vec_length * 2, MPI_DOUBLE, dest_src_rank, 0, vec11.data(), vec_length * 2,
            MPI_DOUBLE, dest_src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for
        for (int i1 = 0; i1 < vec_length; i1++) {
            int i = i1 + start_idx;
            int iq1 = (i & shift1) >> n - q1,
                iq2 = (i & shift2) >> n - q2;

            int iq = (iq1 << 1) + iq2;
            b[i1] = u[iq][(0 << 1) + 0] * vec00[i1] + u[iq][(0 << 1) + 1] * vec01[i1]
                  + u[iq][(1 << 1) + 0] * vec10[i1] + u[iq][(1 << 1) + 1] * vec11[i1];
        }
        return b;
    }
    return b;
}

vector<complexd> Hadamar(vector<complexd>& a, int n, int k,
                         int proc_exp, int rank) {
    vector<vector<complexd>> u(2);
 	for (size_t i = 0; i < 2; ++i) {
 		u[i].resize(2);
 	}
    u[0][0] = u[0][1] = u[1][0] = 1.0 / sqrt(2.0);
    u[1][1] = -u[0][0];
    return QubitTransform(a, n, u, k, proc_exp, rank);
}

vector<complexd> N_hadamar(vector<complexd>& a, int n,
                           int proc_exp, int rank) {
    vector<vector<complexd>> u(2);
    for (size_t i = 0; i < 2; ++i) {
        u[i].resize(2);
    }
    u[0][0] = u[0][1] = u[1][0] = 1.0 / sqrt(2.0);
    u[1][1] = -u[0][0];
    vector<complexd> temp = QubitTransform(a, n, u, 1, proc_exp, rank);
    vector<complexd> b;
    for (int i = 2; i <= n; ++i) {
        b = QubitTransform(temp, n, u, i, proc_exp, rank);
        temp = b;
    }
    return b;
}

vector<complexd> Not(vector<complexd>& a, int n, int k,
                     int proc_exp, int rank) {
    vector<vector<complexd>> u(2);
 	for (size_t i = 0; i < 2; ++i) {
 		u[i].resize(2);
 	}
    u[0][0] = u[1][1] = 0;
    u[0][1] = u[1][0] = 1;
    return QubitTransform(a, n, u, k, proc_exp, rank);
}

vector<complexd> ROT(vector<complexd>& a, int n, int k,
                    int proc_exp, int rank, double phi) {
    vector<vector<complexd>> u(2);
	for (size_t i = 0; i < 2; ++i) {
		u[i].resize(2);
	}
    u[0][0] = 1;
    u[0][1] = u[1][0] = 0;
    u[1][1] = exp(complexd(0, 1) * phi);
    return QubitTransform(a, n, u, k, proc_exp, rank);
}





vector<complexd> C_not(vector<complexd>& a, int n, int q1, int q2,
                       long proc_exp, int rank) {
    vector<vector<complexd>> u(4);
	for (size_t i = 0; i < 4; ++i) {
		u[i].resize(4);
	}
    u[0][0] = u[1][1] = u[2][3] = u[3][2] = 1;
    u[0][1] = u[0][2] = u[0][3] = u[1][0] =
    u[1][2] = u[1][3] = u[2][0] = u[2][1] =
    u[2][2] = u[3][0] = u[3][1] = u[3][3] = 0;
    return TwoQubitsEvolution(a, n, q1, q2, u, proc_exp, rank);
}


vector<complexd> C_ROT(vector<complexd>& a, int n, int q1, int q2,
                      long proc_exp, int rank, double phi) {
    vector<vector<complexd>> u(4);
	for (size_t i = 0; i < 4; ++i) {
		u[i].resize(4);
	}
    u[0][0] = u[1][1] = u[2][2] = 1;
    u[0][1] = u[0][2] = u[0][3] =
    u[1][0] = u[1][2] = u[1][3] =
    u[2][0] = u[2][1] = u[2][3] =
    u[3][0] = u[3][1] = u[3][2] = 0;
    u[3][3] = exp(complexd(0, 1) * phi);
    return TwoQubitsEvolution(a, n, q1, q2, u, proc_exp, rank);
}
