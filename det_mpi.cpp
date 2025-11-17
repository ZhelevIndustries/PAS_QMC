#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

extern "C" {
    void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int N = 3;
    double A[N*N] = {
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    };

    double A_col[N*N];
    if (rank == 0) {
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                A_col[c*N + r] = A[r*N + c];
    }

    MPI_Bcast(A_col, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // LU decomposition
    int ipiv[N];
    int m = N, n = N, lda = N, info;

    dgetrf_(&m, &n, A_col, &lda, ipiv, &info);

    double det = 1.0;
    if (info == 0) {
        for (int i = 0; i < N; i++)
            det *= A_col[i*N + i];
        
        int pivot_sign = 1;
        for (int i = 0; i < N; i++)
            if (ipiv[i] != i + 1)
                pivot_sign *= -1;

        det *= pivot_sign;
    }

    if (rank == 0) {
        std::cout << "Determinant = " << det << std::endl;
    }

    MPI_Finalize();
    return 0;
}
