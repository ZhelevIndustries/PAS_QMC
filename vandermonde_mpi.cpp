// vandermonde_mpi.cpp
// Compile: mpic++ -O3 -std=c++17 -o vandermonde_mpi vandermonde_mpi.cpp
// Run: mpirun -np 4 ./vandermonde_mpi alphas.txt

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 2){
        if(rank==0) std::cerr << "Usage: " << argv[0] << " alphas.txt\n"
                             << "File format: first line n, then n numbers (alpha_0 ... alpha_{n-1})\n";
        MPI_Finalize();
        return 1;
    }

    std::vector<long double> alpha;
    if(rank == 0){
        std::ifstream fin(argv[1]);
        if(!fin){
            std::cerr << "Error: cannot open file " << argv[1] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int n;
        fin >> n;
        if(n <= 0){
            std::cerr << "Invalid n in input\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        alpha.resize(n);
        for(int i=0;i<n;++i) fin >> alpha[i];
        if(!fin){
            std::cerr << "Error reading alphas\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // broadcast n
    int n = (int)alpha.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank != 0) alpha.resize(n);

    // broadcast alpha values
    MPI_Bcast(alpha.data(), n, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    // Each rank processes i = rank, rank+size, rank+2*size, ... up to n-2
    long double local_log_abs_sum = 0.0L; // sum of log(|differences|)
    // sign will be represented as parity of negative factors (0 -> positive, 1 -> negative)
    uint64_t local_neg_count = 0;

    for(int i = rank; i <= n-2; i += size){
        for(int j = i+1; j <= n-1; ++j){
            long double diff = alpha[j] - alpha[i];
            if(diff == 0.0L){
                // determinant is zero if any two alphas equal
                // reduce and exit early
                long double zero_log = -INFINITY; // log(0)
                uint64_t zero_neg = 0;
                MPI_Allreduce(MPI_IN_PLACE, &zero_log, 1, MPI_LONG_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                // use a special communicator reduction by flags: skip, just print zero on rank 0
                if(rank==0) std::cout << "Determinant = 0 (two alpha values are identical)\n";
                MPI_Finalize();
                return 0;
            }
            if(diff < 0) ++local_neg_count;
            local_log_abs_sum += logl(fabsl(diff));
        }
    }

    // global reductions
    long double global_log_abs_sum = 0.0L;
    MPI_Reduce(&local_log_abs_sum, &global_log_abs_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t global_neg_count = 0;
    MPI_Reduce(&local_neg_count, &global_neg_count, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        int sign = (global_neg_count % 2 == 0) ? 1 : -1;
        std::cout.setf(std::ios::scientific);
        std::cout.precision(12);
        std::cout << "n = " << n << "\n";
        std::cout << "Sign of determinant: " << sign << "\n";
        std::cout << "log(|determinant|) = " << global_log_abs_sum << " (natural log)\n";

        // attempt to compute determinant value if not overflowing/underflowing
        // check if exponent would be in a safe range for long double
        long double max_exp = logl(std::numeric_limits<long double>::max());
        long double min_exp = logl(std::numeric_limits<long double>::min()); // negative large magnitude
        if(global_log_abs_sum < max_exp && global_log_abs_sum > -max_exp){
            long double abs_det = expl(global_log_abs_sum);
            long double det = sign * abs_det;
            std::cout << "determinant = " << det << "\n";
        } else {
            std::cout << "Determinant magnitude is outside representable range; use log(|det|) above.\n";
        }
    }

    MPI_Finalize();
    return 0;
}
