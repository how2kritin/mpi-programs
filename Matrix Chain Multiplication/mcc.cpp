#include "mpi.h"
#include <iostream>
#include <vector>
#include <limits.h>

// References:
// https://rookiehpc.org/mpi/docs/index.html
// https://www.cs.purdue.edu/homes/ayg/book/Slides/chap12_slides.pdf

// Problem: We can't possibly parallelize by splitting the rows up into chunks - since each row depends on the rows below it.

// Idea:
// 1. So, the trick is to split it up into diagonal parallelization.
// 2. Basically, start from the principal diagonal (n elements) and split the work up amongst processors.
// 3. I will be splitting it up into different block chunks to allocate amongst processes.
// 3. Keep solving the other diagonals.

// Time complexity:
// O(N^3 / p)

int main(int argc, char *argv[]) {
    // initialize the MPI exec env here.
    MPI_Init(&argc, &argv);

    int id, num_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    // some variables initialized.
    int n;
    FILE *fPtr;

    // let only the 0th processor take input, and broadcast it to each array.
    if (id == 0) {
        // file containing input has its path stored in argv[1].
        if (argc < 2) {
            fprintf(stderr, "No input file found! Please pass its path as a command line argument.");
            MPI_Finalize();
            return 1;
        }

        fPtr = fopen(argv[1], "r");
        fscanf(fPtr, "%d", &n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> chain_dims(n + 1); // these are the dimensions of the matrix chains
    std::vector<int> DP((n + 1) * (n + 1), INT_MAX); // this is the DP matrix.
    for (int i = 0; i <= n; i++) DP[i * (n + 1) + i] = 0; // set diagonal elements cost to 0.

    if (id == 0)
        for (int i = 0; i < n + 1; i++)
            fscanf(fPtr, "%d", &chain_dims[i]);

    MPI_Bcast(chain_dims.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ------------------------------------
//    double start_time = MPI_Wtime();

    // len is length of matrix chain. Ex: (2, 3) represents chain length of 2. This is what determines which diagonal we are currently computing.
    for (int len = 2; len <= n; len++) {
        // on each diagonal, split work up in blocks amongst the processors.
        int num_tasks = n - len + 1;
        int block_size = num_tasks / num_p + (num_tasks % num_p > 0); // basically ceil function.

        int start_i = id * block_size + 1;
        int end_i = std::min((id + 1) * block_size, num_tasks);

        // now, make the processor calculate for its respective chunk.
        for (int i = start_i; i <= end_i; i++) {
            int j = i + len - 1;
            for (int k = i; k <= j - 1; k++) {
                int chain_cost = DP[i * (n + 1) + k] + DP[(k + 1) * (n + 1) + j] + chain_dims[i - 1] * chain_dims[k] * chain_dims[j];
                if (chain_cost < DP[i * (n + 1) + j]) DP[i * (n + 1) + j] = chain_cost;
            }
        }

        // common broadcasting that's visible to all processors so that they can get the values and synchronise.
        // if we put this in the calc loop above, it wouldn't be visible to all processors. it must be called by all processors.
        for (int i = 1; i <= num_tasks; i++) {
            int j = i + len - 1;
            MPI_Bcast(&DP[i * (n + 1) + j], 1, MPI_INT, (i - 1) / block_size, MPI_COMM_WORLD);
        }
    }

//    double time_elapsed = MPI_Wtime() - start_time;

    // ------------------------------------
//    double *all_times = nullptr;
//    double max_time = 0;
//    if (id == 0) all_times = new double[num_p];
//    MPI_Gather(&time_elapsed, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    // printing the max time taken by any single processor here.
//    if (id == 0) {
//        for(int i = 0; i < num_p; i++) max_time = std::max(all_times[i], max_time);
//        std::cout << "For n = " << n << " on " << num_p << " processors, max elapsed time is " << max_time << "s.\n";
//        delete[] all_times;
//    }

    // print the answer
    if (id == 0)
        printf("%d", DP[1 * (n + 1) + n]);

    // terminate the MPI exec env.
    MPI_Finalize();
}