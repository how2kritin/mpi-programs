#include "mpi.h"
#include <iostream>
#include <vector>

// References:
// https://rookiehpc.org/mpi/docs/index.html
// https://stackoverflow.com/questions/13578548/dynamic-allocation-2d-array-in-c-difference-with-a-simple-2d-array-mpi
// https://cse.buffalo.edu/faculty/miller/Courses/CSE633/thanigachalam-Spring-2014-CSE633.pdf

// Assumption: The given matrix is guaranteed to be Non-Singular.

// Idea: Was told to use the Row Reduction Method (Gaussian Inversion -> Identity Matrix Augmentation) in the document, for Matrix Inversion.
// 1. Preprocess the matrix. If the ith element in ith row is 0, find a row somewhere below it where this element is not 0, and swap with it.
// 2. Split up chunk of rows and spread them across the processors. Each processor gets a certain number of rows depending on the number of elements in the matrix. Note that some processors may have to process at most 1 row less than the others.
// 3. While making identity matrix as a whole, you just need to make one for the current row only. Put a 1 in the (i+1)th position (1-indexed) for row i (where i is the index of the row in the ORIGINAL matrix), and the rest are zeros.
// 4. Now, iterating over the rows, pivot row i, and get the process that owns this row to normalize it and broadcast just this 1 row to all the other processes so that they can subtract it from their own rows (except pivot row) after multiplying it with some factor.
// 5. Combine the rows of the initial identity matrix (augmented matrix) which is now A^-1 itself into root process, and return the answer.


// Time complexity:
// O(N^3 / p)

// Notes regarding MPI:
// 1. The group communication primitives require contiguous memory. So, I am using a 1D array instead of a 2D array.
// 2. Using Scatterv and Gatherv to send a variable amount of data (variable number of rows in this case).


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

    // determining how many rows to send to each process, and the offsets for the same.
    int rows_per_proc = n / num_p;
    int rem_rows = n % num_p;

    // needed for variable send and receive via scatter and gather. all processes need these.
    int my_rows = rows_per_proc + (id < rem_rows ? 1 : 0);
    int my_start_row = id * rows_per_proc + std::min(id, rem_rows);

    std::vector<int> sendcounts(num_p), displs(num_p, 0);
    for (int i = 0; i < num_p; i++) {
        sendcounts[i] = n * (rows_per_proc + (i < rem_rows ? 1 : 0));
        if (i > 0) displs[i] = displs[i-1] + sendcounts[i-1];
        // std::cout << i << " " << displs[i] << "\n";
    }

    std::vector<float> matrix(n * n);
    std::vector<int> row_swap_tracker(n), row_owner_tracker(n);

    if (id == 0) {
        for (int i = 0; i < n; i++) row_swap_tracker[i] = i;

        for (int i = 0; i < n * n; i++)
            fscanf(fPtr, "%f", &matrix[i]);

        for (int i = 0; i < n; i++) {
            if (matrix[i * n + i] == 0) {
                for (int j = i + 1; j < n; j++) {
                    if (matrix[i * n + j] != 0) {
                        for (int k = 0; k < n; k++)
                            std::swap(matrix[i * n + k], matrix[j * n + k]); // swapping the elements
                        std::swap(row_swap_tracker[i], row_swap_tracker[j]);
                        break;
                    }
                }
            }
        }

        // set up the row owner tracker.
        int current_row = 0; // Track the current row index

for (int p = 0; p < num_p; p++) {
    int rows_for_this_process = rows_per_proc + (p < rem_rows ? 1 : 0); // Calculate rows for each process

    for (int j = 0; j < rows_for_this_process; j++) {
        row_owner_tracker[current_row++] = p; // Assign rows to the current process
    }
}

    }

    // now, split up the rows among the processes.
    // MPI_Scatter and MPI_Scatterv are too slow!
//    MPI_Scatterv(matrix.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
//                 local_rows.data(), (int) local_rows.size(), MPI_FLOAT,
//                 0, MPI_COMM_WORLD);

    // using broadcast instead.
    //  printf("%d---------",id);
    MPI_Bcast(matrix.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_swap_tracker.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_owner_tracker.data(), n, MPI_INT, 0, MPI_COMM_WORLD);


    // ------------------------------------
//    double start_time = MPI_Wtime();

    // HANDLING MATRIX INVERSION HERE: Gauss-Jordan Inversion
    std::vector<float> local_rows(my_rows * n), local_aug_rows(my_rows * n);

    // copy the respective local row over.
    // also, making the augmented identity matrix. do this with respect to the swaps we made at the start.
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_rows[i * n + j] = matrix[(my_start_row + i) * n + j];
            local_aug_rows[i * n + j] = (row_swap_tracker[my_start_row + i] == j) ? 1.0f : 0.0f;
        }
    }

    // iterating over each row;
    // pivoting row i, and getting the process that owns this row to normalize it and broadcast just this 1 row to all the other processes so that they can subtract it from their own rows (except pivot row) after multiplying it with some factor
    for (int i = 0; i < n; i++) {
        int owner = row_owner_tracker[i];

        // if the current processor owns this row, normalize this row using the pivot, and then broadcast the normalized row to all the other processors.
        std::vector<float> pivot_row(n), pivot_aug_row(n);
        if (id == owner) {
            int local_row_num = i - my_start_row;
            float pivot = local_rows[local_row_num * n + i];
            for (int j = 0; j < n; j++) {
                local_rows[local_row_num * n + j] /= pivot;
                local_aug_rows[local_row_num * n + j] /= pivot;
            }
            pivot_row = std::vector<float>(local_rows.begin() + local_row_num * n,
                                           local_rows.begin() + (local_row_num + 1) * n);
            pivot_aug_row = std::vector<float>(local_aug_rows.begin() + local_row_num * n,
                                               local_aug_rows.begin() + (local_row_num + 1) * n);
        }
       
        MPI_Bcast(pivot_row.data(), n, MPI_FLOAT, owner, MPI_COMM_WORLD);
        MPI_Bcast(pivot_aug_row.data(), n, MPI_FLOAT, owner, MPI_COMM_WORLD);


        // by this point, due to the Bcast operation, all processes are sync'ed up.

        // now that you have the pivot row, get every process to multiply this row with the number that needs to be made 0, and subtract this row from all its rows except the pivot row itself.
        for (int k = 0; k < my_rows; k++) {
            int global_row_num = my_start_row + k;
            if (global_row_num != i) {
                float mult_factor = local_rows[k * n + i];
                for (int j = 0; j < n; j++) {
                    local_rows[k * n + j] -= mult_factor * pivot_row[j];
                    local_aug_rows[k * n + j] -= mult_factor * pivot_aug_row[j];
                }
            }
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

    // gather all the inverted rows
    // printf("%d--------------\n",id);
    std::vector<float> result(n * n);
    MPI_Gatherv(local_aug_rows.data(), my_rows * n, MPI_FLOAT,
                result.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (id == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%.2f ", result[n * i + j]);
            printf("\n");
        }
        fclose(fPtr);
    }
    // printf("cbdscs\n");
    // terminate the MPI exec env.
    MPI_Finalize();
    // printf("%dhere----\n",id);
    return 0;
}