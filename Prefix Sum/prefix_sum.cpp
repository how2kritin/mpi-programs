#include "mpi.h"
#include <iostream>
#include <vector>

// References:
// https://rookiehpc.org/mpi/docs/index.html
// https://stackoverflow.com/questions/14456705/make-slaves-wait-for-mpi-bcast-from-master

// Idea:
// 1. If there are 'p' processes and 'N' elements, then I will split the main array into N/p chunks, and send each chunk to a process. Padding with 0s if N is not divisible by p.
// 2. Now, prefix sum of each chunk will be computed by a process in O(N/p), but of course, they won't be reflective of the correct (GLOBAL) prefix sum as a whole.
// 3. To fix this, what if I make each of them send the last element of their local prefix sum back to the root process, and ask the root process to do a O(p) run over this, and prefix sum this up?
// 4. Then, send each one of these elements to the local processes, and ask them to add this number to all their prefix sums in O(N/p).
// 5. Lastly, ask each process to send their prefix sum arrays back to the root process, and it will return the final array after gathering them.

// Time complexity:
// O(N / p + p)

// Notes regarding MPI:
// 1. The group communication primitives must be exposed to all processes, for them to synchronize properly. Thus, they do not require a barrier to sync as a result.
// 2. The above send elements in the order of process rank, to ALL processes (including the processing sending it, itself!)
// 3. The above write data directly into a variable; no need to use 'recv' of any sort for each.

// to be able to compute prefix sum of a subarray of an array in-place.
void inplace_prefix_sum(std::vector<float>::iterator startItr, std::vector<float>::iterator endItr){
    float prefix_sum = 0;
    auto itr = startItr;

    while(itr != endItr) {
        prefix_sum += *itr;
        *itr = prefix_sum;
        itr++;
    }
}

int main(int argc, char *argv[]) {
    // initialize the MPI exec env here.
    MPI_Init(&argc, &argv);

    int id, num_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    // some variables initialized.
    int num_elems, num_elems_per_process;
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
        fscanf(fPtr, "%d", &num_elems);
    }
    MPI_Bcast(&num_elems, 1, MPI_INT, 0, MPI_COMM_WORLD);
    num_elems_per_process = num_elems / num_p + (num_elems % num_p > 0); // basically ceil function.

    std::vector<float> main_arr, local_arr(num_elems_per_process, 0);
    if (id == 0) {
        int num_arr_elems = num_elems + (num_p - (num_elems % num_p)); // pad the array with 0s, to split exactly into n/p elems.
        main_arr.resize(num_arr_elems);
        for(int i = 0; i < num_elems; i++)
            fscanf(fPtr, "%f", &main_arr[i]);
    }
    MPI_Scatter(main_arr.data(), num_elems_per_process, MPI_FLOAT, local_arr.data(), num_elems_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // ------------------------------------
//    double start_time = MPI_Wtime();

    // compute prefix sum.
    inplace_prefix_sum(local_arr.begin(), local_arr.end());

    // send last element to root process.
    std::vector<float> last_nums(num_p + 1, 0);
    MPI_Gather(&local_arr[local_arr.size() - 1], 1, MPI_FLOAT, last_nums.data() + 1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); // keep the 0 at the front to send to process 1.

    float num_to_add;

    // compute prefix sum of last elements.
    if (id == 0)
        inplace_prefix_sum(last_nums.begin() + 1, last_nums.end() - 1); // no need to get the total sum by adding last element. pointless to add first elem since it is 0 anyway.

    // send the corresponding number to be added to each process. send only the first num_p elements (ignore the last one which computes total sum of all).
    MPI_Scatter(last_nums.data(), 1, MPI_FLOAT, &num_to_add, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // compute final prefix sums.
    for(int i = 0; i < num_elems_per_process; i++)
        local_arr[i] += num_to_add;

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
//        std::cout << "For n = " << num_elems << " on " << num_p << " processors, max elapsed time is " << max_time << "s.\n";
//        delete[] all_times;
//    }

    // send them back to parent.
    MPI_Gather(local_arr.data(), num_elems_per_process, MPI_FLOAT, main_arr.data(), num_elems_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // print the answers.
    if (id == 0)
        for(int i = 0; i < num_elems; i++)
            std::cout << main_arr[i] << " ";

    // terminate the MPI exec env.
    MPI_Finalize();
}