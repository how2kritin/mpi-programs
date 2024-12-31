# Parallel Algorithms implemented using MPI

This repository contains a subset of problems from Assignment-3 of the Distributed Systems course, taken in Monsoon'24 semester at IIIT Hyderabad. These algorithms have been implemented in a distributed setting using the Message Passing Interface, in C++.

Note that comments containing reference material and implementation ideas, as well as comments to indicate what each block of code does, have been provided in the source code for each of these programs.
1. [Prefix Sum](#prefix-sum)
2. [Matrix Inversion](#matrix-inversion)
3. [Matrix Chain Multiplication](#matrix-chain-multiplication)

---

## Pre-requisites

```mpic++```
Kindly ensure that `mpich` and `build-essential` packages are installed, if on a Linux distribution.

---

## Instructions to run

Compile each program using `mpic++ <source_file>` to create a `a.out` executable. 
Now, you can run this executable directly and provide it an input of the correct input format, as specified under the respective algorithm's section in this README.  

Alternatively, uncomment the commented lines of code in the source files that correspond to timing the algorithm, recompile the program and run the corresponding `runner.sh` script to run the program on different number of processes, from 1 to 12, on the input provided in `inp_file.txt` (you can change these parameters in the `runner.sh` script). This script generates an output in JSON format for easy visualisation.
Use the `testcase_gen.py` script to generate testcases for each of the problems.  

---

## Prefix Sum
Please find the source code at [prefix_sum.cpp](Prefix%20Sum/prefix_sum.cpp).

### Input Format
* The first line contains `N`, i.e., the number of elements there are.
* The second line contains `N` space-separated floating point numbers.

### Output Format
* A single line containing `N` space-separated floating point numbers, where the `i`th element is the prefix sum of the first `i` elements.

For additional details regarding implementation and time complexity, please refer to [the report.](Prefix%20Sum/Report.pdf).

---

## Matrix Inversion
Please find the source code at [matrix_inversion.cpp](Matrix%20Inversion/matrix_inversion.cpp).

### Input Format
* The first line contains `N`, i.e., the size of the `N x N` square matrix.
* The next `N` lines each contain `N` floating point numbers, which are the elements of the `N x N` square matrix.

### Output Format
* `N` lines each containing `N` floating point numbers, which are the elements of the inverse of the provided `N x N` square matrix.

### Note
A checker script, `checker.py` has been provided (requires `numpy`) to compute the inverse of the matrix present in `inp_file.txt`. Use this before running `runner.sh` to create the ground truth result.

For additional details regarding implementation and time complexity, please refer to [the report.](Matrix%20Inversion/Report.pdf).

---

## Matrix Chain Multiplication
Please find the source code at [mcc.cpp](Matrix%20Chain%20Multiplication/mcc.cpp).

### Input Format
* The first line contains `N`, i.e., the number of matrices in the chain.
* The second line contains `N + 1` integers, which represent the dimensions of the matrices in the chain. The dimensions of the `i`th matrix is `i x (i + 1)`th elements.

### Output Format
* A single integer representing the minimum number of scalar multiplications required to multiply the entire matrix chain.

For additional details regarding implementation and time complexity, please refer to [the report.](Matrix%20Chain%20Multiplication/Report.pdf).

---