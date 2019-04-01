# Multi-threaded and SIMD computation of variance

The goal of the project is to write in C some code to compute the variance of a
vector using SIMD (AVX) and multithreading.

Simply run `make` on a computer supporting AVX to compare the speedup between a
simple scalar version of the computation of the variance and one using both SIMD
and multithreading.

Using AVX (256 bits) and 8 threads on a Macbook Pro with 4 cores and
hyperthreading, the computation of the variance of a random vector of length
1000000 is about **60 times faster**.

`N_THREADS` and `N` corresponds resp. to the number of thread used and the
length of the random vector on which the variance is computed.

