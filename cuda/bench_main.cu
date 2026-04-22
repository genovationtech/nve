/*
 * NVE kernel microbenchmark driver.
 * Calls nve_run_microbench from nve_kernels.cu.
 *
 * Usage: nve_bench [hidden] [intermediate] [num_heads] [num_kv_heads] [head_dim] [iters]
 */

#include <stdio.h>
#include <stdlib.h>

extern "C" void nve_run_microbench(int hidden, int intermediate,
                                    int num_heads, int num_kv_heads,
                                    int head_dim, int iters);

int main(int argc, char** argv) {
    int hidden       = (argc > 1) ? atoi(argv[1]) : 2048;
    int intermediate = (argc > 2) ? atoi(argv[2]) : 8192;
    int num_heads    = (argc > 3) ? atoi(argv[3]) : 32;
    int num_kv_heads = (argc > 4) ? atoi(argv[4]) : 8;
    int head_dim     = (argc > 5) ? atoi(argv[5]) : 64;
    int iters        = (argc > 6) ? atoi(argv[6]) : 2000;

    printf("NVE Kernel Microbenchmark\n");
    printf("  hidden=%d  intermediate=%d  heads=%d/%d  head_dim=%d  iters=%d\n",
           hidden, intermediate, num_heads, num_kv_heads, head_dim, iters);
    printf("---\n");
    nve_run_microbench(hidden, intermediate, num_heads, num_kv_heads, head_dim, iters);
    printf("---\nDone.\n");
    return 0;
}
