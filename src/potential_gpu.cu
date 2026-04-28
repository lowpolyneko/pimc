#include "potential_gpu.cuh"
#include "potential_gpu_common.h"

__global__ void gp_potential_kernel(double* values, const double* positions,
        int count, const double* trainx, const double* prod, int numPoints)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        values[i] = gp_potential_eval_device(positions + NDIM * i,
                trainx, prod, numPoints);
}

void gp_potential_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* trainx,
        const double* prod, int numPoints)
{
    const int threads = GPU_BLOCK_SIZE;
    const int blocks = (count + threads - 1) / threads;
    gp_potential_kernel<<<blocks, threads, 0, stream>>>(values, positions,
            count, trainx, prod, numPoints);
}
