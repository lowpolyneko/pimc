#include "potential_gpu.cuh"
#include "potential_gpu_common.h"

__global__ void gp_potential_kernel(double* values, const double* positions,
        int count, const double* trainx, const double* prod, int numPoints)
{
    __shared__ double partial[GPU_BLOCK_SIZE];
    const int i = blockIdx.x;
    const int lid = threadIdx.x;
    if (i >= count)
        return;

    const GPGpuPoint point = gp_prepare_point_device(positions + NDIM * i);
    if (point.cutoff) {
        if (lid == 0)
            values[i] = 10000.0;
        return;
    }

    partial[lid] = gp_accumulate_device(point, trainx, prod, numPoints,
            lid, blockDim.x);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            partial[lid] += partial[lid + stride];
        __syncthreads();
    }

    if (lid == 0)
        values[i] = gp_finalize_device(point, partial[0]);
}

void gp_potential_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* trainx,
        const double* prod, int numPoints)
{
    const int threads = GPU_BLOCK_SIZE;
    const int blocks = count;
    gp_potential_kernel<<<blocks, threads, 0, stream>>>(values, positions,
            count, trainx, prod, numPoints);
}
