#ifndef POTENTIAL_GPU_HIP_H
#define POTENTIAL_GPU_HIP_H

#include "common_gpu.h"
#include "potential_gpu_common.h"

__global__ void gp_potential_kernel(double* values, const double* positions,
        int count, const double* gpdata, int numPoints)
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

    partial[lid] = gp_accumulate_device(point, gpdata, numPoints,
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

inline void gp_potential_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* gpdata, int numPoints)
{
    const int threads = GPU_BLOCK_SIZE;
    const int blocks = count;
    hipLaunchKernelGGL(gp_potential_kernel, dim3(blocks), dim3(threads), 0,
            stream, values, positions, count, gpdata, numPoints);
}

#endif
