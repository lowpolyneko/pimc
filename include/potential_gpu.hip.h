#ifndef POTENTIAL_GPU_HIP_H
#define POTENTIAL_GPU_HIP_H

#include "common_gpu.h"
#include "potential_gpu_common.h"

__global__ void generic_gp_kernel(double* values, const double* positions,
        int count, const double* gpdata, int numPoints,
        GenericGPGpuParams params)
{
    __shared__ double partial[GPU_BLOCK_SIZE];
    constexpr double sqrt3 = 1.7320508075688772935;
    constexpr double sqrt5 = 2.2360679774997896964;

    const int i = blockIdx.x;
    const int lid = threadIdx.x;
    if (i >= count)
        return;

    double r[NDIM];
    for (int dim = 0; dim < NDIM; ++dim)
        r[dim] = (positions[NDIM * i + dim] - params.normOffset[dim]) /
            params.normScale[dim];

    double accum = 0.0;
    for (int k = lid; k < numPoints; k += blockDim.x) {
        const double* tx = gpdata + (NDIM + 1) * k;
        double sep2 = 0.0;
        for (int dim = 0; dim < NDIM; ++dim) {
            const double d = (tx[dim] - r[dim]) / params.ell[dim];
            sep2 += d * d;
        }
        const double sep = sqrt(sep2);
        double kval;
        if (fabs(params.maternNu - 0.5) < 1.0e-12) {
            kval = exp(-sep);
        } else if (fabs(params.maternNu - 1.5) < 1.0e-12) {
            const double x = sqrt3 * sep;
            kval = (1.0 + x) * exp(-x);
        } else {
            const double x = sqrt5 * sep;
            kval = (1.0 + x + x * x / 3.0) * exp(-x);
        }
        accum += kval * tx[NDIM];
    }

    partial[lid] = accum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            partial[lid] += partial[lid + stride];
        __syncthreads();
    }

    if (lid == 0)
        values[i] = params.dataStandardMean +
            params.dataStandardStd * (params.mean + params.sigma2 * partial[0]);
}

inline void generic_gp_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* gpdata, int numPoints,
        const dVec normOffset, const dVec normScale, const dVec ell,
        double dataStandardMean, double dataStandardStd, double mean,
        double sigma2, double maternNu)
{
    const int threads = GPU_BLOCK_SIZE;
    const int blocks = count;
    GenericGPGpuParams params{};
    for (int dim = 0; dim < NDIM; ++dim) {
        params.normOffset[dim] = normOffset[dim];
        params.normScale[dim] = normScale[dim];
        params.ell[dim] = ell[dim];
    }
    params.dataStandardMean = dataStandardMean;
    params.dataStandardStd = dataStandardStd;
    params.mean = mean;
    params.sigma2 = sigma2;
    params.maternNu = maternNu;
    hipLaunchKernelGGL(generic_gp_kernel, dim3(blocks), dim3(threads), 0,
            stream, values, positions, count, gpdata, numPoints,
            params);
}

#endif
