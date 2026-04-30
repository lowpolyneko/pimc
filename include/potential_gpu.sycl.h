#ifndef POTENTIAL_GPU_SYCL_H
#define POTENTIAL_GPU_SYCL_H

#include "common_gpu.h"

inline void generic_gp_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* gpdata, int numPoints,
        const dVec normOffset, const dVec normScale, const dVec ell,
        double dataStandardMean, double dataStandardStd, double mean,
        double sigma2, double maternNu)
{
    constexpr int localSize = GPU_BLOCK_SIZE;
    constexpr double sqrt3 = 1.7320508075688772935;
    constexpr double sqrt5 = 2.2360679774997896964;
    const int kernelKind = (sycl::fabs(maternNu - 0.5) < 1.0e-12) ? 0 :
        ((sycl::fabs(maternNu - 1.5) < 1.0e-12) ? 1 : 2);

    stream.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> partial(sycl::range<1>(localSize), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(count * localSize),
                    sycl::range<1>(localSize)),
                [=](sycl::nd_item<1> item) {
            const int i = static_cast<int>(item.get_group(0));
            const int lid = static_cast<int>(item.get_local_id(0));

            double r[NDIM];
            for (int dim = 0; dim < NDIM; ++dim)
                r[dim] = (positions[NDIM * i + dim] - normOffset[dim]) / normScale[dim];

            double accum = 0.0;
            for (int k = lid; k < numPoints; k += localSize) {
                const double* tx = gpdata + (NDIM + 1) * k;
                double sep2 = 0.0;
                for (int dim = 0; dim < NDIM; ++dim) {
                    const double d = (tx[dim] - r[dim]) / ell[dim];
                    sep2 += d * d;
                }
                const double sep = sycl::sqrt(sep2);
                double kval;
                if (kernelKind == 0) {
                    kval = sycl::exp(-sep);
                } else if (kernelKind == 1) {
                    const double x = sqrt3 * sep;
                    kval = (1.0 + x) * sycl::exp(-x);
                } else {
                    const double x = sqrt5 * sep;
                    kval = (1.0 + x + x * x / 3.0) * sycl::exp(-x);
                }
                accum += kval * tx[NDIM];
            }

            partial[lid] = accum;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride)
                    partial[lid] += partial[lid + stride];
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0)
                values[i] = dataStandardMean +
                    dataStandardStd * (mean + sigma2 * partial[0]);
        });
    });
}

#endif
