#ifndef POTENTIAL_GPU_CUH
#define POTENTIAL_GPU_CUH

#include "common_gpu.h"

void generic_gp_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* gpdata, int numPoints,
        const dVec normOffset, const dVec normScale, const dVec ell,
        double dataStandardMean, double dataStandardStd, double mean,
        double sigma2, double maternNu);

#endif
