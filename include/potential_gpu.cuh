#ifndef POTENTIAL_GPU_CUH
#define POTENTIAL_GPU_CUH

#include "common_gpu.h"

void gp_potential_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* trainx,
        const double* prod, int numPoints);

#endif
