#ifndef POTENTIAL_GPU_COMMON_H
#define POTENTIAL_GPU_COMMON_H

struct GenericGPGpuParams {
    double normOffset[NDIM];
    double normScale[NDIM];
    double ell[NDIM];
    double dataStandardMean;
    double dataStandardStd;
    double mean;
    double sigma2;
    double maternNu;
};

#endif
