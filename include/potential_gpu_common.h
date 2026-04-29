#ifndef POTENTIAL_GPU_COMMON_H
#define POTENTIAL_GPU_COMMON_H

#include <math.h>

struct GPGpuPoint {
    double finx;
    double finy;
    double z;
    double rho;
    double rp0;
    double rp1;
    double rp2;
    bool cutoff;
};

__device__ inline GPGpuPoint gp_prepare_point_device(const double* position)
{
    constexpr double xoffset2 = -0.09459459;
    constexpr double xscale0 = 7.0;
    constexpr double xscale1 = 3.5;
    constexpr double xscale2 = 7.09459459;
    constexpr double pi = 3.14159265358979323846;

    double x = position[0];
    double y = position[1];
    double z = position[2];
    if (z <= 0.0)
        z = -z;

    double angle = round((atan2(y, x) * 180.0 / pi) * 10000.0) / 10000.0;
    if (angle < 0.0)
        angle = 180.0 + (180.0 + angle);

    double finx = 0.0;
    double finy = 0.0;
    if (angle <= 30.0) {
        finx = x;
        finy = y;
    } else {
        double rotateangle = fmod(angle, 60.0) - angle;
        const double folded = angle + rotateangle;
        rotateangle = rotateangle * pi / 180.0;
        if (folded > 30.0) {
            const double rotatedanglex = cos(rotateangle) * x - sin(rotateangle) * y;
            const double rotatedangley = sin(rotateangle) * x + cos(rotateangle) * y;
            finx = cos(pi / 3.0) * rotatedanglex + sin(pi / 3.0) * rotatedangley;
            finy = sin(pi / 3.0) * rotatedanglex - cos(pi / 3.0) * rotatedangley;
        } else {
            finx = cos(rotateangle) * x - sin(rotateangle) * y;
            finy = sin(rotateangle) * x + cos(rotateangle) * y;
        }
    }

    const double rho = sqrt(finx * finx + finy * finy);
    GPGpuPoint point{};
    point.finx = finx;
    point.finy = finy;
    point.z = z;
    point.rho = rho;
    point.rp0 = finx / xscale0;
    point.rp1 = finy / xscale1;
    point.rp2 = (z - xoffset2) / xscale2;
    point.cutoff = (rho < 3.0 && z < 2.0);
    return point;
}

__device__ inline double gp_accumulate_device(const GPGpuPoint& point,
        const double* gpdata, int numPoints, int offset,
        int stride)
{
    constexpr double inv_ell10 = 1.0 / 0.78638807;
    constexpr double inv_ell11 = 1.0 / 2.17270815;
    constexpr double inv_ell12 = 1.0 / 0.77220716;
    constexpr double sqrt5 = 2.2360679774997896964;

    double accum = 0.0;
    for (int k = offset; k < numPoints; k += stride) {
        const double* tx = gpdata + 4 * k;
        const double d10 = (tx[0] - point.rp0) * inv_ell10;
        const double d11 = (tx[1] - point.rp1) * inv_ell11;
        const double d12 = (tx[2] - point.rp2) * inv_ell12;
        const double sep1 = d10 * d10 + d11 * d11 + d12 * d12;
        const double sep1r = sqrt(sep1);
        const double k1v = (1.0 + sqrt5 * sep1r + 5.0 * (sep1 / 3.0)) * exp(-sqrt5 * sep1r);
        accum += k1v * tx[3];
    }
    return accum;
}

__device__ inline double gp_finalize_device(const GPGpuPoint& point,
        double accum)
{
    constexpr double mean = 22.0553313;
    constexpr double meany = 8.64215696;
    constexpr double stdy = 104.91133484;
    constexpr double r0 = 5.765366674e+00;
    constexpr double gama = 1.671477473e+01;
    constexpr double x1 = 2.290626821e+05;
    constexpr double x2 = 4.069217828e+06;
    constexpr double x3 = 9.542607077e+07;
    constexpr double x4 = 0.0;
    constexpr double x5 = -4.775615288e+04;
    constexpr double x6 = -1.263899296e+07;
    constexpr double x7 = -2.959330054e+08;
    constexpr double x8 = 0.0;
    constexpr double x9 = 9.511488779e+05;
    constexpr double x10 = 2.551023972e+08;
    constexpr double x11 = 0.0;
    constexpr double x12 = 1.883887454e+08;

    const double val_gp = meany + stdy * (mean + accum);
    const double rnorm = sqrt(point.finx * point.finx + point.finy * point.finy + point.z * point.z);
    const double h_long = 1.0 / (1.0 + exp(-gama * (rnorm - r0)));
    double value = val_gp;
    if (point.rho > 5.0 || point.z > 6.5) {
        const double ph0 = atan2(point.finy, point.finx);
        const double tt = point.z / rnorm;
        const double tt2 = tt * tt;
        const double tt4 = tt2 * tt2;
        const double r2 = rnorm * rnorm;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r4 * r4;
        const double r10 = r8 * r2;
        const double r12 = r8 * r4;
        const double sin2 = 1.0 - tt2;
        const double sin6 = sin2 * sin2 * sin2;
        const double normm1 = sqrt(5.0) / 5.0;
        const double normm2 = sqrt(13.0) / 13.0;
        const double normm3 = 0.1792151994e-4;
        const double p1 = -x1 / r6 - x2 / r8 - x3 / r10 - x4 / r12;
        const double p2 = -normm1 * (1.5 * tt2 - 0.5) *
            (x5 / r6 + x6 / r8 + x7 / r10 + x8 / r12);
        const double p3 = -normm2 * (3.0 / 8.0 + 35.0 / 8.0 * tt4 - 15.0 / 4.0 * tt2);
        const double p4 = x9 / r8 + x10 / r10 + x11 / r12;
        const double p5 = -normm3 * 10395.0 * sin6 * cos(6.0 * ph0) * x12 / r10;
        value = (1.0 - h_long) * val_gp + h_long * (p1 + p2 + p3 * p4 + p5);
    }
    return value / 0.695;
}

__device__ inline double gp_potential_eval_device(const double* position,
        const double* gpdata, int numPoints)
{
    const GPGpuPoint point = gp_prepare_point_device(position);
    if (point.cutoff)
        return 10000.0;
    return gp_finalize_device(point,
            gp_accumulate_device(point, gpdata, numPoints, 0, 1));
}

#endif
