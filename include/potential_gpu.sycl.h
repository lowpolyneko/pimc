#ifndef POTENTIAL_GPU_SYCL_H
#define POTENTIAL_GPU_SYCL_H

#include "common_gpu.h"

inline void gp_potential_gpu_launcher(gpu_stream_t stream, double* values,
        const double* positions, int count, const double* trainx,
        const double* prod, int numPoints)
{
    constexpr int localSize = GPU_BLOCK_SIZE;
    constexpr double xoffset2 = -0.09459459;
    constexpr double xscale0 = 7.0;
    constexpr double xscale1 = 3.5;
    constexpr double xscale2 = 7.09459459;
    constexpr double inv_ell10 = 1.0 / 0.78638807;
    constexpr double inv_ell11 = 1.0 / 2.17270815;
    constexpr double inv_ell12 = 1.0 / 0.77220716;
    constexpr double inv_ell20 = 1.0 / 2.03248109;
    constexpr double inv_ell21 = 1.0 / 5.05874827;
    constexpr double inv_ell22 = 1.0 / 1.71032378;
    constexpr double power = 0.63911297;
    constexpr double oscale = 814.69663559;
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
    constexpr double sqrt5 = 2.2360679774997896964;
    constexpr double pi = 3.14159265358979323846;

    stream.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> partial(sycl::range<1>(localSize), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(count * localSize),
                    sycl::range<1>(localSize)),
                [=](sycl::nd_item<1> item) {
            const int i = static_cast<int>(item.get_group(0));
            const int lid = static_cast<int>(item.get_local_id(0));

            double x = positions[NDIM * i + 0];
            double y = positions[NDIM * i + 1];
            double z = positions[NDIM * i + 2];
            if (z <= 0.0)
                z = -z;

            double angle = sycl::round((sycl::atan2(y, x) * 180.0 / pi) * 10000.0) / 10000.0;
            if (angle < 0.0)
                angle = 180.0 + (180.0 + angle);

            double finx = 0.0;
            double finy = 0.0;
            if (angle <= 30.0) {
                finx = x;
                finy = y;
            } else {
                double rotateangle = sycl::fmod(angle, 60.0) - angle;
                const double folded = angle + rotateangle;
                rotateangle = rotateangle * pi / 180.0;
                if (folded > 30.0) {
                    const double rotatedanglex = sycl::cos(rotateangle) * x - sycl::sin(rotateangle) * y;
                    const double rotatedangley = sycl::sin(rotateangle) * x + sycl::cos(rotateangle) * y;
                    finx = sycl::cos(pi / 3.0) * rotatedanglex + sycl::sin(pi / 3.0) * rotatedangley;
                    finy = sycl::sin(pi / 3.0) * rotatedanglex - sycl::cos(pi / 3.0) * rotatedangley;
                } else {
                    finx = sycl::cos(rotateangle) * x - sycl::sin(rotateangle) * y;
                    finy = sycl::sin(rotateangle) * x + sycl::cos(rotateangle) * y;
                }
            }

            const double rho = sycl::sqrt(finx * finx + finy * finy);
            if (rho < 3.0 && z < 2.0) {
                if (lid == 0)
                    values[i] = 10000.0;
                return;
            }

            const double rp0 = finx / xscale0;
            const double rp1 = finy / xscale1;
            const double rp2 = (z - xoffset2) / xscale2;
            constexpr int z2 = 1;

            double accum = 0.0;
            for (int k = lid; k < numPoints; k += localSize) {
                const double* tx = trainx + 4 * k;
                const int z1 = static_cast<int>(tx[3]);

                const double d10 = (tx[0] - rp0) * inv_ell10;
                const double d11 = (tx[1] - rp1) * inv_ell11;
                const double d12 = (tx[2] - rp2) * inv_ell12;
                const double sep1 = d10 * d10 + d11 * d11 + d12 * d12;
                const double sep1r = sycl::sqrt(sep1);
                const double k1v = (1.0 + sqrt5 * sep1r + 5.0 * (sep1 / 3.0)) * sycl::exp(-sqrt5 * sep1r);

                const double d20 = (tx[0] - rp0) * inv_ell20;
                const double d21 = (tx[1] - rp1) * inv_ell21;
                const double d22 = (tx[2] - rp2) * inv_ell22;
                const double sep2 = d20 * d20 + d21 * d21 + d22 * d22;
                const double sep2r = sycl::sqrt(sep2);
                const double k2v = (1.0 + sqrt5 * sep2r + 5.0 * (sep2 / 3.0)) * sycl::exp(-sqrt5 * sep2r);

                const double bias = (1.0 - z1) * (1.0 - z2) * sycl::pow(1.0 + z1 * z2, power);
                accum += oscale * (k1v + bias * k2v) * prod[k];
            }

            partial[lid] = accum;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = localSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride)
                    partial[lid] += partial[lid + stride];
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
                const double val_gp = meany + stdy * (mean + partial[0]);
                const double rnorm = sycl::sqrt(finx * finx + finy * finy + z * z);
                const double h_long = 1.0 / (1.0 + sycl::exp(-gama * (rnorm - r0)));
                double value = val_gp;
                if (rho > 5.0 || z > 6.5) {
                    const double ph0 = sycl::atan2(finy, finx);
                    const double th0 = sycl::acos(z / rnorm);
                    const double tt = sycl::cos(th0);
                    const double normm1 = sycl::sqrt(5.0) / 5.0;
                    const double normm2 = sycl::sqrt(13.0) / 13.0;
                    const double normm3 = 0.1792151994e-4;
                    const double p1 = -x1 / sycl::pow(rnorm, 6.0) - x2 / sycl::pow(rnorm, 8.0) - x3 / sycl::pow(rnorm, 10.0) - x4 / sycl::pow(rnorm, 12.0);
                    const double p2 = -normm1 * (1.5 * tt * tt - 0.5) *
                        (x5 / sycl::pow(rnorm, 6.0) + x6 / sycl::pow(rnorm, 8.0) + x7 / sycl::pow(rnorm, 10.0) + x8 / sycl::pow(rnorm, 12.0));
                    const double p3 = -normm2 * (3.0 / 8.0 + 35.0 / 8.0 * sycl::pow(tt, 4.0) - 15.0 / 4.0 * tt * tt);
                    const double p4 = x9 / sycl::pow(rnorm, 8.0) + x10 / sycl::pow(rnorm, 10.0) + x11 / sycl::pow(rnorm, 12.0);
                    const double p5 = -normm3 * 10395.0 * sycl::pow(1.0 - tt * tt, 3.0) * sycl::cos(6.0 * ph0) * x12 / sycl::pow(rnorm, 10.0);
                    const double vdw = p1 + p2 + p3 * p4 + p5;
                    value = (1.0 - h_long) * val_gp + h_long * vdw;
                }
                values[i] = value / 0.695;
            }
        });
    });
}

#endif
