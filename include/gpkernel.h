/**
 * @file gpkernel.h
 * @author Adrian Del Maestro
 * @date 04.28.2026
 *
 * @brief gpkernel class definitions.
 */

#include "common.h"

#ifndef GPKERNEL_H 
#define GPKERNEL_H

class Container;

constexpr double SQRT3 = 1.7320508075688772935;
constexpr double SQRT5 = 2.2360679774997896964;
constexpr double NU_TOL = 1.0E-12; 

// ========================================================================  
// GaussianProcessKernelBase Class
// ========================================================================  
/** 
 * A base class that all Gausian process kernel classes will be derived from. 
 *
 */
class GaussianProcessKernelBase {

    public:
        GaussianProcessKernelBase (const Container*, double, const dVec&); 
        virtual ~GaussianProcessKernelBase() = default;

        /** The kernel function */
        virtual double K(const dVec&, const dVec&) {return 0.0;}

    protected:
        const Container *boxPtr;

};

// ========================================================================  
// MaternKernel Class
// ========================================================================  
/**
 * A simple matern kernel.
 * @see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
 */
class MaternKernel: public GaussianProcessKernelBase {
    public:
        MaternKernel(const Container *, const double, const dVec&);

        static const std::string name;
        std::string getName() const {return name;}

        double K(const dVec&, const dVec&);

    private:
        dVec invℓ;          // the inverse of the kernel length scale

        using KernelFunction = double (*)(double);
        KernelFunction kernelFunction;
        KernelFunction selectKernel(double);
    
        /* Three matern kernels for ν = 1/2, 3/2, and 5/2 */ 
        static double matern1o2(double r) {return std::exp(-r);};
        static double matern3o2(double r) {
            const double x = SQRT3 * r;
            return (1.0 + x) * std::exp(-x);
        }
        static double matern5o2(double r) {
            const double x = SQRT5 * r;
            return (1.0 + x + x * x / 3.0) * std::exp(-x);
        }

};

#endif
