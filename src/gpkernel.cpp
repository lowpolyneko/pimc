/**
 * @file gpkernel.cpp
 * @author Adrian Del Maestro
 * @date 04.26.2026
 *
 * @brief GaussianProcessKernel implementation
 */

#include "gpkernel.h"
#include "factory.h"
#include "container.h"

/**************************************************************************//**
 * Setup the GaussianProcessKernel factory.
******************************************************************************/
GaussianProcessKernelFactory gaussianProcessKernelFactory;

#define REGISTER_GPKERNEL(NAME,TYPE) \
    const std::string TYPE::name = NAME;\
    bool reg ## TYPE = gaussianProcessKernelFactory()->Register<TYPE>(TYPE::name); 

/**************************************************************************//**
 * GP Kernel naming conventions:
 *
 * 1) be as descriptive as possible
 * 2) use only lower case letters
 * 3) spaces are fine, but then the option needs to be bracketed at the 
 *    command line.
******************************************************************************/
REGISTER_GPKERNEL("matern",MaternKernel)

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GAUSSIAN PROCESS KERNEL BASE CLASS ----------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/**************************************************************************//**
 *  Setup the path data members for the constant trial wavefunction.
******************************************************************************/
GaussianProcessKernelBase::GaussianProcessKernelBase (const Container * _boxPtr, 
        const double _ν, const dVec& ℓ) :
    boxPtr(_boxPtr)
{
    // empty constructor
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MATERN KERNEL CLASS -------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/**************************************************************************//**
 * Constructor.
******************************************************************************/
MaternKernel::MaternKernel(const Container* _boxPtr, const double _ν, const dVec &ℓ) : 
    GaussianProcessKernelBase(_boxPtr,_ν,ℓ),
    kernelFunction(selectKernel(_ν))
{
   invℓ = 1.0/ℓ;
}


/**************************************************************************//**
 * Choose the appropriate kernel based on the value of nu
******************************************************************************/
MaternKernel::KernelFunction MaternKernel::selectKernel(double _ν)
{
    if (std::abs(_ν - 0.5) < NU_TOL) {
        return &MaternKernel::matern1o2;
    }

    if (std::abs(_ν - 1.5) < NU_TOL) {
        return &MaternKernel::matern3o2;
    }

    if (std::abs(_ν - 2.5) < NU_TOL) {
        return &MaternKernel::matern5o2;
    }

    throw std::invalid_argument(
        "MaternKernel: only ν = 1/2, 3/2, and 5/2 are supported."
    );

}

/**************************************************************************//**
 * Implement the matern kernel for stifness ν.
******************************************************************************/
double MaternKernel::K(const dVec &r1, const dVec &r2) {

    /* Calculate scaled distance between points */
    dVec sep;
    sep = r2 - r1;
    boxPtr->putInBC(sep);
    sep = sep*invℓ;

    return kernelFunction(sqrt(dot(sep,sep)));
}
