#ifndef LIBGEODECOMP_MISC_CUDAUTIL_H
#define LIBGEODECOMP_MISC_CUDAUTIL_H

#include <libgeodecomp/config.h>

#include <iostream>
#include <stdexcept>

#ifdef LIBGEODECOMP_WITH_CUDA

#ifdef __ICC
// disabling this warning as implicit type conversion here as it's an intented feature for dim3
#pragma warning push
#pragma warning (disable: 2304)
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef __ICC
#pragma warning pop
#endif

#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifdef LIBGEODECOMP_WITH_CUDA
#ifdef __CUDACC__

namespace LibGeoDecomp {

/**
 * A loose collection of helper functions for error handling, data
 * transfer etc. required for CUDA-capable GPUs.
 */
class CUDAUtil
{
public:
    static void checkForError()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
            throw std::runtime_error("CUDA error");
        }
    }
};

}

#endif
#endif

#endif
