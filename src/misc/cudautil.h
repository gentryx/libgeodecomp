#ifndef LIBGEODECOMP_MISC_CUDAUTIL_H
#define LIBGEODECOMP_MISC_CUDAUTIL_H

#include <libgeodecomp/config.h>

#include <iostream>
#include <stdexcept>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifdef LIBGEODECOMP_WITH_CUDA

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
