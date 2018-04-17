#ifndef LIBGEODECOMP_MISC_CUDAUTIL_H
#define LIBGEODECOMP_MISC_CUDAUTIL_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA

#ifdef __ICC
// disabling this warning as implicit type conversion here as it's an intented feature for dim3
#pragma warning push
#pragma warning (disable: 2304)
#endif

#include <libflatarray/macros.hpp>

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_PRE
#include <iostream>
#include <stdexcept>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_POST

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

    template<typename COORD_TYPE>
    static void generateLaunchConfig(dim3 *cudaGridDim, dim3 *cudaBlockDim, const COORD_TYPE& dim)
    {
        int maxDimX = 512;
        int dimY = 1;

        if (dim.y() >= 4) {
            maxDimX = 128;
            dimY = 4;
        }

        int dimX = 32;
        for (; dimX < maxDimX; dimX <<= 1) {
            if (dimX >= dim.x()) {
                break;
            }
        }

        *cudaBlockDim = dim3(dimX, dimY, 1);

        cudaGridDim->x = divideAndRoundUp(dim.x(), cudaBlockDim->x);
        cudaGridDim->y = divideAndRoundUp(dim.y(), cudaBlockDim->y);
        cudaGridDim->z = divideAndRoundUp(dim.z(), cudaBlockDim->z);
    }

private:
    static int divideAndRoundUp(int i, int dividend)
    {
        int ret = i / dividend;
        if (i % dividend) {
            ret += 1;
        }

        return ret;
    }

};

}

#endif

#endif
