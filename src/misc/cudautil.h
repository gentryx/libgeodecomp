#ifndef _libgeodecomp_misc_cudautil_h_
#define _libgeodecomp_misc_cudautil_h_

#include <iostream>
#include <stdexcept>
#include <libgeodecomp/config.h>

#ifndef LIBGEODECOMP_FEATURE_CUDA

#ifndef __host__
#define __host__
#endif 

#ifndef __device__
#define __device__
#endif 

#endif

#ifdef LIBGEODECOMP_FEATURE_CUDA

#ifdef __CUDACC__

#include <cuda.h>

namespace LibGeoDecomp {

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
