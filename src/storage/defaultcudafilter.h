#ifndef LIBGEODECOMP_STORAGE_DEFAULTCUDAFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTCUDAFILTER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/filter.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDACC__

namespace LibGeoDecomp {

namespace DefaultCUDAFilterHelpers {

/**
 * This workaround is necessary to pass a member pointer to a kernel.
 * Seems to be a bug in CUDA:
 *
 * http://stackoverflow.com/questions/23199824/c-cuda-pointer-to-member
 */
template<typename MEMBER_POINTER>
class MemberPointerWrapper
{
public:
    MEMBER_POINTER value;

    explicit inline
    MemberPointerWrapper(MEMBER_POINTER value) :
        value(value)
    {}
};

template<typename CELL, typename EXTERNAL, typename MEMBER>
__global__
void aggregateMember(
    const CELL *source,
    EXTERNAL *target,
    MemberPointerWrapper<MEMBER CELL::*> memberPointerWrapper,
    int num)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num) {
        target[index] = source[index].*memberPointerWrapper.value;
    }
}

template<typename CELL, typename EXTERNAL, typename MEMBER>
__global__
void distributeMember(
    const EXTERNAL *source,
    CELL *target,
    MemberPointerWrapper<MEMBER CELL::*> memberPointerWrapper,
    int num)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num) {
        target[index].*memberPointerWrapper.value = source[index];
    }
}


template<typename CELL, typename EXTERNAL, typename MEMBER>
void runAggregateMemberKernel(
    const CELL *source,
    EXTERNAL *target,
    MEMBER CELL:: *memberPointer,
    std::size_t num)
{
    dim3 gridDim(num / 32 + 1, 1, 1);
    dim3 blockDim(32, 1, 1);
    CUDAUtil::checkForError();

    std::size_t byteSize = num * sizeof(MEMBER);
    MEMBER *deviceBuffer;
    cudaMalloc(&deviceBuffer, byteSize);

    aggregateMember<CELL, EXTERNAL, MEMBER><<<gridDim, blockDim>>>(
        source,
        deviceBuffer,
        MemberPointerWrapper<MEMBER CELL:: *>(memberPointer),
        num);

    cudaMemcpy(target, deviceBuffer, byteSize, cudaMemcpyDeviceToHost);
    cudaFree(deviceBuffer);
    CUDAUtil::checkForError();
}

template<typename EXTERNAL, typename CELL, typename MEMBER>
void runDistributeMemberKernel(
    const EXTERNAL *source,
    CELL *target,
    MEMBER CELL:: *memberPointer,
    std::size_t num)
{
    dim3 gridDim(num / 32 + 1, 1, 1);
    dim3 blockDim(32, 1, 1);
    CUDAUtil::checkForError();

    std::size_t byteSize = num * sizeof(MEMBER);
    MEMBER *deviceBuffer;
    cudaMalloc(&deviceBuffer, byteSize);
    cudaMemcpy(deviceBuffer, source, byteSize, cudaMemcpyHostToDevice);

    distributeMember<CELL, EXTERNAL, MEMBER><<<gridDim, blockDim>>>(
        deviceBuffer,
        target,
        MemberPointerWrapper<MEMBER CELL:: *>(memberPointer),
        num);

    cudaFree(deviceBuffer);
    CUDAUtil::checkForError();
}

}

/**
 * CUDA-aware re-implementation of DefaultFilter
 */
template<typename CELL, typename MEMBER, typename EXTERNAL>
class DefaultCUDAFilter : public Filter<CELL, MEMBER, EXTERNAL>
{
public:
    friend class Serialization;

    void copyStreakInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        MEMBER *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        std::cout << "  OOOOPPPPSSSIIIIDOOOPPPSSSYYY1\n";
    }

    void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        std::cout << "  OOOOPPPPSSSIIIIDOOOPPPSSSYYY2\n";
    }

    void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        DefaultCUDAFilterHelpers::runDistributeMemberKernel(source, target, memberPointer, num);
    }

    void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        DefaultCUDAFilterHelpers::runAggregateMemberKernel(source, target, memberPointer, num);
    }
};


}

#endif

#endif
