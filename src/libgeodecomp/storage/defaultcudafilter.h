#ifndef LIBGEODECOMP_STORAGE_DEFAULTCUDAFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTCUDAFILTER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/defaultfilter.h>
#include <libgeodecomp/storage/filter.h>

#ifdef __CUDACC__

#ifdef LIBGEODECOMP_WITH_CUDA

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif
#include <cuda_runtime.h>
#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#endif

namespace LibGeoDecomp {

namespace DefaultCUDAFilterHelpers {

/**
 * This workaround is necessary to pass a member pointer to a kernel.
 * Seems to be a bug in CUDA:
 *
 * http://stackoverflow.com/questions/23199824/c-cuda-pointer-to-member
 *
 * Update: problem is slightly different, but persists in CUDA 8.0.44.
 */
template<typename MEMBER, typename MEMBER_POINTER>
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
    MemberPointerWrapper<MEMBER, char CELL::*> memberPointerWrapper,
    const std::size_t num)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num) {
        target[index] = source[index].*reinterpret_cast<MEMBER CELL::*>(memberPointerWrapper.value);
    }
}

template<typename CELL, typename EXTERNAL, typename MEMBER>
__global__
void distributeMember(
    const EXTERNAL *source,
    CELL *target,
    MemberPointerWrapper<MEMBER, char CELL::*> memberPointerWrapper,
    const std::size_t num)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num) {
        target[index].*reinterpret_cast<MEMBER CELL::*>(memberPointerWrapper.value) = source[index];
    }
}


template<typename CELL, typename EXTERNAL, typename MEMBER>
void runAggregateMemberKernel(
    const CELL *source,
    EXTERNAL *target,
    MEMBER CELL:: *memberPointer,
    const std::size_t num)
{
    dim3 gridDim(num / 32 + 1, 1, 1);
    dim3 blockDim(32, 1, 1);

    aggregateMember<CELL, EXTERNAL, MEMBER><<<gridDim, blockDim>>>(
        source,
        target,
        MemberPointerWrapper<MEMBER, char CELL:: *>(reinterpret_cast<char CELL::*>(memberPointer)),
        num);

    CUDAUtil::checkForError();
}

template<typename EXTERNAL, typename CELL, typename MEMBER>
void runDistributeMemberKernel(
    const EXTERNAL *source,
    CELL *target,
    MEMBER CELL:: *memberPointer,
    const std::size_t num)
{
    dim3 gridDim(num / 32 + 1, 1, 1);
    dim3 blockDim(32, 1, 1);

    distributeMember<CELL, EXTERNAL, MEMBER><<<gridDim, blockDim>>>(
        source,
        target,
        MemberPointerWrapper<MEMBER, char CELL:: *>(reinterpret_cast<char CELL::*>(memberPointer)),
        num);

    CUDAUtil::checkForError();
}

}

/**
 * CUDA-aware re-implementation of DefaultFilter. Originally this
 * functionality was to be merged into the DefaultFilter. However, the
 * DefaultFilter needs to be available in compilation units which are
 * not handled by NVCC, so conditionally calling kernels there is no
 * option (no, not even with weak symbols).
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
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyDeviceToDevice);
            return;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyDeviceToHost);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyHostToDevice);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyStreakInImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                stride);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAFilter::copyStreakInImpl()");
    }

    void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyDeviceToDevice);
            return;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyDeviceToHost);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            cudaMemcpy(target, source, byteSize, cudaMemcpyHostToDevice);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyStreakInImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                stride);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAFilter::copyStreakOutImpl()");
    }

    void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            DefaultCUDAFilterHelpers::runDistributeMemberKernel(source, target, memberPointer, num);
            return;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            std::vector<EXTERNAL> hostBuffer(num);
            cudaMemcpy(&hostBuffer[0], source, num * sizeof(EXTERNAL), cudaMemcpyDeviceToHost);

            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyMemberInImpl(
                &hostBuffer[0],
                MemoryLocation::HOST,
                target,
                targetLocation,
                num,
                memberPointer);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            MEMBER *deviceBuffer;
            cudaMalloc(&deviceBuffer, byteSize);
            cudaMemcpy(deviceBuffer, source, byteSize, cudaMemcpyHostToDevice);

            DefaultCUDAFilterHelpers::runDistributeMemberKernel(deviceBuffer, target, memberPointer, num);
            cudaFree(deviceBuffer);
            return;

        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyMemberInImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                memberPointer);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAFilter::copyMemberInImpl()");
    }

    void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            DefaultCUDAFilterHelpers::runAggregateMemberKernel(source, target, memberPointer, num);
            return;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            std::size_t byteSize = num * sizeof(MEMBER);
            MEMBER *deviceBuffer;
            cudaMalloc(&deviceBuffer, byteSize);

            DefaultCUDAFilterHelpers::runAggregateMemberKernel(source, deviceBuffer, memberPointer, num);

            cudaMemcpy(target, deviceBuffer, byteSize, cudaMemcpyDeviceToHost);
            cudaFree(deviceBuffer);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            std::vector<EXTERNAL> hostBuffer(num);
            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyMemberOutImpl(
                source,
                sourceLocation,
                &hostBuffer[0],
                MemoryLocation::HOST,
                num,
                memberPointer);
            cudaMemcpy(target, &hostBuffer[0], num * sizeof(EXTERNAL), cudaMemcpyHostToDevice);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultFilter<CELL, MEMBER, EXTERNAL>().copyMemberOutImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                memberPointer);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAFilter::copyMemberOutImpl()");
    }
};


}

#endif

#endif
