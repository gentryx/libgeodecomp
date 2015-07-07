#ifndef LIBGEODECOMP_STORAGE_DEFAULTCUDAARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTCUDAARRAYFILTER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/defaultarrayfilter.h>
#include <libgeodecomp/storage/arrayfilter.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __CUDACC__

namespace LibGeoDecomp {

namespace DefaultCUDAArrayFilterHelpers {

template<typename MEMBER, typename EXTERNAL, int ARITY>
__global__
void aggregateMemberArray(
    const MEMBER *source,
    EXTERNAL *target,
    const std::size_t num,
    const std::size_t stride)
{
    for (int i = threadIdx.x; i < ARITY; i += blockDim.x) {
        target[blockIdx.x * ARITY + i] = source[i * stride + blockIdx.x];
    }
}

template<typename MEMBER, typename EXTERNAL, int ARITY>
__global__
void distributeMemberArray(
    const EXTERNAL *source,
    MEMBER *target,
    const std::size_t num,
    const std::size_t stride)
{
    for (int i = threadIdx.x; i < num; i += blockDim.x) {
        target[blockIdx.x * num + i] = source[i * ARITY + blockIdx.x];
    }
}

template<typename EXTERNAL, typename MEMBER, int ARITY>
void runAggregateMemberArrayKernel(
    const MEMBER *source,
    EXTERNAL *target,
    const std::size_t num,
    const std::size_t stride)
{
    dim3 gridDim(num, 1, 1);
    dim3 blockDim(32, 1, 1);

    aggregateMemberArray<MEMBER, EXTERNAL, ARITY><<<gridDim, blockDim>>>(
        source,
        target,
        num,
        stride);

    CUDAUtil::checkForError();
}

template<typename EXTERNAL, typename MEMBER, int ARITY>
void runDistributeMemberArrayKernel(
    const EXTERNAL *source,
    MEMBER *target,
    const std::size_t num,
    const std::size_t stride)
{
    dim3 gridDim(ARITY, 1, 1);
    dim3 blockDim(32, 1, 1);

    distributeMemberArray<MEMBER, EXTERNAL, ARITY><<<gridDim, blockDim>>>(
        source,
        target,
        num,
        stride);

    CUDAUtil::checkForError();
}

}

/**
 * CUDA-aware re-implementation of DefaultArrayFilter. Originally this
 * functionality was to be merged into the DefaultFilter. However, the
 * DefaultFilter needs to be available in compilation units which are
 * not handled by NVCC, so conditionally calling kernels there is no
 * option (no, not even with weak symbols).
 */
template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
class DefaultCUDAArrayFilter : public ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>
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
            DefaultCUDAArrayFilterHelpers::runDistributeMemberArrayKernel<MEMBER, EXTERNAL, ARITY>(
                source,
                target,
                num,
                stride);
            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {

            std::size_t byteSize = num * ARITY * sizeof(EXTERNAL);
            MEMBER *deviceBuffer;
            cudaMalloc(&deviceBuffer, byteSize);
            cudaMemcpy(deviceBuffer, source, byteSize, cudaMemcpyHostToDevice);

            DefaultCUDAArrayFilterHelpers::runDistributeMemberArrayKernel<MEMBER, EXTERNAL, ARITY>(
                deviceBuffer,
                target,
                num,
                stride);

            cudaFree(deviceBuffer);

            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>().copyStreakInImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                stride);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAArrayFilter::copyStreakInImpl()");
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
            DefaultCUDAArrayFilterHelpers::runAggregateMemberArrayKernel<MEMBER, EXTERNAL, ARITY>(
                source,
                target,
                num,
                stride);
            return;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {

            std::size_t byteSize = num * ARITY * sizeof(MEMBER);
            MEMBER *deviceBuffer;
            cudaMalloc(&deviceBuffer, byteSize);

            DefaultCUDAArrayFilterHelpers::runAggregateMemberArrayKernel<MEMBER, EXTERNAL, ARITY>(
                source,
                deviceBuffer,
                num,
                stride);

            cudaMemcpy(target, deviceBuffer, byteSize, cudaMemcpyDeviceToHost);
            cudaFree(deviceBuffer);

            return;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>().copyStreakOutImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                stride);
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAArrayFilter::copyStreakOutImpl()");
    }

    void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER (CELL:: *memberPointer)[ARITY])
    {
        cudaMemcpyKind direction = cudaMemcpyDefault;

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {

            direction = cudaMemcpyDeviceToDevice;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {

            direction = cudaMemcpyDeviceToHost;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {

            direction = cudaMemcpyHostToDevice;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {

            DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>().copyMemberInImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                memberPointer);
            return;
        }

        // we use this as a special case to catch combinations of
        // locations which are not covered by the conditions above:
        if (direction != cudaMemcpyDefault) {
            for (std::size_t i = 0; i < num; ++i) {
                cudaMemcpy(
                    (target[i].*memberPointer),
                    source + i * ARITY,
                    ARITY * sizeof(MEMBER),
                    direction);
            }

            CUDAUtil::checkForError();
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAArrayFilter::copyMemberInImpl()");
    }

    virtual void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER (CELL:: *memberPointer)[ARITY])
    {
        cudaMemcpyKind direction = cudaMemcpyDefault;

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {

            direction = cudaMemcpyDeviceToDevice;
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            direction = cudaMemcpyDeviceToHost;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {

            direction = cudaMemcpyHostToDevice;
        }

        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            DefaultArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>().copyMemberOutImpl(
                source,
                sourceLocation,
                target,
                targetLocation,
                num,
                memberPointer);
            return;
        }

        // we use this as a special case to catch combinations of
        // locations which are not covered by the conditions above:
        if (direction != cudaMemcpyDefault) {
            for (std::size_t i = 0; i < num; ++i) {
                cudaMemcpy(
                    target + i * ARITY,
                    (source[i].*memberPointer),
                    ARITY * sizeof(EXTERNAL),
                    direction);
            }

            CUDAUtil::checkForError();
            return;
        }

        throw std::logic_error("Unsupported combination of sourceLocation and targetLocation"
                               " in DefaultCUDAArrayFilter::copyMemberOutImpl()");
    }

};

}


#endif

#endif
