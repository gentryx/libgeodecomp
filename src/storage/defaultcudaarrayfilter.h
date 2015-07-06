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
        // fixme
    }

    void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        // fixme
    }

    void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        // fixme
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER (CELL:: *memberPointer)[ARITY])
    {
        // fixme
    }

};

}


#endif

#endif
