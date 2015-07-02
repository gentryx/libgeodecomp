#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/filter.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace LibGeoDecomp {

/**
 * The DefaultFilter just copies ofer the specified member -- sans
 * modification.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL>
class DefaultFilter : public Filter<CELL, MEMBER, EXTERNAL>
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
        const EXTERNAL *end = source + num;
        std::copy(source, end, target);
    }

    void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        const MEMBER *end = source + num;
        std::copy(source, end, target);
    }

    void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
// #if defined LIBGEODECOMP_WITH_CUDA  && defined __CUDACC__
//         cudaPointerAttributes attributes;
//         cudaPointerGetAttributes(&attributes, target);

//         // if (attributes.memoryType == cudaMemoryTypeDevice) {
//         //     std::cout << "would now copy to device...\n";
//         //     return;
//         // } else {
//         //     std::cout << "ok1\n";
//         // }
//         return;
// #endif

        for (std::size_t i = 0; i < num; ++i) {
            target[i].*memberPointer = source[i];
        }
    }

    void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::HOST)) {
            for (std::size_t i = 0; i < num; ++i) {
                target[i] = source[i].*memberPointer;
            }
            return;
        }

#ifdef LIBGEODECOMP_WITH_CUDA
        if ((sourceLocation == MemoryLocation::HOST) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            throw std::logic_error("not implemented yet (HOST->CUDA_DEVICE)");
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::HOST)) {
            throw std::logic_error("not implemented yet (CUDA_DEVICE->HOST)");
        }

        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            throw std::logic_error("not implemented yet (CUDA_DEVICE->CUDA_DEVICE)");
        }

        throw std::invalid_argument("unknown combination of source and target memory locations");
#else
        throw std::invalid_argument("LibGeoDecomp was configured without support for CUDA, can't access device memory.");
#endif


    }
};


}

#endif
