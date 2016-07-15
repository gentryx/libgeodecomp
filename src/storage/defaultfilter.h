#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTER_H

#include <libgeodecomp/storage/filter.h>

namespace LibGeoDecomp {

/**
 * The DefaultFilter just copies over the specified member -- sans
 * modification.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL>
class DefaultFilter : public Filter<CELL, MEMBER, EXTERNAL>
{
public:
    void copyStreakInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        MEMBER *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        checkMemoryLocations(sourceLocation, targetLocation);

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
        checkMemoryLocations(sourceLocation, targetLocation);

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
        checkMemoryLocations(sourceLocation, targetLocation);

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
        checkMemoryLocations(sourceLocation, targetLocation);

        for (std::size_t i = 0; i < num; ++i) {
            target[i] = source[i].*memberPointer;
        }
    }

    void checkMemoryLocations(
        MemoryLocation::Location sourceLocation,
        MemoryLocation::Location targetLocation)
    {
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) ||
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            throw std::logic_error("DefaultFilter can't access CUDA device memory");
        }

        if ((sourceLocation != MemoryLocation::HOST) ||
            (targetLocation != MemoryLocation::HOST)) {
            throw std::invalid_argument("unknown combination of source and target memory locations");
        }

    }
};


}

#endif
