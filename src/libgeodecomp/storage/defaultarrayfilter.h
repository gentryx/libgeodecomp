#ifndef LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H

#include <libgeodecomp/storage/arrayfilter.h>

namespace LibGeoDecomp {

/**
 * The DefaultArrayFilter is the counter piece to the DefaultFilter,
 * just for member arrays.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
class DefaultArrayFilter : public ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>
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

        for (std::size_t i = 0; i < num; ++i) {
            for (std::size_t j = 0; j < ARITY; ++j) {
                target[j * stride + i] = source[i * ARITY + j];
            }
        }
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

        for (std::size_t i = 0; i < num; ++i) {
            for (std::size_t j = 0; j < ARITY; ++j) {
                target[i * ARITY + j] = source[j * stride + i];
            }
        }
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER (CELL:: *memberPointer)[ARITY])
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        for (std::size_t i = 0; i < num; ++i) {
            std::copy(
                source + i * ARITY,
                source + (i + 1) * ARITY,
                (target[i].*memberPointer) + 0);
        }
    }


    virtual void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER (CELL:: *memberPointer)[ARITY])
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        for (std::size_t i = 0; i < num; ++i) {
            std::copy(
                (source[i].*memberPointer) + 0,
                (source[i].*memberPointer) + ARITY,
                target + i * ARITY);
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
