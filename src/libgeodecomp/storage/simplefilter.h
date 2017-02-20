#ifndef LIBGEODECOMP_STORAGE_SIMPLEFILTER_H
#define LIBGEODECOMP_STORAGE_SIMPLEFILTER_H

#include <libgeodecomp/storage/filter.h>

namespace LibGeoDecomp {

/**
 * Inheriting from this class instead of Filter will spare you
 * having to implement 4 functions (instead you'll have to write
 * just 2). It'll be a little slower though.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL>
class SimpleFilter : public Filter<CELL, MEMBER, EXTERNAL>
{
public:
    virtual void load(
        const EXTERNAL& source,
        MEMBER *target) = 0;

    virtual void save(
        const MEMBER& source,
        EXTERNAL *target) = 0;

    virtual void copyStreakInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        MEMBER *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            load(source[i], &target[i]);
        }
    }

    virtual void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            save(source[i], &target[i]);
        }
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        for (std::size_t i = 0; i < num; ++i) {
            load(source[i], &(target[i].*memberPointer));
        }
    }

    virtual void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        for (std::size_t i = 0; i < num; ++i) {
            save(source[i].*memberPointer, &target[i]);
        }
    }
};

}

#endif
