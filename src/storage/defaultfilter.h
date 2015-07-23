#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTER_H

#include <libgeodecomp/storage/filter.h>
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/runtime/serialization/serialize.hpp>
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
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(DefaultFilter)

    friend class PolymorphicSerialization;
    friend class BoostSerialization;
    friend class HPXSerialization;

    void copyStreakInImpl(
        const EXTERNAL *source, MEMBER *target, const std::size_t num, const std::size_t stride)
    {
        const EXTERNAL *end = source + num;
        std::copy(source, end, target);
    }

    void copyStreakOutImpl(
        const MEMBER *source, EXTERNAL *target, const std::size_t num, const std::size_t stride)
    {
        const MEMBER *end = source + num;
        std::copy(source, end, target);
    }

    void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, std::size_t num, MEMBER CELL:: *memberPointer)
    {
        for (std::size_t i = 0; i < num; ++i) {
            target[i].*memberPointer = source[i];
        }
    }

    void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, std::size_t num, MEMBER CELL:: *memberPointer)
    {
        for (std::size_t i = 0; i < num; ++i) {
            target[i] = source[i].*memberPointer;
        }
    }
};


}

#endif
