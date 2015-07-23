#ifndef LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H

#include <libgeodecomp/storage/arrayfilter.h>
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/runtime/serialization/serialize.hpp>
#endif


namespace LibGeoDecomp {

/**
 * The DefaultArrayFilter is the counter piece to the DefaultFilter,
 * just for member arrays.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
class DefaultArrayFilter : public ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>
{
public:
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(DefaultArrayFilter)

    friend class PolymorphicSerialization;
    friend class BoostSerialization;
    friend class HPXSerialization;

    void copyStreakInImpl(
        const EXTERNAL *source, MEMBER *target, const std::size_t num, const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            for (std::size_t j = 0; j < ARITY; ++j) {
                target[j * stride + i] = source[i * ARITY + j];
            }
        }
    }

    void copyStreakOutImpl(
        const MEMBER *source, EXTERNAL *target, const std::size_t num, const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            for (std::size_t j = 0; j < ARITY; ++j) {
                target[i * ARITY + j] = source[j * stride + i];
            }
        }
    }

    void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, std::size_t num, MEMBER CELL:: *memberPointer)
    {
        copyMemberInImpl(
            source,
            target,
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, std::size_t num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (std::size_t i = 0; i < num; ++i) {
            std::copy(
                source + i * ARITY,
                source + (i + 1) * ARITY,
                (target[i].*memberPointer) + 0);
        }
    }

    void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, std::size_t num, MEMBER CELL:: *memberPointer)
    {
        copyMemberOutImpl(
            source,
            target,
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    virtual void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, std::size_t num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (std::size_t i = 0; i < num; ++i) {
            std::copy(
                (source[i].*memberPointer) + 0,
                (source[i].*memberPointer) + ARITY,
                target + i * ARITY);
        }
    }
};


}

#endif
