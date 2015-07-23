#ifndef LIBGEODECOMP_STORAGE_SIMPLEARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_SIMPLEARRAYFILTER_H

#include <libgeodecomp/storage/arrayfilter.h>

namespace LibGeoDecomp {

/**
 * This class corresponds to SimpleFilter, but may be used for array
 * members (hence we inherit from ArrayFilter).
 */
template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
class SimpleArrayFilter : public ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>
{
public:
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(SimpleArrayFilter);

    friend class PolymorphicSerialization;
    friend class BoostSerialization;
    friend class HPXSerialization;

    virtual void load(const EXTERNAL source[ARITY], MEMBER   target[ARITY]) = 0;
    virtual void save(const MEMBER   source[ARITY], EXTERNAL target[ARITY]) = 0;

    virtual void copyStreakInImpl(const EXTERNAL *source, MEMBER *target, const std::size_t num, const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            MEMBER buffer[ARITY];
            load(&source[i * ARITY], buffer);

            for (std::size_t j = 0; j < ARITY; ++j) {
                target[i + j * stride] = buffer[j];
            }
        }
    }

    virtual void copyStreakOutImpl(const MEMBER *source, EXTERNAL *target, const std::size_t num, const std::size_t stride)
    {
        for (std::size_t i = 0; i < num; ++i) {
            MEMBER buffer[ARITY];
            for (std::size_t j = 0; j < ARITY; ++j) {
                buffer[j] = source[i + j * stride];
            }

            save(buffer, &target[i * ARITY]);
        }
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, const std::size_t num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (std::size_t i = 0; i < num; ++i) {
            load(&source[i * ARITY], target[i].*memberPointer);
        }
    }

    virtual void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, const std::size_t  num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (std::size_t i = 0; i < num; ++i) {
            save(source[i].*memberPointer, &target[i * ARITY]);
        }
    }
};

}

#endif
