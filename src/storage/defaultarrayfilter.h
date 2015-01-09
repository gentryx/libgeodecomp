#ifndef LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTARRAYFILTER_H

#include <libgeodecomp/storage/arrayfilter.h>

namespace LibGeoDecomp {

/**
 * The DefaultFilter just copies ofer the specified member -- sans
 * modification.
 */
template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY>
class DefaultArrayFilter : public ArrayFilter<CELL, MEMBER, EXTERNAL, ARITY>
{
public:
    friend class Serialization;

    void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target)
    {
        // fixme: needs test
        std::copy(first, last, target);
    }

    void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target)
    {
        // fixme: needs test
        std::copy(first, last, target);
    }

    void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, int num, MEMBER CELL:: *memberPointer)
    {
        copyMemberInImpl(
            source,
            target,
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, int num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (int i = 0; i < num; ++i) {
            std::copy(
                source + i * ARITY,
                source + (i + 1) * ARITY,
                (target[i].*memberPointer) + 0);
        }
    }

    void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, int num, MEMBER CELL:: *memberPointer)
    {
        copyMemberOutImpl(
            source,
            target,
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    virtual void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, int num, MEMBER (CELL:: *memberPointer)[ARITY])
    {
        for (int i = 0; i < num; ++i) {
            std::copy(
                (source[i].*memberPointer) + 0,
                (source[i].*memberPointer) + ARITY,
                target + i * ARITY);
        }
    }
};


}

#endif
