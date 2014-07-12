#ifndef LIBGEODECOMP_STORAGE_DEFAULTFILTER_H
#define LIBGEODECOMP_STORAGE_DEFAULTFILTER_H

#include <libgeodecomp/storage/filter.h>

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

    void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target)
    {
        std::copy(first, last, target);
    }

    void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target)
    {
        std::copy(first, last, target);
    }

    void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, int num, MEMBER CELL:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            target[i].*memberPointer = source[i];
        }
    }

    void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, int num, MEMBER CELL:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            target[i] = source[i].*memberPointer;
        }
    }
};


}

#endif
