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
    friend class Serialization;

    virtual void load(const EXTERNAL& source, MEMBER   *target) = 0;
    virtual void save(const MEMBER&   source, EXTERNAL *target) = 0;

    virtual void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target)
    {
        MEMBER *cursor = target;

        for (const EXTERNAL *i = first; i != last; ++i, ++cursor) {
            load(*i, cursor);
        }
    }

    virtual void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target)
    {
        EXTERNAL *cursor = target;

        for (const MEMBER *i = first; i != last; ++i, ++cursor) {
            save(*i, cursor);
        }
    }

    virtual void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, int num, MEMBER CELL:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            load(source[i], &(target[i].*memberPointer));
        }
    }

    virtual void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, int num, MEMBER CELL:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            save(source[i].*memberPointer, &target[i]);
        }
    }
};

}

#endif
