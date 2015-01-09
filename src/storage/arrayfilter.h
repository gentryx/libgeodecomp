#ifndef LIBGEODECOMP_STORAGE_ARRAYFILTER_H
#define LIBGEODECOMP_STORAGE_ARRAYFILTER_H

#include <libgeodecomp/storage/filter.h>

namespace LibGeoDecomp {

template<typename CELL, typename MEMBER, typename EXTERNAL, int ARITY = 1>
class ArrayFilter : public FilterBase<CELL>
{
public:
    friend class Serialization;

    std::size_t sizeOf() const
    {
        return ARITY * sizeof(EXTERNAL);
    }

#ifdef LIBGEODECOMP_WITH_SILO
    int siloTypeID() const
    {
        return FilterHelpers::GetSiloTypeID<EXTERNAL>()();
    }
#endif

#ifdef LIBGEODECOMP_WITH_MPI
    virtual MPI_Datatype mpiDatatype() const
    {
        return FilterHelpers::GetMPIDatatype<EXTERNAL>()();
    }
#endif

    virtual std::string typeName() const
    {
        return FilterHelpers::GetTypeName<EXTERNAL>()();
    }

    virtual int arity() const
    {
        return ARITY;
    }

    /**
     * Copy a streak of variables to an AoS layout.
     */
    virtual void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target) = 0;

    /**
     * Extract a steak of members from an AoS layout.
     */
    virtual void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target) = 0;

    /**
     * Copy a streak of variables to the members of a streak of cells.
     */
    virtual void copyMemberInImpl(
        const EXTERNAL *source, CELL *target, int num, MEMBER (CELL:: *memberPointer)[ARITY]) = 0;

    /**
     * Extract a streak of members from a streak of cells.
     */
    virtual void copyMemberOutImpl(
        const CELL *source, EXTERNAL *target, int num, MEMBER (CELL:: *memberPointer)[ARITY]) = 0;

    /**
     * Do not override this function! It is final.
     */
    void copyStreakIn(const char *first, const char *last, char *target)
    {
        copyStreakInImpl(
            reinterpret_cast<const EXTERNAL*>(first),
            reinterpret_cast<const EXTERNAL*>(last),
            reinterpret_cast<MEMBER*>(target));
    }

    /**
     * Do not override this function! It is final.
     */
    void copyStreakOut(const char *first, const char *last, char *target)
    {
        copyStreakOutImpl(
            reinterpret_cast<const MEMBER*>(first),
            reinterpret_cast<const MEMBER*>(last),
            reinterpret_cast<EXTERNAL*>(target));
    }

    /**
     * Do not override this function! It is final.
     */
    void copyMemberIn(
        const char *source, CELL *target, int num, char CELL:: *memberPointer)
    {
        copyMemberInImpl(
            reinterpret_cast<const EXTERNAL*>(source),
            target,
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    /**
     * Do not override this function! It is final.
     */
    void copyMemberOut(
        const CELL *source, char *target, int num, char CELL:: *memberPointer)
    {
        copyMemberOutImpl(
            source,
            reinterpret_cast<EXTERNAL*>(target),
            num,
            reinterpret_cast<MEMBER (CELL:: *)[ARITY]>(memberPointer));
    }

    bool checkExternalTypeID(const std::type_info& otherID) const
    {
        return typeid(EXTERNAL) == otherID;
    }

};

}

#endif
