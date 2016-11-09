#ifndef LIBGEODECOMP_STORAGE_MEMBERFILTER_H
#define LIBGEODECOMP_STORAGE_MEMBERFILTER_H

#include <libflatarray/cuda_array.hpp>
#include <libgeodecomp/storage/filter.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/misc/sharedptr.h>

namespace LibGeoDecomp {

/**
 * This filter relies on Selectors to extract data from a member of a
 * cell. This is useful if a user wants to write out data that's not a
 * plain member of a cell. For example, in the following Cell class, a
 * user might want to output Cell::member::foo:
 *
 * class Cell {
 *   class ComplexMember {
 *     int foo;
 *   };
 *
 *  ComplexMember member;
 * };
 */
template<typename CELL, typename MEMBER>
// fixme: check external type, e.g. via loadmember/savemember
// fixme: should be identical to members_member
// fixme: how to handle array members_member?
// fixme: solution: inherit from filterbase, handle initial extraction via selector and using the selector's internal filter
class MemberFilter : public FilterBase<CELL>
{
public:
    template<typename MEMBERS_MEMBER>
    MemberFilter(MEMBERS_MEMBER MEMBER:: *membersMemberPointer) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo"))
    {}

    template<typename MEMBERS_MEMBER, int ARITY>
    MemberFilter(MEMBERS_MEMBER (MEMBER:: *membersMemberPointer[ARITY])) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo"))
    {}

    // fixme: needs test
    template<typename MEMBERS_MEMBER>
    MemberFilter(
        MEMBERS_MEMBER MEMBER:: *membersMemberPointer,
        const typename SharedPtr<FilterBase<MEMBER> >::Type& filter) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo", filter))
    {}

    template<typename MEMBERS_MEMBER, int ARITY>
    MemberFilter(
        MEMBERS_MEMBER (MEMBER:: *membersMemberPointer)[ARITY],
        const typename SharedPtr<FilterBase<MEMBER> >::Type& filter) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo", filter))
    {}

    std::size_t sizeOf() const
    {
        return innerSelector->sizeOfMember();
    }

#ifdef LIBGEODECOMP_WITH_SILO
    int siloTypeID() const
    {
        return innerSelector->siloTypeID();
    }
#endif

#ifdef LIBGEODECOMP_WITH_MPI
    MPI_Datatype mpiDatatype() const
    {
        return innerSelector->mpiDatatype();
    }
#endif

    virtual std::string typeName() const
    {
        return innerSelector->typeName();
    }

    virtual int arity() const
    {
        return innerSelector->arity();
    }

    /**
     * Copy a Streak of variables to an AoS layout.
     */
    virtual void copyStreakIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        // fixme: needs test
    }

    /**
     * Extract a Streak of members from an AoS layout.
     */
    virtual void copyStreakOut(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        // fixme: needs test
    }

    /**
     * Copy a Streak of variables to the members of a Streak of cells.
     */
    virtual void copyMemberIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer)
    {
#ifdef __CUDACC__
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            // fixme: needs test
            LibFlatArray::cuda_array<MEMBER> buf(num);
            innerSelector->copyMemberIn((char*)source, sourceLocation, &buf[0], MemoryLocation::CUDA_DEVICE, num);
            outerFilter->copyMemberIn((char*)&buf[0], MemoryLocation::CUDA_DEVICE, target, targetLocation, num, memberPointer);
            return;
        }
#endif

        std::vector<MEMBER> buf(num);
        innerSelector->copyMemberIn((char*)source, sourceLocation, &buf[0], MemoryLocation::HOST, num);
        outerFilter->copyMemberIn((char*)&buf[0], MemoryLocation::HOST, target, targetLocation, num, memberPointer);
    }

    /**
     * Extract a Streak of members from a Streak of cells.
     */
    virtual void copyMemberOut(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer)
    {
#ifdef __CUDACC__
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) &&
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            // fixme: needs test
            LibFlatArray::cuda_array<MEMBER> buf(num);
            outerFilter->copyMemberOut(source, sourceLocation, (char*)&buf[0], MemoryLocation::CUDA_DEVICE, num, memberPointer);
            innerSelector->copyMemberOut(&buf[0], MemoryLocation::CUDA_DEVICE, (char*)target, targetLocation, num);
            return;
        }
#endif

        std::vector<MEMBER> buf(num);
        outerFilter->copyMemberOut(source, sourceLocation, (char*)&buf[0], MemoryLocation::HOST, num, memberPointer);
        innerSelector->copyMemberOut(&buf[0], MemoryLocation::HOST, (char*)target, targetLocation, num);
    }

    bool checkExternalTypeID(const std::type_info& otherID) const
    {
        return innerSelector->checkExternalTypeID(otherID);
    }

private:
    typename SharedPtr<FilterBase<CELL> >::Type outerFilter;
    typename SharedPtr<Selector<MEMBER> >::Type innerSelector;
};


}

#endif
