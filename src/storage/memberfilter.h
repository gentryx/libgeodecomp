#ifndef LIBGEODECOMP_STORAGE_MEMBERFILTER_H
#define LIBGEODECOMP_STORAGE_MEMBERFILTER_H

#include <libflatarray/cuda_array.hpp>
#include <libgeodecomp/storage/filter.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/misc/sharedptr.h>

namespace LibGeoDecomp {

/**
 * This filter is useful if an IO object (Writer or Steerer) needs to
 * work with a nested member of a simulation model, e.g. a member of a
 * member of a cell.
 *
 * It relies on existing Filters and Selectors to extract that data.
 * This is useful for limiting the volume of IO as well as relieving
 * users from writing custom filters.
 *
 * Example: consider the following Cell class. A user might want to
 * output Cell::sampleMember::foo. This could be achieved by creating
 * a Selector<Cell> and adding to that a suitable MemberFilter:
 *
 * class Cell {
 *   class ComplexMember {
 *     int foo;
 *   };
 *
 *  ComplexMember sampleMember;
 * };
 *
 * Selector<Cell> selector(
 *   &Cell::sampleMember,
 *   "sample",
 *   MemberFilter<Cell, ComplexMember>(&ComplexMember::foo));
 */
template<typename CELL, typename MEMBER>
class MemberFilter : public FilterBase<CELL>
{
public:
    template<typename MEMBERS_MEMBER>
    MemberFilter(MEMBERS_MEMBER MEMBER:: *membersMemberPointer) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo"))
    {}

    template<typename MEMBERS_MEMBER, int ARITY>
    MemberFilter(MEMBERS_MEMBER (MEMBER:: *membersMemberPointer)[ARITY]) :
        outerFilter(makeDefaultFilter<CELL, MEMBER>()),
        innerSelector(new Selector<MEMBER>(membersMemberPointer, "foo"))
    {}

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
        return innerSelector->sizeOfExternal();
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

    std::string typeName() const
    {
        return innerSelector->typeName();
    }

    int arity() const
    {
        return innerSelector->arity();
    }

    /**
     * Copy a Streak of variables to an AoS layout.
     */
    void copyStreakIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        innerSelector->copyMemberIn(
            source,
            sourceLocation,
            reinterpret_cast<MEMBER*>(target),
            targetLocation,
            num);
    }

    /**
     * Extract a Streak of members from an AoS layout.
     */
    void copyStreakOut(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        innerSelector->copyMemberOut(
            reinterpret_cast<const MEMBER*>(source),
            sourceLocation,
            target,
            targetLocation,
            num);
    }

    /**
     * Copy a Streak of variables to the members of a Streak of cells.
     */
    void copyMemberIn(
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
            innerSelector->copyMemberIn(
                reinterpret_cast<char*>(source),
                sourceLocation,
                &buf[0],
                MemoryLocation::CUDA_DEVICE,
                num);

            outerFilter->copyMemberIn(
                reinterpret_cast<MEMBER>(&buf[0]),
                MemoryLocation::CUDA_DEVICE,
                target,
                targetLocation,
                num,
                memberPointer);

            return;
        }
#endif

        std::vector<MEMBER> buf(num);
        innerSelector->copyMemberIn(
            reinterpret_cast<const char*>(source),
            sourceLocation,
            &buf[0],
            MemoryLocation::HOST,
            num);

        outerFilter->copyMemberIn(
            reinterpret_cast<char*>(&buf[0]),
            MemoryLocation::HOST,
            target,
            targetLocation,
            num,
            memberPointer);
    }

    /**
     * Extract a Streak of members from a Streak of cells.
     */
    void copyMemberOut(
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
            outerFilter->copyMemberOut(
                source,
                sourceLocation,
                reinterpret_cast<(char*>(&buf[0]),
                MemoryLocation::CUDA_DEVICE,
                num,
                memberPointer);

            innerSelector->copyMemberOut(
                &buf[0],
                MemoryLocation::CUDA_DEVICE,
                reinterpret_cast<char*>(target),
                targetLocation,
                num);

            return;
        }
#endif

        std::vector<MEMBER> buf(num);
        outerFilter->copyMemberOut(
            source,
            sourceLocation,
            reinterpret_cast<char*>(&buf[0]),
            MemoryLocation::HOST,
            num,
            memberPointer);

        innerSelector->copyMemberOut(
            &buf[0],
            MemoryLocation::HOST,
            reinterpret_cast<char*>(target),
            targetLocation,
            num);
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
