#ifndef LIBGEODECOMP_STORAGE_SELECTOR_H
#define LIBGEODECOMP_STORAGE_SELECTOR_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libflatarray/member_ptr_to_offset.hpp>
#include <libgeodecomp/storage/defaultfilterfactory.h>
#include <libgeodecomp/storage/filterbase.h>
#include <libgeodecomp/storage/memberfilter.h>
#include <stdexcept>
#include <typeinfo>

#ifdef LIBGEODECOMP_WITH_SILO
#include <silo.h>
#endif

#ifdef LIBGEODECOMP_WITH_MPI
#include <libgeodecomp/communication/typemaps.h>
#endif

namespace LibGeoDecomp {

class APITraits;

template<typename CELL, typename MEMBER>
class MemberFilter;

namespace SelectorHelpers {

/**
 * A member's offset is basically the sum of all its preceeding
 * members (storage-order-wise). To calculate an actuall address, this
 * offset then needs to be scaled by the number of elements in the
 * grid/array.
 */
template<typename CELL, typename MEMBER>
class GetMemberOffset
{
public:
    int operator()(MEMBER CELL:: *memberPointer, APITraits::TrueType)
    {
        return LibFlatArray::member_ptr_to_offset()(memberPointer);
    }

    template<int ARITY>
    int operator()(MEMBER (CELL:: *memberPointer)[ARITY], APITraits::TrueType)
    {
        return LibFlatArray::member_ptr_to_offset()(memberPointer);
    }

    /**
     * Dummy return value, won't ever be used as the model doesn't
     * support SoA anyway.
     */
    int operator()(MEMBER CELL::* /* memberPointer */, APITraits::FalseType)
    {
        return -1;
    }

    /**
     * Same as above, but for array members.
     */
    template<int ARITY>
    int operator()(MEMBER (CELL::* /* memberPointer */)[ARITY], APITraits::FalseType)
    {
        return -1;
    }
};

/**
 * Primitive datatypes don't have member pointers (or members in the
 * first place). So we provide this primitive implementation to copy
 * them en block.
 */
template<typename CELL>
class PrimitiveSelector
{
public:
    explicit PrimitiveSelector(const std::string& variableName = "primitiveType") :
        variableName(variableName)
    {}

    template<typename MEMBER>
    void operator()(const CELL *source, MEMBER *target, const std::size_t length) const
    {
        std::copy(source, source + length, target);
    }

    int arity() const
    {
        return 1;
    }

    std::string typeName() const
    {
        return filterBasePrimitiveTypeName<CELL>();
    }

    const std::string name() const
    {
        return variableName;
    }

#ifdef LIBGEODECOMP_WITH_MPI
    MPI_Datatype mpiDatatype() const
    {
        return Typemaps::lookup<CELL>();
    }
#endif

    std::size_t sizeOfMember() const
    {
        return sizeof(CELL);
    }

    std::size_t sizeOfExternal() const
    {
        return sizeof(CELL);
    }

    int offset() const
    {
        return 0;
    }

    void copyMemberIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num) const
    {
        if (sourceLocation != MemoryLocation::HOST) {
            throw std::logic_error("PrimitiveSelector is for test purposes only,"
                                   " it's limited to HOST source memory (1)");
        }
        if (targetLocation != MemoryLocation::HOST) {
            throw std::logic_error("PrimitiveSelector is for test purposes only,"
                                   " it's limited to HOST target memory (1)");
        }


        const CELL *actualSource = reinterpret_cast<const CELL*>(source);
        std::copy(actualSource, actualSource + num, target);
    }

    void copyMemberOut(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        std::size_t num) const
    {
        if (sourceLocation != MemoryLocation::HOST) {
            throw std::logic_error("PrimitiveSelector is for test purposes only,"
                                   " it's limited to HOST source memory (2)");
        }
        if (targetLocation != MemoryLocation::HOST) {
            throw std::logic_error("PrimitiveSelector is for test purposes only,"
                                   " it's limited to HOST target memory (2)");
        }

        CELL *actualTarget = reinterpret_cast<CELL*>(target);
        std::copy(source, source + num, actualTarget);
    }

    // intentionally leaving our copyStreak() as it should only be
    // called by SoAGrid, which isn't defined for primitive types
    // anyway.

private:
    std::string variableName;
};

}

/**
 * A Selector can be used by library code to extract data from user
 * code, e.g. so that writers can access a cell's member variable.
 *
 * The code is based on a rather obscure C++ feature. Thanks to Evan
 * Wallace for pointing this out:
 * http://madebyevan.com/obscure-cpp-features/#pointer-to-member-operators
 *
 * fixme: use this class in RemoteSteerer...
 */
template<typename CELL>
class Selector
{
public:
    friend class PPMWriterTest;

    Selector() :
        memberPointer(0),
        memberSize(0),
        externalSize(0),
        memberOffset(0),
        memberName("memberName not initialized")
    {}

    virtual ~Selector()
    {}

    template<typename MEMBER, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER CELL:: *memberPointer,
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBER)),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(DefaultFilterFactory<FORCE_CUDA>().template make<CELL, MEMBER>())
    {}

    template<typename MEMBER, int ARITY, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER (CELL:: *memberPointer)[ARITY],
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBER) * ARITY),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(DefaultFilterFactory<FORCE_CUDA>().template make<CELL, MEMBER, ARITY>())
    {}

    template<typename MEMBER>
    Selector(
        MEMBER CELL:: *memberPointer,
        const std::string& memberName,
        const typename SharedPtr<FilterBase<CELL> >::Type& filter) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(filter->sizeOf()),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(filter)
    {}

    template<typename MEMBER, int ARITY>
    Selector(
        MEMBER (CELL:: *memberPointer)[ARITY],
        const std::string& memberName,
        const typename SharedPtr<FilterBase<CELL> >::Type& filter) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(filter->sizeOf()),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(filter)
    {}

    template<typename MEMBER, typename MEMBERS_MEMBER, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER CELL:: *memberPointer,
        MEMBERS_MEMBER MEMBER:: *membersMemberPointer,
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBERS_MEMBER)),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(makeShared(new MemberFilter<CELL, MEMBER>(membersMemberPointer)))
    {}

    template<typename MEMBER, typename MEMBERS_MEMBER, int ARITY, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER CELL:: *memberPointer,
        MEMBERS_MEMBER (MEMBER:: *membersMemberPointer)[ARITY],
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBERS_MEMBER) * ARITY),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(makeShared(new MemberFilter<CELL, MEMBER>(membersMemberPointer)))
    {}

    template<typename MEMBER, typename MEMBERS_MEMBER, typename MEMBERS_MEMBERS_MEMBER, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER CELL:: *memberPointer,
        MEMBERS_MEMBER MEMBER:: *membersMemberPointer,
        MEMBERS_MEMBERS_MEMBER MEMBERS_MEMBER:: *membersMembersMemberPointer,
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBERS_MEMBERS_MEMBER)),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(makeShared(
                   new MemberFilter<CELL, MEMBER>(
                       membersMemberPointer,
                       makeShared(
                           new MemberFilter<MEMBER, MEMBERS_MEMBER>(membersMembersMemberPointer)))))
    {}

    template<typename MEMBER, typename MEMBERS_MEMBER, typename MEMBERS_MEMBERS_MEMBER, int ARITY, bool FORCE_CUDA =
#ifdef __CUDACC__
             true
#else
             false
#endif
             >
    Selector(
        MEMBER CELL:: *memberPointer,
        MEMBERS_MEMBER MEMBER:: *membersMemberPointer,
        MEMBERS_MEMBERS_MEMBER (MEMBERS_MEMBER:: *membersMembersMemberPointer)[ARITY],
        const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBERS_MEMBERS_MEMBER) * ARITY),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        filter(makeShared(
                   new MemberFilter<CELL, MEMBER>(
                       membersMemberPointer,
                       makeShared(new MemberFilter<MEMBER, MEMBERS_MEMBER>(membersMembersMemberPointer)))))
    {}

    inline const std::string& name() const
    {
        return memberName;
    }

    inline std::size_t sizeOfMember() const
    {
        return memberSize;
    }

    inline std::size_t sizeOfExternal() const
    {
        return externalSize;
    }

    template<typename MEMBER>
    inline bool checkTypeID() const
    {
        return filter->checkExternalTypeID(typeid(MEMBER));
    }

    bool checkExternalTypeID(const std::type_info& otherID) const
    {
        return filter->checkExternalTypeID(otherID);
    }

    /**
     * The member's offset in LibFlatArray's SoA memory layout
     */
    inline int offset() const
    {
        return memberOffset;
    }

    /**
     * Read the data from source and set the corresponding member of
     * each CELL at target. Only useful for AoS memory layout.
     */
    inline void copyMemberIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num) const
    {
        filter->copyMemberIn(source, sourceLocation, target, targetLocation, num, memberPointer);
    }

    /**
     * Read the member of all CELLs at source and store them
     * contiguously at target. Only useful for AoS memory layout.
     */
    inline void copyMemberOut(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        std::size_t num) const
    {
        filter->copyMemberOut(source, sourceLocation, target, targetLocation, num, memberPointer);
    }

    /**
     * This is a helper function for writing members of a SoA memory
     * layout.
     */
    inline void copyStreakIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) const
    {
        filter->copyStreakIn(source, sourceLocation, target, targetLocation, num, stride);
    }

    /**
     * This is a helper function for reading members from a SoA memory
     * layout.
     */
    inline void copyStreakOut(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) const
    {
        filter->copyStreakOut(source, sourceLocation, target, targetLocation, num, stride);
    }

#ifdef LIBGEODECOMP_WITH_SILO
    int siloTypeID() const
    {
        return filter->siloTypeID();
    }
#endif

#ifdef LIBGEODECOMP_WITH_MPI
    MPI_Datatype mpiDatatype() const
    {
        return filter->mpiDatatype();
    }
#endif

    std::string typeName() const
    {
        return filter->typeName();
    }

    int arity() const
    {
        return filter->arity();
    }

    template<class ARCHIVE>
    void serialize(ARCHIVE& archive, const unsigned int version)
    {
        std::size_t *buf =
            reinterpret_cast<std::size_t*>(
                const_cast<char CELL::**>(&memberPointer));
        archive & *buf;

        archive & memberSize;
        archive & externalSize;
        archive & memberOffset;
        archive & memberName;
        archive & filter;
    }

private:
    char CELL:: *memberPointer;
    std::size_t memberSize;
    std::size_t externalSize;
    int memberOffset;
    std::string memberName;
    typename SharedPtr<FilterBase<CELL> >::Type filter;
};

/**
 * We provide these specializations to allow our unit tests to use
 * grids with primitive datatypes.
 */
template<>
class Selector<char> : public SelectorHelpers::PrimitiveSelector<char>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<char>(name)
    {}
};

/**
 * see above
 */
template<>
class Selector<unsigned char> : public SelectorHelpers::PrimitiveSelector<unsigned char>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<unsigned char>(name)
    {}
};

/**
 * see above
 */
template<>
class Selector<int> : public SelectorHelpers::PrimitiveSelector<int>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<int>(name)
    {}
};

/**
 * see above
 */
template<>
class Selector<unsigned> : public SelectorHelpers::PrimitiveSelector<unsigned>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<unsigned>(name)
    {}
};

/**
 * see above
 */
template<>
class Selector<float> : public SelectorHelpers::PrimitiveSelector<float>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<float>(name)
    {}
};

/**
 * see above
 */
template<>
class Selector<double> : public SelectorHelpers::PrimitiveSelector<double>
{
public:
    explicit Selector(const std::string name = "primitiveType") :
        SelectorHelpers::PrimitiveSelector<double>(name)
    {}
};

}

#endif
