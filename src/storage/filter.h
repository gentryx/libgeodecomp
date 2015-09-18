#ifndef LIBGEODECOMP_STORAGE_FILTER_H
#define LIBGEODECOMP_STORAGE_FILTER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/filterbase.h>
#include <libgeodecomp/storage/memorylocation.h>

#ifdef LIBGEODECOMP_WITH_SILO
#include <silo.h>
#endif

#ifdef LIBGEODECOMP_WITH_MPI
#include <libgeodecomp/communication/typemaps.h>
#endif

namespace LibGeoDecomp {

namespace FilterHelpers {

#ifdef LIBGEODECOMP_WITH_SILO

/**
 * some helper classes required to set the type constant for SILO's
 * library calls:
 */
template<typename MEMBER>
class GetSiloTypeID
{
public:
    inline int operator()()
    {
        LOG(WARN, "Warning: using type unknown to Silo for output");
        return DB_NOTYPE;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<int>
{
public:
    inline int operator()()
    {
        return DB_INT;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<short int>
{
public:
    inline int operator()()
    {
        return DB_SHORT;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<float>
{
public:
    inline int operator()()
    {
        return DB_FLOAT;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<double>
{
public:
    inline int operator()()
    {
        return DB_DOUBLE;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<char>
{
public:
    inline int operator()()
    {
        return DB_CHAR;
    }
};

/**
 * ditto
 */
template<>
class GetSiloTypeID<long long>
{
public:
    inline int operator()()
    {
        return DB_LONG_LONG;
    }
};

#endif

#ifdef LIBGEODECOMP_WITH_MPI

/**
 * see below
 */
template<typename MEMBER, int FLAG>
class GetMPIDatatype0;

/**
 * see below
 */
template<typename MEMBER, int FLAG>
class GetMPIDatatype1;

/**
 * see below
 */
template<typename MEMBER>
class GetMPIDatatype0<MEMBER, 0>
{
public:
    inline MPI_Datatype operator()()
    {
        throw std::invalid_argument("no MPI data type defined for this type");
    }
};

/**
 * see below
 */
template<typename MEMBER>
class GetMPIDatatype0<MEMBER, 1>
{
public:
    inline MPI_Datatype operator()()
    {
        return Typemaps::lookup<MEMBER>();
    }
};

/**
 * see below
 */
template<typename MEMBER>
class GetMPIDatatype1<MEMBER, 0>
{
public:
    inline MPI_Datatype operator()()
    {
        return GetMPIDatatype0<MEMBER, APITraits::HasLookupMemberFunction<Typemaps, MPI_Datatype, MEMBER>::value>()();
    }
};

/**
 * see below
 */
template<typename MEMBER>
class GetMPIDatatype1<MEMBER, 1>
{
public:
    inline MPI_Datatype operator()()
    {
        return APITraits::SelectMPIDataType<MEMBER>::value();
    }
};

/**
 * This class is a shim to deduce a member's MPI data type via, different methods are tried:
 */
template<typename MEMBER>
class GetMPIDatatype
{
public:
    inline MPI_Datatype operator()()
    {
        return GetMPIDatatype1<
            MEMBER,
            APITraits::HasValueFunction<APITraits::SelectMPIDataType<MEMBER>, MPI_Datatype>::value>()();
    }
};

#endif

/**
 * We're intentionally giving only few specializations for this helper
 * as it's mostly meant to be used with VisIt's BOV format, and this
 * is only defined on tree types.
 */
template<typename MEMBER>
class GetTypeName
{
public:
    std::string operator()() const
    {
        throw std::invalid_argument("no string representation known for member type");
    }
};

/**
 * see above
 */
template<>
class GetTypeName<bool>
{
public:
    std::string operator()() const
    {
        return "BYTE";
    }
};

/**
 * see above
 */
template<>
class GetTypeName<char>
{
public:
    std::string operator()() const
    {
        return "BYTE";
    }
};

/**
 * see above
 */
template<>
class GetTypeName<float>
{
public:
    std::string operator()() const
    {
        return "FLOAT";
    }
};

/**
 * see above
 */
template<>
class GetTypeName<double>
{
public:
    std::string operator()() const
    {
        return "DOUBLE";
    }
};

/**
 * see above
 */
template<>
class GetTypeName<int>
{
public:
    std::string operator()() const
    {
        return "INT";
    }
};

/**
 * see above
 */
template<>
class GetTypeName<long>
{
public:
    std::string operator()() const
    {
        return "LONG";
    }
};

}

/**
 * Derive from this class if you wish to add custom data
 * adapters/converters to your Selector. Useful for scalar members,
 * refer to ArrayFilter for array members (e.g. int Cell::foo[4]).
 */
template<typename CELL, typename MEMBER, typename EXTERNAL>
class Filter : public FilterBase<CELL>
{
public:
    friend class PolymorphicSerialization;
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class PPMWriterTest;

    std::size_t sizeOf() const
    {
        return sizeof(EXTERNAL);
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
        return 1;
    }

    /**
     * Copy a streak of variables to an AoS layout.
     */
    virtual void copyStreakInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        MEMBER *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) = 0;

    /**
     * Extract a steak of members from an AoS layout.
     */
    virtual void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) = 0;

    /**
     * Copy a streak of variables to the members of a streak of cells.
     */
    virtual void copyMemberInImpl(
        const EXTERNAL *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer) = 0;

    /**
     * Extract a streak of members from a streak of cells.
     */
    virtual void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        EXTERNAL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer) = 0;

    /**
     * Do not override this function! It is final.
     */
    void copyStreakIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        copyStreakInImpl(
            reinterpret_cast<const EXTERNAL*>(source),
            sourceLocation,
            reinterpret_cast<MEMBER*>(target),
            targetLocation,
            num,
            stride);
    }

    /**
     * Do not override this function! It is final.
     */
    void copyStreakOut(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride)
    {
        copyStreakOutImpl(
            reinterpret_cast<const MEMBER*>(source),
            sourceLocation,
            reinterpret_cast<EXTERNAL*>(target),
            targetLocation,
            num,
            stride);
    }

    /**
     * Do not override this function! It is final.
     */
    void copyMemberIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer)
    {
        copyMemberInImpl(
            reinterpret_cast<const EXTERNAL*>(source),
            sourceLocation,
            target,
            targetLocation,
            num,
            reinterpret_cast<MEMBER CELL:: *>(memberPointer));
    }

    /**
     * Do not override this function! It is final.
     */
    void copyMemberOut(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer)
    {
        copyMemberOutImpl(
            source,
            sourceLocation,
            reinterpret_cast<EXTERNAL*>(target),
            targetLocation,
            num,
            reinterpret_cast<MEMBER CELL:: *>(memberPointer));
    }

    bool checkExternalTypeID(const std::type_info& otherID) const
    {
        return typeid(EXTERNAL) == otherID;
    }
};

}

#endif
