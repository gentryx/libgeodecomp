#ifndef LIBGEODECOMP_STORAGE_FILTERBASE_H
#define LIBGEODECOMP_STORAGE_FILTERBASE_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/memorylocation.h>
#include <typeinfo>

#ifdef LIBGEODECOMP_WITH_MPI
#include <libgeodecomp/communication/typemaps.h>
#endif

namespace LibGeoDecomp {

namespace FilterBaseHelpers {

/**
 * We're intentionally giving only few specializations for this helper
 * as it's mostly meant to be used with VisIt's BOV format, and this
 * is only defined on three types.
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

template<typename CELL>
std::string filterBasePrimitiveTypeName()
{
    return FilterBaseHelpers::GetTypeName<CELL>()();
}

/**
 * Base class for adding user-defined data filters to a Selector.
 * This can be used to do on-the-fly data extraction, scale
 * conversion for live output etc. without having to rewrite a
 * complete ParallelWriter output plugin.
 *
 * It is suggested to derive from Filter instead of FilterBase, as
 * the latter has some convenience functionality already in place.
 */
template<typename CELL>
class FilterBase
{
public:
    virtual ~FilterBase()
    {}

    virtual std::size_t sizeOf() const = 0;
#ifdef LIBGEODECOMP_WITH_SILO
    virtual int siloTypeID() const = 0;
#endif
#ifdef LIBGEODECOMP_WITH_MPI
    /**
     * Yields the member's MPI data type (or that of its external
     * representation). May source from APITraits or fall back to
     * Typemaps. If neither yields, no compiler error will follow
     * as it is assumed that such code is still valid (e.g. if a
     * Selector is instantiated for the SiloWriter, so that
     * mpiDatatype() is never called).
     */
    virtual MPI_Datatype mpiDatatype() const = 0;
#endif
    virtual std::string typeName() const = 0;
    virtual int arity() const = 0;

    /**
     * The stride denotes by how many elements two components of an
     * array member (vector) are separated. In an SoA layout a member
     * "double foo[3]" will be stored as a field of foo[0] of size
     * "stride", followed by all values for "foo[1]" and finally
     * "foo[2]".
     */
    virtual void copyStreakIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) = 0;

    virtual void copyStreakOut(
        const char *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t stride) = 0;

    virtual void copyMemberIn(
        const char *source,
        MemoryLocation::Location sourceLocation,
        CELL *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer) = 0;

    virtual void copyMemberOut(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        char *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        char CELL:: *memberPointer) = 0;

    virtual bool checkExternalTypeID(const std::type_info& otherID) const = 0;
};

}

#endif
