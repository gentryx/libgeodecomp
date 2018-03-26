#ifndef LIBGEODECOMP_STORAGE_SERIALIZATIONBUFFER_H
#define LIBGEODECOMP_STORAGE_SERIALIZATIONBUFFER_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

namespace SerializationBufferHelpers {

/**
 * This is an n-way switch to allow other classes to select the
 * appropriate type to buffer regions of a grid; for use with
 * GridBase::loadRegion() and saveRegion().
 */
template<typename CELL, typename SUPPORTS_SOA = void, typename SUPPORTS_BOOST_SERIALIZATION = void>
class Implementation
{
public:
    typedef std::vector<CELL> BufferType;
    typedef CELL ElementType;
    typedef typename APITraits::TrueType FixedSize;

    template<typename REGION>
    static BufferType create(const REGION& region)
    {
        return BufferType(minimumStorageSize(region));
    }

    template<typename REGION>
    static std::size_t minimumStorageSize(const REGION& region)
    {
        return region.size();
    }

    template<typename REGION>
    static void resize(BufferType *buffer, const REGION& region)
    {
        return buffer->resize(minimumStorageSize(region));
    }

    static ElementType *getData(BufferType& buffer)
    {
        return &buffer.front();
    }

#ifdef LIBGEODECOMP_WITH_MPI
    static inline MPI_Datatype cellMPIDataType()
    {
        return APITraits::SelectMPIDataType<ElementType>::value();
    }
#endif
};

/**
 * see above
 */
template<typename CELL>
class Implementation<CELL, typename CELL::API::SupportsSoA, void>
{
public:
    typedef std::vector<char> BufferType;
    typedef char ElementType;
    typedef typename APITraits::TrueType FixedSize;

    template<typename REGION>
    static BufferType create(const REGION& region)
    {
        return BufferType(minimumStorageSize(region));
    }

    template<typename REGION>
    static std::size_t minimumStorageSize(const REGION& region)
    {
        return LibFlatArray::aggregated_member_size<CELL>::VALUE * region.size();
    }

    template<typename REGION>
    static void resize(BufferType *buffer, const REGION& region)
    {
        return buffer->resize(minimumStorageSize(region));
    }

    static ElementType *getData(BufferType& buffer)
    {
        return &buffer.front();
    }

#ifdef LIBGEODECOMP_WITH_MPI
    static inline MPI_Datatype cellMPIDataType()
    {
        return MPI_CHAR;
    }
#endif
};

/**
 * see above
 */
// temporarily disabled until #46 ( https://github.com/gentryx/libgeodecomp/issues/46 ) is fixed.

// template<typename CELL>
// class Implementation<CELL, void, typename CELL::API::SupportsBoostSerialization>
// {
// public:
//     typedef std::vector<char> BufferType;
//     typedef char ElementType;
//     typedef typename APITraits::FalseType FixedSize;

//     template<typename REGION>
//     static BufferType create(const REGION& region)
//     {
//         return BufferType();
//     }

//     template<typename REGION>
//     static std::size_t minimumStorageSize(const REGION& region)
//     {
//         return region.size();
//     }

//     template<typename REGION>
//     static void resize(BufferType *buffer, const REGION& region)
//     {
//         buffer->resize(minimumStorageSize(region));
//     }
//     static ElementType *getData(BufferType& buffer)
//     {
//         return &buffer.front();
//     }

// #ifdef LIBGEODECOMP_WITH_MPI
//     static inline MPI_Datatype cellMPIDataType()
//     {
//         return MPI_CHAR;
//     }
// #endif
// };

}

/**
 * This class provides a uniform interface to the different buffer
 * types to be used with GridVecConv.
 */
template<typename CELL>
class SerializationBuffer
{
public:
    typedef SerializationBufferHelpers::Implementation<CELL> Implementation;
    typedef typename Implementation::BufferType BufferType;
    typedef typename Implementation::ElementType ElementType;
    typedef typename Implementation::FixedSize FixedSize;

    template<typename REGION>
    static inline BufferType create(const REGION& region)
    {
        return Implementation::create(region);
    }

    static inline ElementType *getData(BufferType& buffer)
    {
        return Implementation::getData(buffer);
    }

    /**
     * Returns the minimum number of bytes that are required to store
     * the number of cells as outlined by the Region.
     *
     * For standard and SoA models this is generally identical with
     * the amount of actually allocated memory. For dynamic
     * serialization such as Boost Serialization the actual ammount is
     * often larger.
     */
    template<typename REGION>
    static std::size_t minimumStorageSize(const REGION& region)
    {
        return Implementation::minimumStorageSize(region);
    }

    template<typename REGION>
    static inline void resize(BufferType *buffer, const REGION& region)
    {
        Implementation::resize(buffer, region);
    }

#ifdef LIBGEODECOMP_WITH_MPI
    static inline  MPI_Datatype cellMPIDataType()
    {
        return Implementation::cellMPIDataType();
    }
#endif
};

}

#endif
