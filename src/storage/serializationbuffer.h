#ifndef LIBGEODECOMP_STORAGE_SERIALIZATIONBUFFER_H
#define LIBGEODECOMP_STORAGE_SERIALIZATIONBUFFER_H

#include <libflatarray/flat_array.hpp>

namespace LibGeoDecomp {

namespace SerializationBufferHelpers
{
    /**
     * This is an n-way switch to allow other classes to select the
     * appropriate type to buffer regions of a grid; for use with GridVecConv.
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
            return BufferType(region.size());
        }

        static ElementType *getData(BufferType& buffer)
        {
            return &buffer.first();
        }

#ifdef LIBGEODECOMP_FEATURE_MPI
        static inline MPI_Datatype cellMPIDataType()
        {
            return APITraits::SelectMPIDataType<ElementType>::value();
        }
#endif
    };

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
            return BufferType(LibFlatArray::aggregated_member_size<CELL>::VALUE * region.size());
        }

        static ElementType *getData(BufferType& buffer)
        {
            return &buffer.front();
        }

#ifdef LIBGEODECOMP_FEATURE_MPI
        static inline MPI_Datatype cellMPIDataType()
        {
            return MPI_CHAR;
        }
#endif
    };

    template<typename CELL>
    class Implementation<CELL, void, typename CELL::API::SupportsBoostSerialization>
    {
    public:
        typedef std::vector<char> BufferType;
        typedef char ElementType;
        typedef typename APITraits::FalseType FixedSize;

        template<typename REGION>
        static BufferType create(const REGION& region)
        {
            return BufferType();
        }

        static ElementType *getData(BufferType& buffer)
        {
            return &buffer.front();
        }

#ifdef LIBGEODECOMP_FEATURE_MPI
        static inline MPI_Datatype cellMPIDataType()
        {
            return MPI_CHAR;
        }
#endif
    };
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

#ifdef LIBGEODECOMP_FEATURE_MPI
    static inline  MPI_Datatype cellMPIDataType()
    {
        return Implementation::cellMPIDataType();
    }
#endif
};

}

#endif
