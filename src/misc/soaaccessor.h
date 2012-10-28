#ifndef _libgeodecomp_misc_soaaccessor_h_
#define _libgeodecomp_misc_soaaccessor_h_

// fixme: check for boost pp in configure
#include <boost/preprocessor/seq.hpp>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/fixedcoord.h>

namespace LibGeoDecomp {

namespace SoAHelpers {

template<typename CELL, int I>
class Offset;

template<typename CELL>
class Offset<CELL, 0>
{
 public:
    static const int OFFSET = 0;
};

}

// fixme: ugly hack for compatibility with non-cuda builds
#ifndef __host__
#define __host__
#endif 

#ifndef __device__
#define __device__
#endif 

#define DEFINE_FIELD_OFFSET(r, CELL_TYPE, t)                            \
    namespace SoAHelpers {                                              \
    template<>                                                          \
    class Offset<CELL_TYPE, r - 1>                                      \
    {                                                                   \
    public:                                                             \
        static const int OFFSET = Offset<CELL_TYPE, r - 2>::OFFSET +    \
            sizeof(BOOST_PP_SEQ_ELEM(0, t));                            \
    };                                                                  \
    }                                                                   

#define DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, CONST)           \
    inline                                                              \
    __host__ __device__                                                 \
    CONST BOOST_PP_SEQ_ELEM(0, MEMBER)& BOOST_PP_SEQ_ELEM(1, MEMBER)() CONST \
    {                                                                   \
        return  *(BOOST_PP_SEQ_ELEM(0, MEMBER)*)(                       \
            data +                                                      \
            (DIM_X * DIM_Y * DIM_Z) * SoAHelpers::Offset<CELL, MEMBER_INDEX - 2>::OFFSET + \
            *index * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)) +             \
            INDEX  * sizeof(BOOST_PP_SEQ_ELEM(0, MEMBER)));             \
    }                                                                   

#define DECLARE_SOA_MEMBER_CONST(MEMBER_INDEX, CELL, MEMBER)    \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, const)

#define DECLARE_SOA_MEMBER_NORMAL(MEMBER_INDEX, CELL, MEMBER)   \
    DECLARE_SOA_MEMBER(MEMBER_INDEX, CELL, MEMBER, )

#define COPY_SOA_MEMBER_IN(MEMBER_INDEX, CELL, MEMBER)                  \
    BOOST_PP_SEQ_ELEM(1, MEMBER)() = cell.BOOST_PP_SEQ_ELEM(1, MEMBER);

#define COPY_SOA_MEMBER_OUT(MEMBER_INDEX, CELL, MEMBER)                 \
    cell.BOOST_PP_SEQ_ELEM(1, MEMBER) = soa.BOOST_PP_SEQ_ELEM(1, MEMBER)();

/**
 * This class provides an object-oriented view to a "Struct of
 * Arrays"-style grid. It requires the user to register the type CELL
 * using the macro LIBGEODECOMP_REGISTER_SOA.
 */
template<typename CELL, int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
class SoAAccessor;

#define LIBGEODECOMP_REGISTER_SOA(CELL_TYPE, CELL_MEMBERS)              \
    namespace LibGeoDecomp {                                            \
    BOOST_PP_SEQ_FOR_EACH(                                              \
        DEFINE_FIELD_OFFSET,                                            \
        CELL_TYPE,                                                      \
        CELL_MEMBERS)                                                   \
                                                                        \
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>                \
    class SoAAccessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX>            \
    {                                                                   \
    public:                                                             \
        typedef CELL_TYPE MyCell;                                       \
                                                                        \
        __host__ __device__                                             \
            SoAAccessor(char *_data=0, int *_index=0) :                 \
            data(_data),                                                \
            index(_index)                                               \
            {}                                                          \
                                                                        \
        template<int X, int Y, int Z>                                   \
            inline                                                      \
            __host__ __device__                                         \
            SoAAccessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X> operator[](FixedCoord<X, Y, Z>) \
            {                                                           \
                return SoAAccessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX + Z * (DIM_X * DIM_Y) + Y * DIM_X + X>(data, index); \
            }                                                           \
                                                                        \
        __host__ __device__                                             \
            inline                                                      \
            void operator=(const CELL_TYPE& cell)                       \
            {                                                           \
                BOOST_PP_SEQ_FOR_EACH(                                  \
                    COPY_SOA_MEMBER_IN,                                 \
                    CELL_TYPE,                                          \
                    CELL_MEMBERS);                                      \
            }                                                           \
                                                                        \
        __host__ __device__                                             \
            inline                                                      \
            void operator<<(const CELL_TYPE& cell)                      \
            {                                                           \
                (*this) = cell;                                         \
            }                                                           \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_SOA_MEMBER_NORMAL,                                  \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_SOA_MEMBER_CONST,                                   \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
                                                                        \
    private:                                                            \
        char *data;                                                     \
        int *index;                                                     \
    };                                                                  \
    }                                                                   \
                                                                        \
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>                \
    __host__ __device__                                                 \
    inline                                                              \
    void operator<<(                                                    \
        CELL_TYPE& cell,                                          \
        const LibGeoDecomp::SoAAccessor<CELL_TYPE, DIM_X, DIM_Y, DIM_Z, INDEX> soa) \
    {                                                                   \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            COPY_SOA_MEMBER_OUT,                                        \
            CELL_TYPE,                                                  \
            CELL_MEMBERS);                                              \
    }
}

#endif
