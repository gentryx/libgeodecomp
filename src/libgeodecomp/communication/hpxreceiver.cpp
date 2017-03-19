#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_HPX

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/geometry/coordbox.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

typedef LibGeoDecomp::CoordBox<1> CoordBox1;
typedef LibGeoDecomp::CoordBox<2> CoordBox2;
typedef LibGeoDecomp::CoordBox<3> CoordBox3;

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(char)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(double)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(float)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(int)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(CoordBox1)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(CoordBox2)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(CoordBox3)
#endif
