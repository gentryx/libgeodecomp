#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/communication/hpxreceiver.h>
#include <libgeodecomp/geometry/coordbox.h>

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

HPX_REGISTER_COMPONENT_MODULE()
#endif
