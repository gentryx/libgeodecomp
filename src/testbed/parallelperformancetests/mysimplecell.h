#ifndef LIBGEODECOMP_TESTBED_PARALLELPERFORMANCETESTS_MYSIMPLECELL_H
#define LIBGEODECOMP_TESTBED_PARALLELPERFORMANCETESTS_MYSIMPLECELL_H

#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

/**
 * Test class used for avaluation of communication performance.
 */
class MySimpleCell
{
public:
    class API :
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasPredefinedMPIDataType<double>
    {};

    double temp;
};

/**
 * Variant of MySimpleCell, but with a registered Struct of Arrays
 * (SoA) memory layout.
 */
class MySimpleCellSoA : public MySimpleCell
{
public:
    class API :
        public MySimpleCell::API,
        public APITraits::HasSoA
    {};
};

}

LIBFLATARRAY_REGISTER_SOA(
    LibGeoDecomp::MySimpleCellSoA,
    ((double)(temp)))

#endif
