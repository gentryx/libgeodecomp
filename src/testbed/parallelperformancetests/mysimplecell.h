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

}

#endif
