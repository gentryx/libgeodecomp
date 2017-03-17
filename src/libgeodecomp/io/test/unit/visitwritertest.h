#include <cxxtest/TestSuite.h>

#include <libgeodecomp/config.h>
// purely here to avoid having to define namespace LibGeoDecomp here
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/io/visitwriter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class VisItWriterTest : public CxxTest::TestSuite
{
public:

/**
 * fixme:
 * - basic rectilinear 2D grid test
 * - basic rectilinear 3D grid test
 * - ditto for point meshes
 * - check for custom mesh offset/spacing
 */

    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_VISIT
        std::cout << "boojaka\n";
#endif
    }
};

}

