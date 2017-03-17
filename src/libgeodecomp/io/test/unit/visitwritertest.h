#include <cxxtest/TestSuite.h>

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/visitwriter.h>

#ifdef LIBGEODECOMP_WITH_VISIT
using namespace LibGeoDecomp;
#endif

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

