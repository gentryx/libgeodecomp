#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/storage/cudasoagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDASoAGridTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::cout << "========================================================================================\n";
    }

};

}
