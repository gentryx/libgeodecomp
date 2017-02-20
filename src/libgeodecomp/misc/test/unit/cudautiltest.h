#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/cudautil.h>
#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDAUtilTest : public CxxTest::TestSuite
{
public:
    void testGenerateLaunchConfig()
    {
        dim3 cudaBlockDim;
        dim3 cudaGridDim;

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(10, 10, 10));
        TS_ASSERT_EQUALS( 32, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  4, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  3, cudaGridDim.y);
        TS_ASSERT_EQUALS( 10, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(1000, 1000, 1000));
        TS_ASSERT_EQUALS( 128, cudaBlockDim.x);
        TS_ASSERT_EQUALS(   4, cudaBlockDim.y);
        TS_ASSERT_EQUALS(   1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(   8, cudaGridDim.x);
        TS_ASSERT_EQUALS( 250, cudaGridDim.y);
        TS_ASSERT_EQUALS(1000, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(100, 100, 1));
        TS_ASSERT_EQUALS(128, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  4, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS( 25, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(2000, 2000, 1));
        TS_ASSERT_EQUALS(128, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  4, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS( 16, cudaGridDim.x);
        TS_ASSERT_EQUALS(500, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(2000, 1, 1));
        TS_ASSERT_EQUALS(512, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  4, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(1, 1, 1));
        TS_ASSERT_EQUALS( 32, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(1, 1, 1));
        TS_ASSERT_EQUALS( 32, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(32, 1, 1));
        TS_ASSERT_EQUALS( 32, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(40, 1, 1));
        TS_ASSERT_EQUALS( 64, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(64, 1, 1));
        TS_ASSERT_EQUALS( 64, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(90, 1, 1));
        TS_ASSERT_EQUALS(128, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(128, 1, 1));
        TS_ASSERT_EQUALS(128, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(150, 1, 1));
        TS_ASSERT_EQUALS(256, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(256, 1, 1));
        TS_ASSERT_EQUALS(256, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(300, 1, 1));
        TS_ASSERT_EQUALS(512, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);

        CUDAUtil::generateLaunchConfig(&cudaGridDim, &cudaBlockDim, Coord<3>(512, 1, 1));
        TS_ASSERT_EQUALS(512, cudaBlockDim.x);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.y);
        TS_ASSERT_EQUALS(  1, cudaBlockDim.z);
        TS_ASSERT_EQUALS(  1, cudaGridDim.x);
        TS_ASSERT_EQUALS(  1, cudaGridDim.y);
        TS_ASSERT_EQUALS(  1, cudaGridDim.z);
        
    }
};

}

