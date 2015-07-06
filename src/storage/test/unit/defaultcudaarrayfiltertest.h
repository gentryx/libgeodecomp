#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/storage/defaultcudaarrayfilter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DefaultCUDAArrayFilterTest : public CxxTest::TestSuite
{
public:
    void testCudaAoSWithGridOnDeviceAndBuffersOnDevice()
    {
        // fixme
    }
    void testCudaAoSWithGridOnDeviceAndBuffersOnHost()
    {
        // fixme
    }

    void testCudaAoSWithGridOnHostAndBuffersOnDevice()
    {
        // fixme
    }
    void testCudaAoSWithGridOnHostAndBuffersOnHost()
    {
        // fixme
    }

    void testCudaSoAWithGridOnDeviceAndBuffersOnDevice()
    {
        // fixme
    }
    void testCudaSoAWithGridOnDeviceAndBuffersOnHost()
    {
        // fixme
    }

    void testCudaSoAWithGridOnHostAndBuffersOnDevice()
    {
        // fixme
    }
    void testCudaSoAWithGridOnHostAndBuffersOnHost()
    {
        // fixme
    }
};

}
