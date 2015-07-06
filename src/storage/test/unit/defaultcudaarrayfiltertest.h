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

class MyDumbestSoACell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit MyDumbestSoACell(
        const int x = 0,
        const double y1 = 0) :
        x(x)
    {
        for (int i = 0; i < 256; ++i) {
            y[i] = y1;
        }
    }

    int x;
    double y[256];
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::MyDumbestSoACell, ((int)(x))((double)(y)(256)) )

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
        std::vector<MyDumbestSoACell> hostCellVec(40);
        std::vector<double> hostBuffer(40 * 256);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            for (std::size_t j = 0; j < 256; ++j) {
                hostCellVec[i].y[j] = i + j / 1000;
            }
        }

        FilterBase<MyDumbestSoACell> *filter =
            new DefaultCUDAArrayFilter<MyDumbestSoACell, double, double, 256>();
        char MyDumbestSoACell::* memberPointer =
            reinterpret_cast<char MyDumbestSoACell::*>(&MyDumbestSoACell::y);

        filter->copyMemberOut(
            &hostCellVec[0],
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            40,
            memberPointer);

        for (std::size_t i = 0; i < hostBuffer.size(); i += 256) {
            for (std::size_t j = 0; j < 256; ++j) {
                TS_ASSERT_EQUALS(i / 256 + (j / 1000), hostBuffer[i + j]);
            }
        }

        for (std::size_t i = 0; i < hostBuffer.size(); i += 256) {
            for (std::size_t j = 0; j < 256; ++j) {
                hostBuffer[i + j] = i * 1000 + j;
            }
        }

        filter->copyMemberIn(
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            &hostCellVec[0],
            MemoryLocation::HOST,
            40,
            memberPointer);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            for (std::size_t j = 0; j < 256; ++j) {
                TS_ASSERT_EQUALS(i * 1000 * 256 + j, hostCellVec[i].y[j]);
            }
        }
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
