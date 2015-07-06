#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/storage/defaultcudafilter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DefaultCUDAFilterTest : public CxxTest::TestSuite
{
public:

    void testCudaAoSFromHostToDeviceAndFromDeviceToHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: From Host to Device:
        CUDAArray<TestCell<2> > deviceCellVec(40);
        std::vector<TestCell<2> > hostCellVec(40);

        std::vector<double> hostBuffer(40, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.54321;
        }
        deviceCellVec.load(&hostCellVec[0]);

        DefaultCUDAFilter<TestCell<2>, double, double> filter;

        filter.copyMemberOutImpl(
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            hostBuffer.data(),
            MemoryLocation::HOST,
            40,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(i + 0.54321, hostBuffer[i]);
        }


        // TEST 2: From Device to Host:

        // new data
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 666 + i;
        }

        filter.copyMemberInImpl(
            hostBuffer.data(),
            MemoryLocation::HOST,
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            40,
            &TestCell<2>::testValue);

        deviceCellVec.save(&hostCellVec[0]);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(666 + i, hostCellVec[i].testValue);
        }

#endif
    }

    void testCudaAoSFromHostToHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // fixme
#endif
    }

    void testCudaAoSFromDeviceToDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // fixme
#endif
    }

};

}
