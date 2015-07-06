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

    void testCudaAoSWithGridOnDeviceAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Host to Device)
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


        // TEST 2: Copy In (Device to Host)

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

    void testCudaAoSWithGridOnDeviceAndBuffersOnDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Device to Device)
        CUDAArray<TestCell<2> > deviceCellVec(1903);
        std::vector<TestCell<2> > hostCellVec(1903);

        CUDAArray<double> deviceBuffer(1903, -1);
        std::vector<double> hostBuffer(1903, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.0513;
        }
        deviceCellVec.load(&hostCellVec[0]);

        DefaultCUDAFilter<TestCell<2>, double, double> filter;

        filter.copyMemberOutImpl(
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            deviceBuffer.data(),
            MemoryLocation::CUDA_DEVICE,
            1903,
            &TestCell<2>::testValue);

        deviceBuffer.save(&hostBuffer[0]);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(i + 0.0513, hostBuffer[i]);
        }

        // TEST 2: Copy In (Device to Device)

        // new data
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 333 + i;
        }

        deviceBuffer.load(&hostBuffer[0]);

        filter.copyMemberInImpl(
            deviceBuffer.data(),
            MemoryLocation::CUDA_DEVICE,
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            1903,
            &TestCell<2>::testValue);

        deviceCellVec.save(&hostCellVec[0]);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(333 + i, hostCellVec[i].testValue);
        }

#endif
    }

    void testCudaAoSWithGridOnHostAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Host to Host)
        std::vector<TestCell<2> > hostCellVec(69);

        std::vector<double> hostBuffer(69, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.666;
        }

        DefaultCUDAFilter<TestCell<2>, double, double> filter;

        filter.copyMemberOutImpl(
            hostCellVec.data(),
            MemoryLocation::HOST,
            hostBuffer.data(),
            MemoryLocation::HOST,
            69,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(i + 0.666, hostBuffer[i]);
        }


        // TEST 2: Copy In (Host to Host)

        // new data
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 314 + i;
        }

        filter.copyMemberInImpl(
            hostBuffer.data(),
            MemoryLocation::HOST,
            hostCellVec.data(),
            MemoryLocation::HOST,
            69,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(314 + i, hostCellVec[i].testValue);
        }

#endif
    }

    void testCudaAoSWithGridOnHostAndBuffersOnDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Host to Device)
        std::vector<TestCell<2> > hostCellVec(4711);

        CUDAArray<double> deviceBuffer(4711, -1);
        std::vector<double> hostBuffer(4711, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 1799;
        }

        DefaultCUDAFilter<TestCell<2>, double, double> filter;

        filter.copyMemberOutImpl(
            hostCellVec.data(),
            MemoryLocation::HOST,
            deviceBuffer.data(),
            MemoryLocation::CUDA_DEVICE,
            4711,
            &TestCell<2>::testValue);

        deviceBuffer.save(&hostBuffer[0]);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(i + 1799, hostBuffer[i]);
        }

        // TEST 2: Copy In (Device to Host)

        // new data
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 4444 + i;
        }

        deviceBuffer.load(&hostBuffer[0]);

        filter.copyMemberInImpl(
            deviceBuffer.data(),
            MemoryLocation::CUDA_DEVICE,
            hostCellVec.data(),
            MemoryLocation::HOST,
            4711,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(4444 + i, hostCellVec[i].testValue);
        }
#endif
    }

    void testCudaSoAWithGridOnDeviceAndBuffersOnDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
#endif
    }

    void testCudaSoAWithGridOnDeviceAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
#endif
    }

    void testCudaSoAWithGridOnHostAndBuffersOnDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
#endif
    }

    void testCudaSoAWithGridOnHostAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
#endif
    }

};

}
