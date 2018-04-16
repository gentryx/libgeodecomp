#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#include <libflatarray/cuda_array.hpp>
#endif

#include <libgeodecomp/misc/testcell.h>
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
        LibFlatArray::cuda_array<TestCell<2> > deviceCellVec(40);
        std::vector<TestCell<2> > hostCellVec(40);

        std::vector<float> hostBuffer(40, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.54321;
        }
        deviceCellVec.load(&hostCellVec[0]);

        DefaultCUDAFilter<TestCell<2>, float, float> filter;

        filter.copyMemberOutImpl(
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            hostBuffer.data(),
            MemoryLocation::HOST,
            40,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.54321), hostBuffer[i]);
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
        LibFlatArray::cuda_array<TestCell<2> > deviceCellVec(1903);
        std::vector<TestCell<2> > hostCellVec(1903);

        LibFlatArray::cuda_array<float> deviceBuffer(1903, -1);
        std::vector<float> hostBuffer(1903, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.0513;
        }
        deviceCellVec.load(&hostCellVec[0]);

        DefaultCUDAFilter<TestCell<2>, float, float> filter;

        filter.copyMemberOutImpl(
            deviceCellVec.data(),
            MemoryLocation::CUDA_DEVICE,
            deviceBuffer.data(),
            MemoryLocation::CUDA_DEVICE,
            1903,
            &TestCell<2>::testValue);

        deviceBuffer.save(&hostBuffer[0]);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.0513), hostBuffer[i]);
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

        std::vector<float> hostBuffer(69, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 0.666;
        }

        DefaultCUDAFilter<TestCell<2>, float, float> filter;

        filter.copyMemberOutImpl(
            hostCellVec.data(),
            MemoryLocation::HOST,
            hostBuffer.data(),
            MemoryLocation::HOST,
            69,
            &TestCell<2>::testValue);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.666), hostBuffer[i]);
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

        LibFlatArray::cuda_array<float> deviceBuffer(4711, -1);
        std::vector<float> hostBuffer(4711, -1);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].testValue = i + 1799;
        }

        DefaultCUDAFilter<TestCell<2>, float, float> filter;

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
        // TEST 1: Copy Out (Device to Device)
        LibFlatArray::cuda_array<float> deviceMemberVec(66, -1);
        std::vector<float> hostMemberVec(66, -2);
        LibFlatArray::cuda_array<float> deviceBuffer(66, -3);
        std::vector<float> hostBuffer(66, -4);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            hostMemberVec[i] = i + 0.661;
        }
        deviceMemberVec.load(&hostMemberVec[0]);

        FilterBase<TestCell<2> > *filter = new DefaultCUDAFilter<TestCell<2>, float, float>();
        filter->copyStreakOut(
            reinterpret_cast<char*>(deviceMemberVec.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(deviceBuffer.data()),
            MemoryLocation::CUDA_DEVICE,
            66,
            66);
        deviceBuffer.save(&hostBuffer[0]);

        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.661), hostBuffer[i]);
        }

        // TEST 2: Copy In (Device to Device)
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 0.662 + i;
        }
        deviceBuffer.load(&hostBuffer[0]);


        filter->copyStreakIn(
            reinterpret_cast<char*>(deviceBuffer.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(deviceMemberVec.data()),
            MemoryLocation::CUDA_DEVICE,
            66,
            66);

        deviceMemberVec.save(&hostMemberVec[0]);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(0.662 + i), hostMemberVec[i]);
        }
#endif
    }

    void testCudaSoAWithGridOnDeviceAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Device to Host)
        LibFlatArray::cuda_array<float> deviceMemberVec(77, -1);
        std::vector<float> hostMemberVec(77, -2);
        std::vector<float> hostBuffer(77, -4);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            hostMemberVec[i] = i + 0.771;
        }
        deviceMemberVec.load(&hostMemberVec[0]);

        FilterBase<TestCell<2> > *filter = new DefaultCUDAFilter<TestCell<2>, float, float>();
        filter->copyStreakOut(
            reinterpret_cast<char*>(deviceMemberVec.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            77,
            77);

        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.771), hostBuffer[i]);
        }

        // TEST 2: Copy In (Host to Device)
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 0.772 + i;
        }

        filter->copyStreakIn(
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(deviceMemberVec.data()),
            MemoryLocation::CUDA_DEVICE,
            77,
            77);

        deviceMemberVec.save(&hostMemberVec[0]);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(0.772 + i), hostMemberVec[i]);
        }
#endif
    }

    void testCudaSoAWithGridOnHostAndBuffersOnDevice()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Host to Device)
        std::vector<float> hostMemberVec(99, -2);
        LibFlatArray::cuda_array<float> deviceBuffer(99, -3);
        std::vector<float> hostBuffer(99, -4);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            hostMemberVec[i] = i + 0.991;
        }

        FilterBase<TestCell<2> > *filter = new DefaultCUDAFilter<TestCell<2>, float, float>();
        filter->copyStreakOut(
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(deviceBuffer.data()),
            MemoryLocation::CUDA_DEVICE,
            99,
            99);
        deviceBuffer.save(&hostBuffer[0]);

        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.991), hostBuffer[i]);
        }

        // TEST 2: Copy In (Device to Host)
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 0.992 + i;
        }
        deviceBuffer.load(&hostBuffer[0]);


        filter->copyStreakIn(
            reinterpret_cast<char*>(deviceBuffer.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            99,
            99);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(0.992 + i), hostMemberVec[i]);
        }
#endif
    }

    void testCudaSoAWithGridOnHostAndBuffersOnHost()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // TEST 1: Copy Out (Host to Host)
        std::vector<float> hostMemberVec(55, -1);
        std::vector<float> hostBuffer(55, -2);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            hostMemberVec[i] = i + 0.551;
        }

        FilterBase<TestCell<2> > *filter = new DefaultCUDAFilter<TestCell<2>, float, float>();
        filter->copyStreakOut(
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            55,
            55);

        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            TS_ASSERT_EQUALS(float(i + 0.551), hostBuffer[i]);
        }

        // TEST 2: Copy In (Host to Host)
        for (std::size_t i = 0; i < hostBuffer.size(); ++i) {
            hostBuffer[i] = 0.552 + i;
        }

        filter->copyStreakIn(
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            55,
            55);

        for (std::size_t i = 0; i < hostMemberVec.size(); ++i) {
            TS_ASSERT_EQUALS(float(0.552 + i), hostMemberVec[i]);
        }
#endif
    }

};

}
