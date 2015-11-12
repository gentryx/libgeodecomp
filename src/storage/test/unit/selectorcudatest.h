#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/communication/hpxserialization.h>
#endif

#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/storage/selector.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class YetAnotherDummyCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit YetAnotherDummyCell(const double x = 0, const int y = 0, const float z0 = 0, const float z1 = 0, const float z2 = 0) :
        x(x),
        y(y)
    {
        z[0] = z0;
        z[1] = z1;
        z[2] = z2;
    }

    double x;
    int y;
    float z[3];
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::YetAnotherDummyCell, ((double)(x))((int)(y))((float)(z)(3)) )

namespace LibGeoDecomp {

class SelectorCUDATest : public CxxTest::TestSuite
{
public:
    void testCopyMemberOut()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // it's sufficient to test with one of the two data locations
        // being on the device. At this point we only need to know
        // that the DefaultCUDA(Array)Filter is being instantiated.

        Selector<YetAnotherDummyCell> selectorX(&YetAnotherDummyCell::x, "varX");
        Selector<YetAnotherDummyCell> selectorY(&YetAnotherDummyCell::y, "varY");
        Selector<YetAnotherDummyCell> selectorZ(&YetAnotherDummyCell::z, "varZ");

        std::vector<YetAnotherDummyCell> vec;
        for (int i = 0; i < 64; ++i) {
            vec << YetAnotherDummyCell(
                12.34 + i,
                i,
                i * 1000 + 55,
                i * 1000 + 66,
                i * 1000 + 77);
        }

        std::vector<double> targetX(64 * 1, -1);
        std::vector<int>    targetY(64 * 1, -1);
        std::vector<float>  targetZ(64 * 3, -1);

        CUDAArray<double> deviceTargetX(&targetX[0], 64 * 1);
        CUDAArray<int>    deviceTargetY(&targetY[0], 64 * 1);
        CUDAArray<float>  deviceTargetZ(&targetZ[0], 64 * 3);

        selectorX.copyMemberOut(
            &vec[0], MemoryLocation::HOST, (char*)deviceTargetX.data(), MemoryLocation::CUDA_DEVICE, 64);
        selectorY.copyMemberOut(
            &vec[0], MemoryLocation::HOST, (char*)deviceTargetY.data(), MemoryLocation::CUDA_DEVICE, 64);
        selectorZ.copyMemberOut(
            &vec[0], MemoryLocation::HOST, (char*)deviceTargetZ.data(), MemoryLocation::CUDA_DEVICE, 64);

        deviceTargetX.save(&targetX[0]);
        deviceTargetY.save(&targetY[0]);
        deviceTargetZ.save(&targetZ[0]);

        for (int i = 0; i < 64; ++i) {
            TS_ASSERT_EQUALS(targetX[i], 12.34 + i);
            TS_ASSERT_EQUALS(targetY[i], i);
            TS_ASSERT_EQUALS(targetZ[i * 3 + 0], i * 1000 + 55);
            TS_ASSERT_EQUALS(targetZ[i * 3 + 1], i * 1000 + 66);
            TS_ASSERT_EQUALS(targetZ[i * 3 + 2], i * 1000 + 77);
        }
#endif
    }

    void testCopyMemberIn()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        // it's sufficient to test with one of the two data locations
        // being on the device. At this point we only need to know
        // that the DefaultCUDA(Array)Filter is being instantiated.

        Selector<YetAnotherDummyCell> selectorX(&YetAnotherDummyCell::x, "varX");
        Selector<YetAnotherDummyCell> selectorY(&YetAnotherDummyCell::y, "varY");
        Selector<YetAnotherDummyCell> selectorZ(&YetAnotherDummyCell::z, "varZ");

        std::vector<YetAnotherDummyCell> vec(90);
        std::vector<double> sourceX;
        std::vector<int>    sourceY;
        std::vector<float>  sourceZ;

        for (int i = 0; i < 90; ++i) {
            sourceX << 56.78 + i;
            sourceY << i * 2;
            sourceZ << (i * 1000 + 22)
                    << (i * 1000 + 33)
                    << (i * 1000 + 44);
        }

        CUDAArray<double> deviceSourceX(&sourceX[0], 90);
        CUDAArray<int>    deviceSourceY(&sourceY[0], 90);
        CUDAArray<float>  deviceSourceZ(&sourceZ[0], 90 * 3);

        selectorX.copyMemberIn(
            (char*)deviceSourceX.data(), MemoryLocation::CUDA_DEVICE, &vec[0], MemoryLocation::HOST, 90);
        selectorY.copyMemberIn(
            (char*)deviceSourceY.data(), MemoryLocation::CUDA_DEVICE, &vec[0], MemoryLocation::HOST, 90);
        selectorZ.copyMemberIn(
            (char*)deviceSourceZ.data(), MemoryLocation::CUDA_DEVICE, &vec[0], MemoryLocation::HOST, 90);

        for (int i = 0; i < 90; ++i) {
            TS_ASSERT_EQUALS(vec[i].x, 56.78 + i);
            TS_ASSERT_EQUALS(vec[i].y, i * 2);
            TS_ASSERT_EQUALS(vec[i].z[0], i * 1000 + 22);
            TS_ASSERT_EQUALS(vec[i].z[1], i * 1000 + 33);
            TS_ASSERT_EQUALS(vec[i].z[2], i * 1000 + 44);
        }
#endif
    }

    void testWithTestCell()
    {
        // this is important to ensure that there are no linker woes
        // because of different instantiations of the selector type
        // from NVCC and the host compiler, where earch gets a
        // different filter (DefaultCUDAFilter vs. DefaultFilter).

        Selector<TestCell<2> > selector = MAKE_SELECTOR(TestCell<2>, testValue);
        selector.copyMemberIn(
            0, MemoryLocation::CUDA_DEVICE, 0, MemoryLocation::CUDA_DEVICE, 0);
    }

};

}
