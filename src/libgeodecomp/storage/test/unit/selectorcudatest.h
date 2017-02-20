#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#include <libflatarray/cuda_array.hpp>
#endif

#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/communication/hpxserialization.h>
#endif

#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/selector.h>

using namespace LibGeoDecomp;

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

LIBFLATARRAY_REGISTER_SOA(
    YetAnotherDummyCell,
    ((double)(x))
    ((int)(y))
    ((float)(z)(3)) )

class AnotherMemberMember
{
public:
    double a;
    int b[3];
};

class AnotherMember
{
public:
    AnotherMemberMember member;
    float c[5];
    double d;
};

class AnotherTestClassForMultiLevelNesting
{
public:
    AnotherMember member;
    int e;
};

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

        LibFlatArray::cuda_array<double> deviceTargetX(&targetX[0], 64 * 1);
        LibFlatArray::cuda_array<int>    deviceTargetY(&targetY[0], 64 * 1);
        LibFlatArray::cuda_array<float>  deviceTargetZ(&targetZ[0], 64 * 3);

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

        LibFlatArray::cuda_array<double> deviceSourceX(&sourceX[0], 90);
        LibFlatArray::cuda_array<int>    deviceSourceY(&sourceY[0], 90);
        LibFlatArray::cuda_array<float>  deviceSourceZ(&sourceZ[0], 90 * 3);

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

        Selector<TestCell<2> > selector(&TestCell<2>::testValue, "testValue");
        selector.copyMemberIn(
            0, MemoryLocation::CUDA_DEVICE, 0, MemoryLocation::CUDA_DEVICE, 0);
    }

    void testNesting()
    {
        Selector<AnotherTestClassForMultiLevelNesting> selectorE(
            &AnotherTestClassForMultiLevelNesting::e,
            "varE");

        Selector<AnotherTestClassForMultiLevelNesting> selectorD(
            &AnotherTestClassForMultiLevelNesting::member,
            &AnotherMember::d,
            "varD");

        Selector<AnotherTestClassForMultiLevelNesting> selectorC(
            &AnotherTestClassForMultiLevelNesting::member,
            &AnotherMember::c,
            "varC");

        Selector<AnotherTestClassForMultiLevelNesting> selectorB(
            &AnotherTestClassForMultiLevelNesting::member,
            &AnotherMember::member,
            &AnotherMemberMember::b,
            "varB");

        Selector<AnotherTestClassForMultiLevelNesting> selectorA(
            &AnotherTestClassForMultiLevelNesting::member,
            &AnotherMember::member,
            &AnotherMemberMember::a,
            "varA");

        std::size_t size = 123;
        std::vector<AnotherTestClassForMultiLevelNesting> hostVec(size);

        for (std::size_t i = 0; i < size; ++i) {
            hostVec[i].e                  = i +  1000;
            hostVec[i].member.d           = i +  2000;
            hostVec[i].member.c[0]        = i +  3000;
            hostVec[i].member.c[1]        = i +  4000;
            hostVec[i].member.c[2]        = i +  5000;
            hostVec[i].member.c[3]        = i +  6000;
            hostVec[i].member.c[4]        = i +  7000;
            hostVec[i].member.member.b[0] = i +  8000;
            hostVec[i].member.member.b[1] = i +  9000;
            hostVec[i].member.member.b[2] = i + 10000;
            hostVec[i].member.member.a    = i + 11000;
        }

        LibFlatArray::cuda_array<AnotherTestClassForMultiLevelNesting> deviceVec(size);
        deviceVec.load(&hostVec[0]);

        std::vector<double> bufA(1 * size);
        std::vector<int> bufB(3 * size);
        std::vector<float> bufC(5 * size);
        std::vector<double> bufD(1 * size);
        std::vector<int> bufE(1 * size);

        selectorA.copyMemberOut(
            deviceVec.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&bufA[0]),
            MemoryLocation::HOST,
            size);

        selectorB.copyMemberOut(
            deviceVec.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&bufB[0]),
            MemoryLocation::HOST,
            size);

        selectorC.copyMemberOut(
            deviceVec.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&bufC[0]),
            MemoryLocation::HOST,
            size);

        selectorD.copyMemberOut(
            deviceVec.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&bufD[0]),
            MemoryLocation::HOST,
            size);

        selectorE.copyMemberOut(
            deviceVec.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(&bufE[0]),
            MemoryLocation::HOST,
            size);

        for (std::size_t i = 0; i < size; ++i) {
            TS_ASSERT_EQUALS(bufE[i]        , i +  1000);
            TS_ASSERT_EQUALS(bufD[i]        , i +  2000);
            TS_ASSERT_EQUALS(bufC[5 * i + 0], i +  3000);
            TS_ASSERT_EQUALS(bufC[5 * i + 1], i +  4000);
            TS_ASSERT_EQUALS(bufC[5 * i + 2], i +  5000);
            TS_ASSERT_EQUALS(bufC[5 * i + 3], i +  6000);
            TS_ASSERT_EQUALS(bufC[5 * i + 4], i +  7000);
            TS_ASSERT_EQUALS(bufB[3 * i + 0], i +  8000);
            TS_ASSERT_EQUALS(bufB[3 * i + 1], i +  9000);
            TS_ASSERT_EQUALS(bufB[3 * i + 2], i + 10000);
            TS_ASSERT_EQUALS(bufA[i]        , i + 11000);
        }
    }
};

}
