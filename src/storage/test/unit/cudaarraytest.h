#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/misc/cudautil.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDAArrayTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        std::vector<double> hostVec1(30, -1);
        std::vector<double> hostVec2(30, -2);
        std::vector<double> hostVec3(30, -3);
        std::vector<double> hostVec4(30, -4);

        for (int i = 0; i < 30; ++i) {
            hostVec1[i] = i + 0.5;
            TS_ASSERT_EQUALS(-2, hostVec2[i]);
            TS_ASSERT_EQUALS(-3, hostVec3[i]);
            TS_ASSERT_EQUALS(-4, hostVec4[i]);
        }

        CUDAArray<double> deviceArray1(&hostVec1[0], 30);
        CUDAArray<double> deviceArray2(hostVec1);
        CUDAArray<double> deviceArray3(deviceArray1);
        CUDAArray<double> deviceArray4;
        deviceArray4 = CUDAArray<double>(30);
        deviceArray4.load(&hostVec1[0]);

        deviceArray2.save(&hostVec2[0]);
        deviceArray3.save(&hostVec3[0]);
        deviceArray4.save(&hostVec4[0]);

        for (int i = 0; i < 30; ++i) {
            double expected = i + 0.5;
            TS_ASSERT_EQUALS(expected, hostVec2[i]);
            TS_ASSERT_EQUALS(expected, hostVec3[i]);
            TS_ASSERT_EQUALS(expected, hostVec4[i]);
        }

        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray2.data());
        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray3.data());
        TS_ASSERT_DIFFERS(deviceArray1.data(), deviceArray3.data());

        TS_ASSERT_DIFFERS(deviceArray2.data(), deviceArray3.data());
        TS_ASSERT_DIFFERS(deviceArray2.data(), deviceArray4.data());

        TS_ASSERT_DIFFERS(deviceArray3.data(), deviceArray4.data());

        cudaDeviceSynchronize();
        CUDAUtil::checkForError();
#endif
    }

};

}
