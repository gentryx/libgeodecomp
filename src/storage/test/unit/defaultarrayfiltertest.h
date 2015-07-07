#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/defaultarrayfilter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyDumbSoACell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit MyDumbSoACell(
        const int x = 0,
        const double y1 = 0,
        const double y2 = 0,
        const double y3 = 0) :
        x(x)
    {
        y[0] = y1;
        y[1] = y2;
        y[2] = y3;
    }

    int x;
    double y[3];
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::MyDumbSoACell, ((int)(x))((double)(y)(3)) )

namespace LibGeoDecomp {

class DefaultArrayFilterTest : public CxxTest::TestSuite
{
public:
    void testHostAoS()
    {
        std::vector<MyDumbSoACell> hostCellVec(40);
        std::vector<double> hostBuffer(40 * 3);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            hostCellVec[i].y[0] = i + 0.5;
            hostCellVec[i].y[1] = i + 0.6;
            hostCellVec[i].y[2] = i + 0.7;
        }

        FilterBase<MyDumbSoACell> *filter = new DefaultArrayFilter<MyDumbSoACell, double, double, 3>();
        char MyDumbSoACell::* memberPointer = reinterpret_cast<char MyDumbSoACell::*>(&MyDumbSoACell::y);

        filter->copyMemberOut(
            &hostCellVec[0],
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            40,
            memberPointer);

        for (std::size_t i = 0; i < hostBuffer.size(); i += 3) {
            TS_ASSERT_EQUALS(i / 3 + 0.5, hostBuffer[i + 0]);
            TS_ASSERT_EQUALS(i / 3 + 0.6, hostBuffer[i + 1]);
            TS_ASSERT_EQUALS(i / 3 + 0.7, hostBuffer[i + 2]);
        }

        for (std::size_t i = 0; i < hostBuffer.size(); i += 3) {
            hostBuffer[i + 0] = i * 2 + 1000.0;
            hostBuffer[i + 1] = i * 2 + 1001.0;
            hostBuffer[i + 2] = i * 2 + 1002.0;
        }

        filter->copyMemberIn(
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            &hostCellVec[0],
            MemoryLocation::HOST,
            40,
            memberPointer);

        for (std::size_t i = 0; i < hostCellVec.size(); ++i) {
            TS_ASSERT_EQUALS(i * 6 + 1000.0, hostCellVec[i].y[0]);
            TS_ASSERT_EQUALS(i * 6 + 1001.0, hostCellVec[i].y[1]);
            TS_ASSERT_EQUALS(i * 6 + 1002.0, hostCellVec[i].y[2]);
        }
    }

    void testHostSoA()
    {
        // SoA format in grid (i.e. hostMemberVec) mandates that the
        // elements of a member array are split up according to a
        // certain stride (in this case 40). In an external buffer
        // (hostBuffer) these array elements will be aggregated and
        // stored directly one after another:
        std::vector<double> hostMemberVec(120);
        std::vector<double> hostBuffer(120, -1);

        for (std::size_t i = 0; i < 40; ++i) {
            hostMemberVec[i + 0 * 40] = i + 0.71;
            hostMemberVec[i + 1 * 40] = i + 0.72;
            hostMemberVec[i + 2 * 40] = i + 0.73;
        }

        FilterBase<MyDumbSoACell > *filter = new DefaultArrayFilter<MyDumbSoACell, double, double, 3>();

        filter->copyStreakOut(
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            40,
            40);

        for (std::size_t i = 0; i < hostBuffer.size(); i += 3) {
            TS_ASSERT_EQUALS(i / 3 + 0.71, hostBuffer[i + 0]);
            TS_ASSERT_EQUALS(i / 3 + 0.72, hostBuffer[i + 1]);
            TS_ASSERT_EQUALS(i / 3 + 0.73, hostBuffer[i + 2]);
        }

        for (std::size_t i = 0; i < hostBuffer.size(); i += 3) {
            hostBuffer[i + 0] = i / 3 + 0.1;
            hostBuffer[i + 1] = i / 3 + 0.2;
            hostBuffer[i + 2] = i / 3 + 0.3;
        }

        filter->copyStreakIn(
            reinterpret_cast<char*>(&hostBuffer[0]),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&hostMemberVec[0]),
            MemoryLocation::HOST,
            40,
            40);

        for (std::size_t i = 0; i < 40; ++i) {
            TS_ASSERT_EQUALS(i + 0.1, hostMemberVec[i +  0]);
            TS_ASSERT_EQUALS(i + 0.2, hostMemberVec[i + 40]);
            TS_ASSERT_EQUALS(i + 0.3, hostMemberVec[i + 80]);
        }
    }
};

}
