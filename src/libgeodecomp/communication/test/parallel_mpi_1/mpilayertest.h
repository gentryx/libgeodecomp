#include <cxxtest/TestSuite.h>
#include <mpi.h>

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPILayerTest : public CxxTest::TestSuite
{
public:

    TestCell<2> demoCell(int testValue = 1)
    {
        TestCell<2> c;
        c.testValue = testValue;
        return c;
    }

    void testRequestHandlingCell()
    {
        MPILayer layer;
        TestCell<2> foo;
        TestCell<2> demo = demoCell();
        TS_ASSERT_EQUALS((unsigned)0, layer.requests[0].size());
        layer.send(&demo, 0);
        TS_ASSERT_EQUALS((unsigned)1, layer.requests[0].size());
        layer.recv(&foo, 0);
        TS_ASSERT_EQUALS((unsigned)2, layer.requests[0].size());
        layer.waitAll();
        TS_ASSERT_EQUALS((unsigned)0, layer.requests[0].size());
    }

    void testSendRecvCell()
    {
        MPILayer layer;
        TestCell<2> sendCell = demoCell(2);
        layer.send(&sendCell, 0);
        TestCell<2> receivedCell;
        layer.recv(&receivedCell, 0);
        layer.waitAll();
        TS_ASSERT_EQUALS(receivedCell, sendCell);
    }

    void testSize()
    {
        MPILayer layer;
        TS_ASSERT_EQUALS(1, layer.size());
    }

    void testRank()
    {
        MPILayer layer;
        TS_ASSERT(layer.rank() < layer.size());
    }

    void testSendRecvVec()
    {
        MPILayer layer;
        std::vector<unsigned> actual(5);
        std::vector<unsigned> expected(5);
        expected[0] = 1;
        expected[1] = 2;
        expected[2] = 3;
        expected[3] = 5;
        expected[4] = 8;

        layer.sendVec(&actual, 0);
        layer.recvVec(&expected, 0);
        layer.waitAll();
        TS_ASSERT_EQUALS(expected, actual);
    }
};

}
