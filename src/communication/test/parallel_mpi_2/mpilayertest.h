#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/testhelper.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ParallelMPILayerTest : public CxxTest::TestSuite
{
public:
    void testAllGather1()
    {
        MPILayer layer;
        std::vector<int> expected;
        for (int i = 0; i < layer.size(); i++) {
            expected.push_back(i);
        }
        std::vector<int> actual = layer.allGather(layer.rank());
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGather2()
    {
        MPILayer layer;
        std::vector<int> expected;
        for (int i = 0; i < layer.size(); i++) {
            expected.push_back(i);
        }
        std::vector<int> actual(layer.size());
        int rank = layer.rank();
        layer.allGather(&rank, &actual[0], 1);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGather3()
    {
        MPILayer layer;
        std::vector<int> expected;
        for (int i = 0; i < layer.size(); i++) {
            expected.push_back(i);
        }
        std::vector<int> actual(layer.size());
        layer.allGather(layer.rank(), &actual);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testGather()
    {
        MPILayer layer;
        int root = 0;
        std::vector<int> expected_root;
        std::vector<int> expected_slave;
        for (int i = 0; i < layer.size(); i++) {
            expected_root.push_back(i);
        }

        std::vector<int> actual = layer.gather(layer.rank(), root);
        if (layer.rank() == root) {
            TS_ASSERT_EQUALS(actual, expected_root);
        } else {
            TS_ASSERT_EQUALS(actual, expected_slave);
        }
    }

    void testBroadcast()
    {
        MPILayer layer;
        int root = 0;
        unsigned expected = 42;
        unsigned actual = 23;
        unsigned source = (layer.rank() == root)? expected : 0;

        actual = layer.broadcast(source, root);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testBroadcast2()
    {
        MPILayer layer;
        int root = 0;
        std::string expected = "hello world";
        std::string buffer = (layer.rank() == root)? expected : std::string(expected.size(), 'X');

        layer.broadcast(&buffer[0], buffer.size(), root);
        TS_ASSERT_EQUALS(expected, buffer);
    }

    void testBroadcastVector()
    {
        MPILayer layer;
        int root = 0;
        std::vector<double> expected;
        expected += 2,4,24;
        std::vector<double> actual;
        std::vector<double> source;

        if (layer.rank() == root) {
            source = expected;
        } else {
            source = std::vector<double>();
        }

        actual = layer.broadcastVector(source, root);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testSendRecvCell()
    {
        MPILayer layer;
        if (layer.rank() == 0) {
            TestCell<2> a(Coord<2>(1, 2), Coord<2>(3, 4), 5);
            Coord<2> b(6, 7);
            layer.send(&a, 1);
            layer.send(&b, 1);
            layer.waitAll();
        } else {
            TestCell<2> a;
            Coord<2> b(8,9);
            layer.recv(&a, 0);
            layer.recv(&b, 0);
            layer.waitAll();

            // For some weird reason TS_ASSERT_EQUALS with these two
            // parameters fails on the Cell blades...
            TS_ASSERT(a == TestCell<2>(Coord<2>(1, 2), Coord<2>(3, 4), 5));
            TS_ASSERT_EQUALS(b, Coord<2>(6, 7));
        }
    }

    void testSendRecvFloatCoord()
    {
        MPILayer layer;
        FloatCoord<3> c;

        if (layer.rank() == 0) {
            c = FloatCoord<3>(0.5 + layer.rank(), 2.0, 3.0);
            layer.send(&c, 1);
        } else {
            layer.recv(&c, 0);
        }

        layer.waitAll();
        TS_ASSERT_EQUALS(c, FloatCoord<3>(0.5, 2.0, 3.0));
    }

    void testSendRecvRegion()
    {
        MPILayer layer;
        Region<2> a;
        a << Streak<2>(Coord<2>(10, 20), 30);
        a << Streak<2>(Coord<2>(11, 21), 31);
        a << Streak<2>(Coord<2>(12, 22), 32);

        if (layer.rank() == 0) {
            layer.sendRegion(a, 1);
        } else {
            Region<2> b;
            layer.recvRegion(&b, 0);
            TS_ASSERT_EQUALS(a, b);
        }
    }

    void testAllGatherAgain()
    {
        MPILayer layer;
        int i = layer.rank() + 1000;
        std::vector<int> actual, expected;
        actual += 4711, 4712;
        expected += 1000, 1001;
        layer.allGather(i, &actual);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGatherV1()
    {
        MPILayer layer;
        std::vector<unsigned> values;
        std::vector<int> lengths;
        lengths += 3, 5;
        if (layer.rank() == 0) {
            values += 1, 2, 3;
        } else {
            values += 4, 5, 6, 7, 8;
        }
        std::vector<unsigned> target(8);
        layer.allGatherV(&values[0], lengths, &target);
        std::vector<unsigned> expected;
        expected += 1, 2, 3, 4, 5, 6, 7, 8;
        TS_ASSERT_EQUALS(expected, target);
    }

    void testAllGatherV2()
    {
        MPILayer layer;
        std::vector<unsigned> values;
        std::vector<int> lengths;
        lengths += 3, 5;
        if (layer.rank() == 0) {
            values += 1, 2, 3;
        } else {
            values += 4, 5, 6, 7, 8;
        }
        std::vector<unsigned> target(layer.allGatherV(&values[0], lengths));
        std::vector<unsigned> expected;
        expected += 1, 2, 3, 4, 5, 6, 7, 8;
        TS_ASSERT_EQUALS(expected, target);
    }

    void testGatherV()
    {
        MPILayer layer;
        int data[] = {1, 2, 3, 4};
        if (layer.rank() == 1) {
            data[0] = 47;
            data[1] = 11;
        }

        std::vector<int> target(5);
        std::vector<int> lengths;
        lengths << 3
                << 2;
        layer.gatherV(data, lengths[layer.rank()], lengths, 0, &target[0]);

        if (layer.rank() == 0) {
            std::vector<int> expected;
            expected << 1
                     << 2
                     << 3
                     << 47
                     << 11;
            TS_ASSERT_EQUALS(expected, target);
        }
    }

    void testCancel()
    {
        if (MPILayer().rank() == 0) {
            MPILayer layer;
            int i = 0;
            layer.recv(&i, 1, 1);
            layer.cancelAll();
        }
    }

    void testWait()
    {
        MPILayer layer;
        TS_ASSERT_EQUALS(layer.wait(4711), 0);
        TS_ASSERT(!layer.wait(4711));

        int otherRank = 1 - layer.rank();

        int sourceData = layer.rank();
        int targetData = -1;

        layer.send(&sourceData, otherRank, 1, 4711);
        layer.recv(&targetData, otherRank, 1, 4711);

        TS_ASSERT_EQUALS(layer.wait(4711), 2);
        TS_ASSERT_EQUALS(targetData, otherRank);
    }
};

}
