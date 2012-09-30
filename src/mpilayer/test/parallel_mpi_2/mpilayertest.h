#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

using namespace boost::assign;
using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class ParallelMPILayerTest : public CxxTest::TestSuite
{
public:
    void testAllGather1()
    {
        MPILayer layer;
        SuperVector<unsigned> expected;
        for (unsigned i = 0; i < layer.size(); i++) expected.push_back(i);
        SuperVector<unsigned> actual = layer.allGather(layer.rank());
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGather2()
    {
        MPILayer layer;
        SuperVector<unsigned> expected;
        for (unsigned i = 0; i < layer.size(); i++) expected.push_back(i);
        SuperVector<unsigned> actual(layer.size());
        unsigned rank = layer.rank();
        layer.allGather(&rank, &actual[0], 1);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGather3()
    {
        MPILayer layer;
        SuperVector<unsigned> expected;
        for (unsigned i = 0; i < layer.size(); i++) expected.push_back(i);
        SuperVector<unsigned> actual(layer.size());
        layer.allGather(layer.rank(), &actual);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testGather()
    {
        MPILayer layer;
        unsigned root = 0;
        SuperVector<unsigned> expected_root;
        SuperVector<unsigned> expected_slave;
        for (unsigned i = 0; i < layer.size(); i++) expected_root.push_back(i);

        SuperVector<unsigned> actual = layer.gather(layer.rank(), root);
        if (layer.rank() == root) {
            TS_ASSERT_EQUALS(actual, expected_root);
        } else {
            TS_ASSERT_EQUALS(actual, expected_slave);
        }
    }

    void testBroadcast()
    {
        MPILayer layer;
        unsigned root = 0;
        unsigned expected = 42;
        unsigned actual = 23;
        unsigned source;
        if (layer.rank() == root) source = expected;
        else source = 0;
        actual = layer.broadcast(source, root);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testBroadcastVector()
    {
        MPILayer layer;
        unsigned root = 0;
        SuperVector<double> expected;
        expected += 2,4,24;
        SuperVector<double> actual;
        SuperVector<double> source;

        if (layer.rank() == root) {
            source = expected; 
        } else {
            source = SuperVector<double>();
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
        SuperVector<int> actual, expected;
        actual += 4711, 4712;
        expected += 1000, 1001;
        layer.allGather(i, &actual);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAllGatherV1()
    {
        MPILayer layer;
        SuperVector<unsigned> values;
        SuperVector<int> lengths;
        lengths += 3, 5;
        if (layer.rank() == 0) {
            values += 1, 2, 3;
        } else {
            values += 4, 5, 6, 7, 8;
        }
        SuperVector<unsigned> target(8);
        layer.allGatherV(&values[0], lengths, &target);
        SuperVector<unsigned> expected;
        expected += 1, 2, 3, 4, 5, 6, 7, 8;
        TS_ASSERT_EQUALS(expected, target);
    }
    
    void testAllGatherV2()
    {
        MPILayer layer;
        SuperVector<unsigned> values;
        SuperVector<int> lengths;
        lengths += 3, 5;
        if (layer.rank() == 0) {
            values += 1, 2, 3;
        } else {
            values += 4, 5, 6, 7, 8;
        }
        SuperVector<unsigned> target(layer.allGatherV(&values[0], lengths));
        SuperVector<unsigned> expected;
        expected += 1, 2, 3, 4, 5, 6, 7, 8;
        TS_ASSERT_EQUALS(expected, target);
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
};

}
