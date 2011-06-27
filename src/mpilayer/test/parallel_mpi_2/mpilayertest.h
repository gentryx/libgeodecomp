#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include "../../../misc/supervector.h"
#include "../../../misc/testhelper.h"
#include "../../mpilayer.h"

using namespace boost::assign;
using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class ParallelMPILayerTest : public CxxTest::TestSuite
{
public:
    void testRegionCommunication()
    {
        MPILayer layer;

        int width = 100;
        int height = 200;
        Grid<double> gridOld(Coord<2>(width, height));
        Grid<double> gridNew(Coord<2>(width, height));
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                gridOld[Coord<2>(x, y)] = -5;
                gridNew[Coord<2>(x, y)] = x + y * width;
            }
        }

        // send/recv a 3x3 rectangle around Coord<2>(47, 11)
        SuperVector<double*> addresses(4);
        addresses[0] = &gridOld[Coord<2>(46, 10)];
        addresses[1] = &gridOld[Coord<2>(46, 11)];
        addresses[2] = &gridOld[Coord<2>(48, 11)];
        addresses[3] = &gridOld[Coord<2>(46, 12)];
        
        SuperVector<unsigned> lengths(4);
        lengths[0] = 3;
        lengths[1] = 1;
        lengths[2] = 1;
        lengths[3] = 3;

        MPILayer::MPIRegionPointer r = layer.registerRegion(
            gridOld.baseAddress(), addresses, lengths);

        if (layer.rank() == 0) {
            layer.sendRegion(gridNew.baseAddress(), r, 1);
            layer.waitAll();
        } else {
            layer.recvRegion(gridOld.baseAddress(), r, 0);
            layer.waitAll();
            
            std::map<Coord<2>, int> rect;
            rect[Coord<2>(46, 10)] = 1;
            rect[Coord<2>(47, 10)] = 1;
            rect[Coord<2>(48, 10)] = 1;
            rect[Coord<2>(46, 11)] = 1;
            rect[Coord<2>(48, 11)] = 1;
            rect[Coord<2>(46, 12)] = 1;
            rect[Coord<2>(47, 12)] = 1;
            rect[Coord<2>(48, 12)] = 1;

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    Coord<2> c(x, y);
                    if (rect.count(c)) {
                        TS_ASSERT_EQUALS(gridOld[c], gridNew[c]);
                    } else {
                        TS_ASSERT_EQUALS(gridOld[c], -5);
                    }                        
                }
            }
        }
    }


    void testAlternativeRegionRegistration()
    {
        MPILayer layer;

        int width = 100;
        int height = 200;
        Grid<double> gridOld(Coord<2>(width, height));
        Grid<double> gridNew(Coord<2>(width, height));
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                gridOld[Coord<2>(x, y)] = -5;
                gridNew[Coord<2>(x, y)] = x + y * width;
            }
        }

        // send/recv a 3x3 rectangle around Coord<2>(47, 11)
        SuperVector<double*> addresses(4);
        addresses[0] = &gridOld[Coord<2>(46, 10)];
        addresses[1] = &gridOld[Coord<2>(46, 11)];
        addresses[2] = &gridOld[Coord<2>(48, 11)];
        addresses[3] = &gridOld[Coord<2>(46, 12)];
        
        SuperVector<unsigned> lengths(4);
        lengths[0] = 3;
        lengths[1] = 1;
        lengths[2] = 1;
        lengths[3] = 3;
        
        MPILayer::MPIRegionPointer r = layer.registerRegion(
            gridOld.baseAddress(), addresses.begin(), lengths.begin(), 4);

        if (layer.rank() == 0) {
            layer.sendRegion(gridNew.baseAddress(), r, 1);
            layer.waitAll();
        } else {
            layer.recvRegion(gridOld.baseAddress(), r, 0);
            layer.waitAll();
            
            std::map<Coord<2>, int> rect;
            rect[Coord<2>(46, 10)] = 1;
            rect[Coord<2>(47, 10)] = 1;
            rect[Coord<2>(48, 10)] = 1;
            rect[Coord<2>(46, 11)] = 1;
            rect[Coord<2>(48, 11)] = 1;
            rect[Coord<2>(46, 12)] = 1;
            rect[Coord<2>(47, 12)] = 1;
            rect[Coord<2>(48, 12)] = 1;

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    Coord<2> c(x, y);
                    if (rect.count(c)) {
                        TS_ASSERT_EQUALS(gridOld[c], gridNew[c]);
                    } else {
                        TS_ASSERT_EQUALS(gridOld[c], -5);
                    }                        
                }
            }
        }
    }


    void testYetAnotherRegionRegistration()        
    {
        MPILayer layer;
        TestCell<2> a, b;
        a.testValue = 4.7;
        b.testValue = 1.1;
        Grid<TestCell<2> > sendgrid(Coord<2>(100, 100), a);
        Grid<TestCell<2> > recvgrid(Coord<2>(100, 100), b);
        Region<2> region;
        for (int i = 15; i < 30; ++i)
            region << Streak<2>(Coord<2>(10, i), 20);
        MPILayer::MPIRegionPointer mpiRegion = layer.registerRegion(sendgrid, region);
        SuperSet<Coord<2> > set;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) 
            set.insert(*i);

        for (int y = 0; y < 100; ++y)
            for (int x = 0; x < 100; ++x)
                TS_ASSERT_EQUALS(recvgrid[Coord<2>(x, y)].testValue, 1.1);

        layer.sendRegion(sendgrid.baseAddress(), mpiRegion, layer.rank());
        layer.recvRegion(recvgrid.baseAddress(), mpiRegion, layer.rank());
        layer.waitAll();

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i)
            TS_ASSERT_EQUALS(recvgrid[*i].testValue, 4.7);
        for (int y = 0; y < 100; ++y)
            for (int x = 0; x < 100; ++x)
                if (!set.count(Coord<2>(x, y)))
                    TS_ASSERT_EQUALS(recvgrid[Coord<2>(x, y)].testValue, 1.1);
    }


    void testAllGather()
    {
        MPILayer layer;
        UVec expected;
        for (unsigned i = 0; i < layer.size(); i++) expected.push_back(i);
        UVec actual = layer.allGather(layer.rank());
        TS_ASSERT_EQUALS(actual, expected);
    }


    void testGather()
    {
        MPILayer layer;
        unsigned root = 0;
        UVec expected_root;
        UVec expected_slave;
        for (unsigned i = 0; i < layer.size(); i++) expected_root.push_back(i);

        UVec actual = layer.gather(layer.rank(), root);
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
        DVec expected;
        expected += 2,4,24;
        DVec actual;
        DVec source;
        if (layer.rank() == root) source = expected;
        else source = DVec();
        actual = layer.broadcastVector(source, root);
        TSMA_ASSERT_EQUALS(actual, expected);
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


    void testSendRecvRows()
    {
        MPILayer layer;
        Grid<TestCell<2> > grid(Coord<2>(2, 2));
        grid[Coord<2>(0, 0)] = TestCell<2>(Coord<2>(21, 22), Coord<2>(23, 24));
        grid[Coord<2>(1, 0)] = TestCell<2>(Coord<2>(25, 26), Coord<2>(27, 28));
        grid[Coord<2>(0, 1)] = TestCell<2>(Coord<2>(29, 30), Coord<2>(31, 32));
        grid[Coord<2>(1, 1)] = TestCell<2>(Coord<2>(33, 34), Coord<2>(35, 36));
        
        if (layer.rank() == 0) {
            Grid<TestCell<2> > expected = grid;
            grid[Coord<2>(0, 0)] = 
                TestCell<2>(Coord<2>( 1,  2), Coord<2>( 3,  4));
            grid[Coord<2>(1, 0)] = 
                TestCell<2>(Coord<2>( 5,  6), Coord<2>( 7,  8));
            grid[Coord<2>(0, 1)] = 
                TestCell<2>(Coord<2>( 9, 10), Coord<2>(11, 12));
            grid[Coord<2>(1, 1)] = 
                TestCell<2>(Coord<2>(13, 14), Coord<2>(15, 16));
            layer.recvRows(&grid, 1, 2, 1, 47);
            layer.waitAll();
            TS_ASSERT_EQUALS(grid[1], expected[1]);
        } else {
            layer.sendRows(&grid, 1, 2, 0, 47);
            layer.waitAll();
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
};

};
