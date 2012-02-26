#include <cxxtest/TestSuite.h>
#include <mpi.h>

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

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

    void testSendRecvGridBox()
    {
        MPILayer layer;
        double edgeCell = -1;

        Grid<double> source(Coord<2>(6, 4));
        Grid<double> target(Coord<2>(7, 8));
        Grid<double> expectedTarget(Coord<2>(7, 8));
        int rectWidth = 3;
        int rectHeight= 2;
        CoordBox<2> sourceRect(Coord<2>(1, 1), 
                               Coord<2>(rectWidth, rectHeight));
        CoordBox<2> targetRect(Coord<2>(2, 4), 
                               Coord<2>(rectWidth, rectHeight));

        // setting up temperatures so that communication integrity can be
        // verified
        fillGrid(source, 100.0);
        fillGrid(target, 200.0);
        fillGrid(expectedTarget, 200.0);
        fillGridRect(source, sourceRect, 300.0);
        fillGridRect(expectedTarget, targetRect, 300.0);
        source.getEdgeCell() = edgeCell;
        target.getEdgeCell() = edgeCell;
        expectedTarget.getEdgeCell() = edgeCell;

        TS_ASSERT_DIFFERS(expectedTarget, target);
        layer.sendGridBox(&source, sourceRect, 0);
        layer.recvGridBox(&target, targetRect, 0);
        layer.waitAll();
        TSM_ASSERT_EQUALS(
            expectedTarget.diff(target).c_str(), expectedTarget, target);
    }

    void testSize() 
    { 
        MPILayer layer; 
        TS_ASSERT_EQUALS((unsigned)1, layer.size());
    } 

    void testRank() 
    { 
        MPILayer layer; 
        TS_ASSERT(layer.rank() < layer.size()); 
    } 

    void testSendRecvVec()
    {
        MPILayer layer; 
        UVec actual(5);
        UVec expected(5);
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

    void testRegionCommunication()
    {
        MPILayer layer; 
        int n = 1024;
        double fieldSend[n];
        double fieldRecv[n];
        
        for (int i = 0; i < n; i++) {
            fieldSend[i] = i + 1.23;
            fieldRecv[i] = -1;            
        }

        int chunks = 5;
        SuperVector<double*> addresses(chunks);
        addresses[0] = &fieldSend[10];
        addresses[1] = &fieldSend[20];
        addresses[2] = &fieldSend[85];
        addresses[3] = &fieldSend[55];
        addresses[4] = &fieldSend[40];

        SuperVector<unsigned> lengths(chunks);
        lengths[0] = 5;
        lengths[1] = 2;
        lengths[2] = 3;
        lengths[3] = 7;
        lengths[4] = 5;
            
        MPILayer::MPIRegionPointer region = layer.registerRegion(
            fieldSend,
            addresses,
            lengths);

        layer.sendRegion(fieldSend, region, 0);
        layer.recvRegion(fieldRecv, region, 0);
        layer.waitAll();

        for (int c = 0; c < chunks; c++) {
            for (unsigned i = 0; i < lengths[c]; i++) {
                TS_ASSERT_EQUALS(addresses[c][i], 
                                 (addresses[c] - fieldSend + fieldRecv)[i]);
            }
        }
    }

    void testCreateRegionForGrid()
    {
        MPILayer layer;

        double edgeCell = -1;
        Coord<2> base(0, 0);
        Grid<double> source(Coord<2>(6, 4));
        Grid<double> target(Coord<2>(7, 8));
        Grid<double> expectedTarget(Coord<2>(7, 8));
        int rectWidth = 3;
        int rectHeight= 2;
        CoordBox<2> sourceRect(Coord<2>(1, 1), 
                               Coord<2>(rectWidth, rectHeight));
        CoordBox<2> targetRect(Coord<2>(2, 4), 
                               Coord<2>(rectWidth, rectHeight));

        // setting up temperatures so that communication integrity can
        // be verified
        fillGrid(source, 100.0);
        fillGrid(target, 200.0);
        fillGrid(expectedTarget, 200.0);
        fillGridRect(source, sourceRect, 300.0);
        fillGridRect(expectedTarget, targetRect, 300.0);
        source.getEdgeCell() = edgeCell;
        target.getEdgeCell() = edgeCell;
        expectedTarget.getEdgeCell() = edgeCell;

        TS_ASSERT_DIFFERS(expectedTarget, target);
        MPILayer::MPIRegionPointer sendRP = 
            layer.registerRegion(source, sourceRect, base);

        MPILayer::MPIRegionPointer recvRP = 
            layer.registerRegion(target, targetRect, base);

        layer.sendRegion(&(source[base]), sendRP, 0);
        layer.recvRegion(&(target[base]), recvRP, 0);
        layer.waitAll();
        TSM_ASSERT_EQUALS(
            expectedTarget.diff(target).c_str(), 
            expectedTarget, target);
    }
    
private:
    template<typename T>
    void fillGridRect(
            Grid<T>& grid, const CoordBox<2>& rect, const T& val)
    {
        for (CoordBoxSequence<2> s = rect.sequence(); s.hasNext();) 
            grid[s.next()] = val;
    }

    template<typename T>
    void fillGrid(Grid<T>& grid, const T& val)
    {
        fillGridRect(grid, grid.boundingBox(), val);
    }

    Grid<TestCell<2> > mangledGrid(double foo)
    {
        Grid<TestCell<2> > grid(Coord<2>(27, 81));
        for (unsigned y = 0; y < grid.getDimensions().y(); y++) 
            for (unsigned x = 0; x < grid.getDimensions().x(); x++) 
                grid[y][x].testValue = 
                    foo + y * grid.getDimensions().y() + x;
        return grid;
    }    
};

}
