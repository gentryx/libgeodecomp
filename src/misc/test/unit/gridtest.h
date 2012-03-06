#include <fstream>
#include <sstream>
#include <cstdio>
#include <cxxtest/TestSuite.h>

#include "../../grid.h"
#include "../../displacedgrid.h"
#include "../../testcell.h"

#define GRIDWIDTH 4
#define GRIDHEIGHT 5

double edge = 0;
     
using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

/*
 * testGrid is
 *     x=0         x=3
 * y=0 200 201 202 203
 *     204 205 206 207
 *     208 209 210 211
 *     212 213 214 215
 * y=4 216 217 218 219
 */
class GridTest : public CxxTest::TestSuite 
{
    Grid<TestCell<2> > *testGrid;

public:

    void setUp()
    {
        testGrid = new Grid<TestCell<2> >(Coord<2>(GRIDWIDTH, GRIDHEIGHT));

        int num = 200;
        for (unsigned y = 0; y < testGrid->getDimensions().y(); y++) {
            for (unsigned x = 0; x < testGrid->getDimensions().x(); x++) {
                (*testGrid)[Coord<2>(x, y)] = 
                    TestCell<2>(Coord<2>(x, y), testGrid->getDimensions());
                (*testGrid)[Coord<2>(x, y)].testValue =  num++;
            }
        }
    }

    void tearDown()
    {
        delete testGrid;
    }

    void testDefaultConstructor()
    {
        Grid<TestCell<2> > g;
        TS_ASSERT_EQUALS(0, (int)g.getDimensions().x());
        TS_ASSERT_EQUALS(0, (int)g.getDimensions().y());
    }

    void testConstructorDefaultInit()
    {
        Grid<double> g(Coord<2>(10, 12), 14.16, 47.11);
        TS_ASSERT_EQUALS(Coord<2>(10, 12), g.getDimensions());
        TS_ASSERT_EQUALS(14.16, g[5][6]);
        TS_ASSERT_EQUALS(47.11, g[Coord<2>(-1, -1)]);
    }

    void testMultiDimensionalConstructor()
    {
        Coord<3> dim(3, 4, 5);
        Grid<double, Topologies::Torus<3>::Topology > g(dim);
        g.getEdgeCell() = -1;
        g[Coord<3>(2, 1, 1)] = 1;
        g[Coord<3>(-1, 27, -2)] = 5;

        TS_ASSERT_EQUALS(dim, g.getDimensions());
        TS_ASSERT_EQUALS(-1, g.getEdgeCell());
        TS_ASSERT_EQUALS(1, g[Coord<3>(-1, 1, 1)]);
        TS_ASSERT_EQUALS(5, g[Coord<3>(2, 3, 3)]);
    }

    void testOperatorEqual1()
    {
        int width = testGrid->getDimensions().x();
        int height = testGrid->getDimensions().y();
        double deltaTemp = 42.0;
        Coord<2> changedCoord(0, 4);
        
        Grid<TestCell<2> > other(Coord<2>(width, height));
        TS_ASSERT(!other[4][2].isValid);
        for(int x = 0; x < width; x++)
            for(int y = 0; y < height; y++) 
                other.cellMatrix[y][x] = testGrid->cellMatrix[y][x];
        
        TS_ASSERT(*testGrid == other);

        other[changedCoord].testValue += deltaTemp;
        TS_ASSERT_EQUALS(
            other[changedCoord].testValue,
            (*testGrid)[changedCoord].testValue + deltaTemp);
        TS_ASSERT(*testGrid != other);
    }

    void testOperatorEqual2()
    {
        Grid<TestCell<2> > a(Coord<2>(0, 0));
        Grid<TestCell<2> > b(Coord<2>(0, 0));
        TS_ASSERT_EQUALS(a, b);
    }
    
    void testDiffDimensions()
    {
        int width = testGrid->getDimensions().x();
        int height = testGrid->getDimensions().y();

        TS_ASSERT_EQUALS(testGrid->diff(Grid<TestCell<2> >(*testGrid)), "");

        int newWidth = width + 5;
        int newHeight = 11;
        std::ostringstream message;
        message << "dimensions mismatch (is (" 
                << CoordBox<2>(Coord<2>(),
                               Coord<2>(width, height))
                << "), got (" 
                << CoordBox<2>(Coord<2>(),
                               Coord<2>(newWidth, newHeight))
                << "))";
        
        TS_ASSERT_EQUALS(testGrid->diff(
                             Grid<TestCell<2> >(Coord<2>(newWidth, newHeight))), 
                         message.str());
    }

    void testDiffCells()
    {
        Grid<TestCell<2> > other(*testGrid);
        Coord<2> changedCoord1(1, 2);
        Coord<2> changedCoord2(0, 4);
        other[changedCoord1].testValue += 1;
        other[changedCoord2].testValue -= 1;

        std::ostringstream message;
        message << "cell differences:\n\n" << 
            "at coordinate " << changedCoord1 << "\n" << 
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" <<
            (*testGrid)[changedCoord1] <<
            "========================================\n" <<
            other[changedCoord1] <<
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n";
        message << 
            "at coordinate " << changedCoord2 << "\n" << 
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" <<
            (*testGrid)[changedCoord2] <<
            "========================================\n" <<
            other[changedCoord2] <<
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";

        TS_ASSERT_EQUALS(testGrid->diff(other), message.str());
    }

    void testDiffEdgeCell()
    {
        Grid<int> muh(Coord<2>(10, 10));
        Grid<int> kuh(muh);
        muh[Coord<2>(-1, -1)] = 47;
        kuh[Coord<2>(-1, -1)] = 11;

        std::string message =  "cell differences:\n\nedge cell differs (self (47), other (11))\n";
        TS_ASSERT_EQUALS(message, muh.diff(kuh));
    }

    void testCopyConstructor()
    {
        Grid<int> *a1 = new Grid<int>(Coord<2>(200, 100));
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 200; j++) {
                (*a1)[i][j] = i * 200 + j;
            }
        Grid<int> *a2 = new Grid<int>;
        *a2 = *a1;
        TS_ASSERT_EQUALS(*a1, *a1);
        (*a2)[Coord<2>(0, 0)] = 4711;
        TS_ASSERT_DIFFERS(*a1, *a2);
    }

    void testOperatorSquareBracketsOfCoord()
    {
        (*testGrid)[Coord<2>(2, 3)].testValue = 47;
        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(2, 3)].testValue, 47);
        (*testGrid)[Coord<2>(2, 3)].testValue = 11;
        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(2, 3)].testValue, 11);
    }

    void testResize()
    {
        Grid<int> g(Coord<2>(64, 20));
        TS_ASSERT_EQUALS(Coord<2>(64, 20), g.getDimensions());
        g.resize(Coord<2>(12, 34));
        TS_ASSERT_EQUALS(Coord<2>(12, 34), g.getDimensions());
    }

    void testGetNeighborhood() 
    {
        CoordMap<TestCell<2> > hood = testGrid->getNeighborhood(Coord<2>(1,2));

        TS_ASSERT_EQUALS(hood[Coord<2>( 0, -1)].testValue, 205);

        TS_ASSERT_EQUALS(hood[Coord<2>(-1,  0)].testValue, 208);
        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  0)].testValue, 209);
        TS_ASSERT_EQUALS(hood[Coord<2>( 1,  0)].testValue, 210);

        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  1)].testValue, 213);
    }

    void testGetNeighborhoodReferenceReturn()
    {
        CoordMap<TestCell<2> > hood = testGrid->getNeighborhood(Coord<2>(1,2));
        /* We need to use absolute coordinates for testGrid, but relative
         * coordinates for the neighbourhood map "hood" 
         */

        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(1,1)], hood[Coord<2>( 0,-1)])

        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(0,2)], hood[Coord<2>(-1, 0)])
        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(1,2)], hood[Coord<2>( 0, 0)])
        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(2,2)], hood[Coord<2>( 1, 0)])

        TS_ASSERT_EQUALS((*testGrid)[Coord<2>(1,3)], hood[Coord<2>( 0, 1)])
    }

    void testGetNeighborhoodInUpperRightCorner()
    {
        CoordMap<TestCell<2> > hood = testGrid->getNeighborhood(Coord<2>(GRIDWIDTH-1, 0));

        TS_ASSERT_EQUALS(hood[Coord<2>( 0, -1)].testValue, 
                         TestCell<2>::defaultValue());

        TS_ASSERT_EQUALS(hood[Coord<2>(-1,  0)].testValue, 202);
        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  0)].testValue, 203);
        TS_ASSERT_EQUALS(hood[Coord<2>( 1,  0)].testValue,
                         TestCell<2>::defaultValue());

        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  1)].testValue, 207);
    }

    void testToString()
    {
        Grid<int> fooBar(Coord<2>(3, 2), 4711);
        // set edge cell...
        fooBar[Coord<2>(-1, -1)] = 2701;
        // ...and two others
        fooBar[Coord<2>( 1,  0)] = 19;
        fooBar[Coord<2>( 2,  1)] = 81;
        std::string expected = 
            "Grid\n"
            "boundingBox: origin: (0, 0)\n"
            "dimensions: (3, 2)\n"
            "edgeCell:\n"
            "2701\n"
            "Coord(0, 0):\n"
            "4711\n"
            "Coord(1, 0):\n"
            "19\n"
            "Coord(2, 0):\n"
            "4711\n"
            "Coord(0, 1):\n"
            "4711\n"
            "Coord(1, 1):\n"
            "4711\n"
            "Coord(2, 1):\n"
            "81\n";
        TS_ASSERT_EQUALS(fooBar.toString(), expected);        
    }

    void testEdgeCell()
    {
        Grid<int> fooBar(Coord<2>(12, 34), 56, 78);
        TS_ASSERT_EQUALS(fooBar.edgeCell, 78);
        TS_ASSERT_EQUALS(fooBar[Coord<2>(-1, -1)], 78);

        fooBar[Coord<2>(-1, -1)] = 90;
        TS_ASSERT_EQUALS(fooBar.edgeCell, 90);
        TS_ASSERT_EQUALS(fooBar[Coord<2>(-1, -1)], 90);

        fooBar.edgeCell = 10;
        TS_ASSERT_EQUALS(fooBar.edgeCell, 10);
        TS_ASSERT_EQUALS(fooBar[Coord<2>(-1, -1)], 10);
    }

    void testDefaultTopology()
    {
        Grid<int> g(Coord<2>(3, 4), 10, 11);
        // in-bounds accesses
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  0)], 10);
        TS_ASSERT_EQUALS(g[Coord<2>( 2,  0)], 10);
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  3)], 10);
        TS_ASSERT_EQUALS(g[Coord<2>( 2,  3)], 10);

        // out-of-bounds accesses should yield edge cell
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  4)], 11);
        TS_ASSERT_EQUALS(g[Coord<2>( 0, -1)], 11);
        TS_ASSERT_EQUALS(g[Coord<2>( 3,  0)], 11);
        TS_ASSERT_EQUALS(g[Coord<2>(-1,  0)], 11);
    }

    inline int trans(const int& x, const int& dimension)
    {
        return (x + dimension) % dimension;
    }

    void testTorusTopology()
    {
        Grid<int, Topologies::Torus<2>::Topology> g(Coord<2>(3, 4), 0, -1);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 3; ++x)
                g[Coord<2>(x, y)] = y * 10 + x;

        // in-bounds accesses
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  0)],  0);
        TS_ASSERT_EQUALS(g[Coord<2>( 2,  0)],  2);
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  3)], 30);
        TS_ASSERT_EQUALS(g[Coord<2>( 2,  3)], 32);

        // out-of-bounds accesses should not yield edge cell
        TS_ASSERT_EQUALS(g[Coord<2>( 0,  4)],  0);
        TS_ASSERT_EQUALS(g[Coord<2>( 0, -1)], 30);
        TS_ASSERT_EQUALS(g[Coord<2>( 3,  0)],  0);
        TS_ASSERT_EQUALS(g[Coord<2>(-1,  0)],  2);
        TS_ASSERT_EQUALS(g[Coord<2>(47, 69)], 12);
    }

    void testFill3d()
    {
        Coord<3> dim(3, 4, 5);
        Grid<int, Topologies::Cube<3>::Topology> g(dim, 47, 11);
        TS_ASSERT_EQUALS(g[Coord<3>(-1, 0, 0)],  11);

        for (int z=0; z < dim.z(); ++z)
            for (int y=0; y < dim.y(); ++y)
                for (int x=0; x < dim.x(); ++x) 
                    TS_ASSERT_EQUALS(g[Coord<3>(x, y, z)], 47);
    }

    void testCompare()
    {
        Coord<2> dim(5, 4);
        Grid<int> g1(dim, 4);
        Grid<int> g2(dim, 9);
        DisplacedGrid<int> g3(CoordBox<2>(Coord<2>(), dim), 4);
        DisplacedGrid<int> g4(CoordBox<2>(Coord<2>(), dim), 9);
        DisplacedGrid<int> g5(CoordBox<2>(Coord<2>(), Coord<2>(5, 5)), 9);

        TS_ASSERT(g1 != g2);
        TS_ASSERT(g1 == g3);
        TS_ASSERT(g1 != g4);
        TS_ASSERT(g1 != g5);

        TS_ASSERT(g2 != g3);
        TS_ASSERT(g2 == g4);
        TS_ASSERT(g2 != g5);
    }
};

}
