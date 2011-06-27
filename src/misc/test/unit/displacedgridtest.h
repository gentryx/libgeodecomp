#include <cxxtest/TestSuite.h>
#include "../../displacedgrid.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class DisplacedGridTest : public CxxTest::TestSuite
{
public:
    
    void testBox()
    {
        CoordBox<2> rect(Coord<2>(10, 11), Coord<2>(12, 13));
        DisplacedGrid<int> grid(rect);
        TS_ASSERT_EQUALS(rect, grid.boundingBox());
        TS_ASSERT_EQUALS(12, grid.getDimensions().x());
        TS_ASSERT_EQUALS(13, grid.getDimensions().y());
    }

    void testPaste()
    {
        DisplacedGrid<double> source(
            CoordBox<2>(Coord<2>( 0,  0), Coord<2>(100, 100)), 47.11);
        DisplacedGrid<double> target(
            CoordBox<2>(Coord<2>(40, 60), Coord<2>( 30,  30)), -1.23);

        Region<2> innerSquare;
        for (int i = 70; i < 80; ++i)
            innerSquare << Streak<2>(Coord<2>(50, i), 60);
        Region<2> outerSquare;
        for (int i = 60; i < 90; ++i)
            outerSquare << Streak<2>(Coord<2>(40, i), 70);
        Region<2> outerRing = outerSquare - innerSquare;

        for (Region<2>::Iterator i = outerSquare.begin(); 
             i != outerSquare.end(); ++i) 
            TS_ASSERT_EQUALS(-1.23, target[*i]);

        target.paste(source, innerSquare);
        for (Region<2>::Iterator i = outerRing.begin(); 
             i != outerRing.end(); ++i)
            TS_ASSERT_EQUALS(-1.23, target[*i]);
        for (Region<2>::Iterator i = innerSquare.begin(); 
             i != innerSquare.end(); ++i)
            TS_ASSERT_EQUALS(47.11, target[*i]);
    }

    void testResize()
    {
        CoordBox<3> oldBox(Coord<3>(1, 1, 2), Coord<3>(3, 4, 2));
        CoordBox<3> newBox(Coord<3>(4, 4, 4), Coord<3>(3, 3, 3));

        DisplacedGrid<double, Topologies::Torus<3>::Topology> grid(oldBox);
        grid[Coord<3>(1, 1, 2)] = 47;
        grid[Coord<3>(0, 0, 1)] = 11;
        TS_ASSERT_EQUALS(47, grid[Coord<3>(1, 1, 2)]);
        TS_ASSERT_EQUALS(11, grid[Coord<3>(3, 4, 3)]);
        TS_ASSERT_EQUALS(oldBox, grid.boundingBox());

        grid.resize(newBox);
        grid[Coord<3>(4, 4, 4)] = 27;
        grid[Coord<3>(3, 3, 3)] = 1;
        TS_ASSERT_EQUALS(27, grid[Coord<3>(4, 4, 4)]);
        TS_ASSERT_EQUALS(1,  grid[Coord<3>(6, 6, 6)]);
        TS_ASSERT_EQUALS(newBox, grid.boundingBox());
    }
    
};

};
