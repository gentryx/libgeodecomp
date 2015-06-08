#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/gridspacingcalculator.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class GridSpacingCalculatorTest : public CxxTest::TestSuite
{
public:
    void testWithFloatCoord2D()
    {
        std::vector<FloatCoord<2> > points;
        std::vector<std::vector<int> > neighbors;
        /**
         *  0123456789
         * 0   1-----3
         * 1  / \--5/
         * 2 0    /
         * 3  \  4
         * 4   \ |  6
         * 5     2
         * 6
         * 7
         * 8 7
         * 9
         */
        points << FloatCoord<2>(1, 2)
               << FloatCoord<2>(3, 0)
               << FloatCoord<2>(5, 5)
               << FloatCoord<2>(9, 0)
               << FloatCoord<2>(5, 3)
               << FloatCoord<2>(7, 1)
               << FloatCoord<2>(8, 4)
               << FloatCoord<2>(1, 8);

        std::vector<int> hood_0;
        hood_0 << 1
               << 2;
        neighbors << hood_0;

        std::vector<int> hood_1;
        hood_1 << 0
               << 3
               << 5;
        neighbors << hood_1;

        std::vector<int> hood_2;
        hood_2 << 0
               << 4;
        neighbors << hood_2;

        std::vector<int> hood_3;
        hood_3 << 1
               << 5;
        neighbors << hood_3;

        std::vector<int> hood_4;
        hood_4 << 2
               << 5;
        neighbors << hood_4;

        std::vector<int> hood_5;
        hood_5 << 1
               << 3
               << 4;
        neighbors << hood_5;

        std::vector<int> hood_6;
        // empty neighborhood
        neighbors << hood_6;

        std::vector<int> hood_7;
        // empty neighborhood
        neighbors << hood_7;

        Coord<2> logicalGridDim;
        FloatCoord<2> cellDim;
        GridSpacingCalculator::determineGridDimensions(points, neighbors, &logicalGridDim, &cellDim);

        TS_ASSERT_EQUALS(cellDim, FloatCoord<2>(6, 3));
        TS_ASSERT_EQUALS(logicalGridDim, Coord<2>(2, 3));
    }
};

}
