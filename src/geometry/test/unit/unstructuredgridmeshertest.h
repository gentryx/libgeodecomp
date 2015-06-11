#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/unstructuredgridmesher.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredGridMesherTest : public CxxTest::TestSuite
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

        UnstructuredGridMesher<2> mesher(points, neighbors);

        TS_ASSERT_EQUALS(FloatCoord<2>(6, 3), mesher.cellDimension());
        TS_ASSERT_EQUALS(Coord<2>(2, 3),      mesher.logicalGridDimension());
        TS_ASSERT_EQUALS(Coord<2>( 0,  0), mesher.positionToLogicalCoord(FloatCoord<2>(1.00,  0.00)));
        TS_ASSERT_EQUALS(Coord<2>(-1,  0), mesher.positionToLogicalCoord(FloatCoord<2>(0.99,  0.00)));
        TS_ASSERT_EQUALS(Coord<2>( 0, -1), mesher.positionToLogicalCoord(FloatCoord<2>(1.00, -0.01)));
        TS_ASSERT_EQUALS(Coord<2>(-1, -1), mesher.positionToLogicalCoord(FloatCoord<2>(0.99, -0.01)));
    }

    void testIfNodesSitDirectlyOnBoundaries()
    {
        std::vector<FloatCoord<2> > points;
        std::vector<std::vector<int> > neighbors;
        /**
         *  0123456789
         * 0
         * 1
         * 2 14562
         * 3     7
         * 4   0 8
         * 5     3
         * 6
         * 7
         * 8
         * 9
         */
        points << FloatCoord<2>(3, 4)
               << FloatCoord<2>(1, 2)
               << FloatCoord<2>(5, 2)
               << FloatCoord<2>(5, 5)
               << FloatCoord<2>(2, 2)
               << FloatCoord<2>(3, 2)
               << FloatCoord<2>(4, 2)
               << FloatCoord<2>(5, 3)
               << FloatCoord<2>(5, 4);

        std::vector<int> hood_0;
        neighbors << hood_0;

        std::vector<int> hood_1;
        hood_1 << 4;
        neighbors << hood_1;

        std::vector<int> hood_2;
        hood_2 << 6
               << 7;
        neighbors << hood_2;

        std::vector<int> hood_3;
        hood_3 << 8;
        neighbors << hood_3;

        std::vector<int> hood_4;
        hood_4 << 1
               << 5;
        neighbors << hood_4;

        std::vector<int> hood_5;
        hood_5 << 4
               << 6;
        neighbors << hood_5;

        std::vector<int> hood_6;
        hood_6 << 5
               << 2;
        neighbors << hood_6;

        std::vector<int> hood_7;
        hood_7 << 2
               << 8;
        neighbors << hood_7;

        std::vector<int> hood_8;
        hood_8 << 7
               << 3;
        neighbors << hood_8;

        UnstructuredGridMesher<2> mesher(points, neighbors);

        TS_ASSERT_EQUALS(FloatCoord<2>(1, 1), mesher.cellDimension());
        TS_ASSERT_EQUALS(Coord<2>(5, 4),      mesher.logicalGridDimension());


    }
};

}
