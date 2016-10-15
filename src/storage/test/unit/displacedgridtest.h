#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/displacedgrid.h>

class MyDummyCell
{
public:
    class API :
        public LibGeoDecomp::APITraits::HasSoA
    {};

    explicit MyDummyCell(const int x = 0, const double y = 0, const char z = 0) :
        x(x),
        y(y),
        z(z)
    {}

    inline bool operator==(const MyDummyCell& other)
    {
        return
            (x == other.x) &&
            (y == other.y) &&
            (z == other.z);
    }

    int x;
    double y;
    char z;
};

LIBFLATARRAY_REGISTER_SOA(MyDummyCell, ((int)(x))((double)(y))((char)(z)) )

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DisplacedGridTest : public CxxTest::TestSuite
{
public:

    void testBoundingBox()
    {
        CoordBox<2> rect(Coord<2>(10, 11), Coord<2>(12, 13));
        DisplacedGrid<int> grid(rect);

        TS_ASSERT_EQUALS(rect, grid.boundingBox());
        TS_ASSERT_EQUALS(12, grid.getDimensions().x());
        TS_ASSERT_EQUALS(13, grid.getDimensions().y());
    }

    void testRegionConstructor()
    {
        CoordBox<2> rect(Coord<2>(44, 77), Coord<2>(55, 66));
        Region<2> region;
        region << rect;

        DisplacedGrid<int> grid(region);
        TS_ASSERT_EQUALS(rect, grid.boundingBox());
        TS_ASSERT_EQUALS(55, grid.getDimensions().x());
        TS_ASSERT_EQUALS(66, grid.getDimensions().y());
    }

    void testBoundingRegion()
    {
        Coord<3> origin(10, 11, 12);
        Coord<3> dim(13, 14, 15);

        CoordBox<3> rect(origin, dim);
        DisplacedGrid<double, Topologies::Cube<3>::Topology> grid(rect);

        Region<3> expectedRegion;
        expectedRegion << rect;

        TS_ASSERT_EQUALS(expectedRegion, grid.boundingRegion());
    }

    void testPaste()
    {
        DisplacedGrid<double> source(
            CoordBox<2>(Coord<2>( 0,  0), Coord<2>(100, 100)), 47.11);
        DisplacedGrid<double> target(
            CoordBox<2>(Coord<2>(40, 60), Coord<2>( 30,  30)), -1.23);
        DisplacedGrid<double> target2(
            CoordBox<2>(Coord<2>(40, 60), Coord<2>( 30,  30)), -5);

        Region<2> innerSquare;
        Region<2> outerSquare;
        for (int i = 70; i < 80; ++i) {
            innerSquare << Streak<2>(Coord<2>(50, i), 60);
        }
        for (int i = 60; i < 90; ++i) {
            outerSquare << Streak<2>(Coord<2>(40, i), 70);
        }

        Region<2> outerRing = outerSquare - innerSquare;

        for (Region<2>::Iterator i = outerSquare.begin(); i != outerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target[*i]);
        }

        target.paste(source, innerSquare);

        for (Region<2>::Iterator i = outerRing.begin(); i != outerRing.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target[*i]);
        }
        for (Region<2>::Iterator i = innerSquare.begin(); i != innerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(47.11, target[*i]);
        }

        target2.paste(target, outerRing);

        for (Region<2>::Iterator i = outerRing.begin(); i != outerRing.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target2[*i]);
        }

        for (Region<2>::Iterator i = innerSquare.begin(); i != innerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(-5, target2[*i]);
        }
    }

    void testSetOrigin()
    {
        DisplacedGrid<int> grid(CoordBox<2>(Coord<2>(10, 20), Coord<2>(2, 3)));

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                grid[Coord<2>(x + 10, y + 20)] = x * 10 + y;
            }
        }

        grid.setOrigin(Coord<2>(0, 50));
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                TS_ASSERT_EQUALS(grid[Coord<2>(x + 0, y + 50)],  x * 10 + y);
            }
        }
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

    void testGetSetManyCells()
    {
        Coord<2> origin(20, 15);
        Coord<2> dim(30, 10);
        Coord<2> end = origin + dim;
        DisplacedGrid<TestCell<2> > testGrid(CoordBox<2>(origin, dim));

        int num = 200;
        for (int y = origin.y(); y < end.y(); y++) {
            for (int x = origin.x(); x < end.x(); x++) {
                testGrid[Coord<2>(x, y)] =
                    TestCell<2>(Coord<2>(x, y), testGrid.getDimensions());
                testGrid[Coord<2>(x, y)].testValue =  num++;
            }
        }

        TestCell<2> cells[5];
        testGrid.get(Streak<2>(Coord<2>(21, 18), 26), cells);

        for (int i = 0; i < 5; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid.get(Coord<2>(i + 21, 18)));
        }

        for (int i = 0; i < 5; ++i) {
            cells[i].testValue = i + 1234;
        }
        testGrid.set(Streak<2>(Coord<2>(21, 18), 26), cells);

        for (int i = 0; i < 5; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid.get(Coord<2>(i + 21, 18)));
        }
    }

    void testFill3D()
    {
        CoordBox<3> insert(Coord<3>(10, 20, 15), Coord<3>(4, 7, 5));
        Coord<3> origin(10, 10, 10);
        Coord<3> dim(40, 70, 30);

        DisplacedGrid<int, Topologies::Cube<3>::Topology> g(CoordBox<3>(origin, dim), -1);
        g.fill(insert, 2);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c = Coord<3>(x, y, z) + origin;
                    int expected = -1;
                    if (insert.inBounds(c)) {
                        expected = 2;
                    }

                    if (expected != g[c]) {
                        std::cout << c << " expected: " << expected << " actual: " << g[c] << "\n";

                    }
                    TS_ASSERT_EQUALS(expected, g[c]);
                }
            }
        }
    }

    void testTopologicalNormalizationWithTorus()
    {
        /**
         * This test should check whether it is possible to let a
         * displaced grid act on a 15x10 torus as if it would cover a
         * 8x6 field shifted by an offset of (-3, -2), as outlined
         * below. The code will access the fields marked with a
         * through d.
         *
         *   0123456789abcde
         * 9 XXXXX_______cXX
         * 8 XXXXX_______aXX
         * 7 _______________
         * 6 _______________
         * 5 _______________
         * 4 _______________
         * 3 XXXXX_______XXX
         * 2 XXXXX_______XXX
         * 1 XXXXX_______XXX
         * 0 bXXXX_______XXd
         *
         */

        DisplacedGrid<int, Topologies::Torus<2>::Topology, true> grid(
            CoordBox<2>(Coord<2>(-3, -2),
                        Coord<2>(8, 6)),
            -2,
            -2,
            Coord<2>(15, 10));

        for (int y = -2; y < 4; ++y)
            for (int x = -3; x < 5; ++x)
                grid[Coord<2>(x, y)] = (y+3) * 10 + (x+3);

        TS_ASSERT_EQUALS(10, grid[Coord<2>(-3, -2)]);
        TS_ASSERT_EQUALS(33, grid[Coord<2>( 0,  0)]);
        TS_ASSERT_EQUALS(20, grid[Coord<2>(12, 9)]);
        TS_ASSERT_EQUALS(32, grid[Coord<2>(14, 0)]);

        TS_ASSERT_EQUALS(21, grid.getNeighborhood(
                             Coord<2>(12, 9))[Coord<2>(1, 0)]);
    }

    void testLoadSaveMember()
    {
        // basic setup:
        Selector<MyDummyCell> xSelector(&MyDummyCell::x, "x");
        Selector<MyDummyCell> ySelector(&MyDummyCell::y, "y");
        Selector<MyDummyCell> zSelector(&MyDummyCell::z, "z");

        Coord<2> offset(100, 200);
        Coord<2> dim(40, 20);
        CoordBox<2> box(offset, dim);
        DisplacedGrid<MyDummyCell, Topologies::Cube<2>::Topology> grid(box);
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            grid[*i] = MyDummyCell(i->x(), i->y(), 13);
        }

        Region<2> region;
        region << Streak<2>(Coord<2>(100, 200), 110)
               << Streak<2>(Coord<2>(110, 210), 120)
               << Streak<2>(Coord<2>(130, 219), 140);

        std::vector<int   > xVector(region.size(), -1);
        std::vector<double> yVector(region.size(), -1);
        std::vector<char  > zVector(region.size(), -1);

        grid.saveMember(&xVector[0], MemoryLocation::HOST, xSelector, region);
        grid.saveMember(&yVector[0], MemoryLocation::HOST, ySelector, region);
        grid.saveMember(&zVector[0], MemoryLocation::HOST, zSelector, region);

        Region<2>::Iterator cursor = region.begin();

        // test whether grid data is accurately copied back:
        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(xVector[i], cursor->x());
            TS_ASSERT_EQUALS(yVector[i], cursor->y());
            TS_ASSERT_EQUALS(zVector[i], 13);
            ++cursor;
        }

        // modify vectors and copy back to grid:
        for (std::size_t i = 0; i < region.size(); ++i) {
            xVector[i] = 1000 + i;
            yVector[i] = 2000 + i;
            zVector[i] = i;
        }

        grid.loadMember(&xVector[0], MemoryLocation::HOST, xSelector, region);
        grid.loadMember(&yVector[0], MemoryLocation::HOST, ySelector, region);
        grid.loadMember(&zVector[0], MemoryLocation::HOST, zSelector, region);

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_EQUALS(grid[*i], MyDummyCell(1000 + counter, 2000 + counter, counter));
            ++counter;
        }
    }

    void testLoadSaveRegion()
    {
        Coord<2> origin(20, 15);
        Coord<2> dim(30, 10);
        Coord<2> end = origin + dim;
        DisplacedGrid<TestCell<2> > testGrid(CoordBox<2>(origin, dim));

        int num = 200;
        for (int y = origin.y(); y < end.y(); y++) {
            for (int x = origin.x(); x < end.x(); x++) {
                testGrid[Coord<2>(x, y)] =
                    TestCell<2>(Coord<2>(x, y), testGrid.getDimensions());
                testGrid[Coord<2>(x, y)].testValue =  num++;
            }
        }

        std::vector<TestCell<2> > buffer(70);
        Region<2> region;
        region << Streak<2>(Coord<2>(20, 15), 50)
               << Streak<2>(Coord<2>(21, 16), 31)
               << Streak<2>(Coord<2>(20, 24), 50);
        TS_ASSERT_EQUALS(buffer.size(), region.size());

        testGrid.saveRegion(&buffer, region);

        Region<2>::Iterator iter = region.begin();
        for (int i = 0; i < 70; ++i) {
            TestCell<2> actual = testGrid.get(*iter);
            TestCell<2> expected(*iter, testGrid.getDimensions());
            int expectedIndex = 200 + (*iter - origin).toIndex(dim);
            expected.testValue = expectedIndex;

            TS_ASSERT_EQUALS(actual, expected);
            ++iter;
        }

        // manupulate test data:
        for (int i = 0; i < 70; ++i) {
            buffer[i].pos = Coord<2>(-i, -10);
        }

        int index = 0;
        testGrid.loadRegion(buffer, region);
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> actual = testGrid.get(*i).pos;
            Coord<2> expected = Coord<2>(index, -10);
            TS_ASSERT_EQUALS(actual, expected);

            --index;
        }
    }

    void testLoadSaveRegionWithOffset()
    {
        Coord<2> origin(20, 15);
        Coord<2> dim(40, 15);
        Coord<2> end = origin + dim;
        DisplacedGrid<TestCell<2> > testGrid(CoordBox<2>(origin, dim));

        int num = 200;
        for (int y = origin.y(); y < end.y(); y++) {
            for (int x = origin.x(); x < end.x(); x++) {
                testGrid[Coord<2>(x, y)] =
                    TestCell<2>(Coord<2>(x, y), testGrid.getDimensions());
                testGrid[Coord<2>(x, y)].testValue =  num++;
            }
        }

        std::vector<TestCell<2> > buffer(90);
        Region<2> region;
        region << Streak<2>(Coord<2>(20, 15), 60)
               << Streak<2>(Coord<2>(21, 16), 31)
               << Streak<2>(Coord<2>(20, 29), 60);
        TS_ASSERT_EQUALS(buffer.size(), region.size());

        Region<2> regionWithOffset59;
        regionWithOffset59 << Streak<2>(Coord<2>(25, 24), 65)
                           << Streak<2>(Coord<2>(26, 25), 36)
                           << Streak<2>(Coord<2>(25, 38), 65);
        TS_ASSERT_EQUALS(buffer.size(), regionWithOffset59.size());

        Region<2> regionWithOffset24;
        regionWithOffset24 << Streak<2>(Coord<2>(22, 19), 62)
                           << Streak<2>(Coord<2>(23, 20), 33)
                           << Streak<2>(Coord<2>(22, 33), 62);
        TS_ASSERT_EQUALS(buffer.size(), regionWithOffset24.size());

        testGrid.saveRegion(&buffer, regionWithOffset59, Coord<2>(-5, -9));

        Region<2>::Iterator iter = region.begin();
        for (int i = 0; i < 90; ++i) {
            TestCell<2> actual = testGrid.get(*iter);
            TestCell<2> expected(*iter, testGrid.getDimensions());
            int expectedIndex = 200 + (*iter - origin).toIndex(dim);
            expected.testValue = expectedIndex;

            TS_ASSERT_EQUALS(actual, expected);
            ++iter;
        }

        // manupulate test data:
        for (int i = 0; i < 90; ++i) {
            buffer[i].pos = Coord<2>(-i, -20);
        }

        int index = 0;
        testGrid.loadRegion(buffer, regionWithOffset24, Coord<2>(-2, -4));
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> actual = testGrid.get(*i).pos;
            Coord<2> expected = Coord<2>(index, -20);
            TS_ASSERT_EQUALS(actual, expected);

            --index;
        }
    }
};

}
