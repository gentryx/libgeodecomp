#include <fstream>
#include <sstream>
#include <cstdio>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/misc/testcell.h>

#define GRID_WIDTH 4
#define GRID_HEIGHT 5

double edge = 0;

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

/**
 * Test model for use with Boost.Serialization
 */
class MyComplicatedCell1
{
public:
    class API : public LibGeoDecomp::APITraits::HasBoostSerialization
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
    }

    inline bool operator==(const MyComplicatedCell1& other)
    {
        return
            (x     == other.x) &&
            (cargo == other.cargo);
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int version)
    {
        archive & x;
        archive & cargo;
    }

    int x;
    std::vector<int> cargo;
};

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
public:

    void setUp()
    {
        testGrid = new Grid<TestCell<2> >(Coord<2>(GRID_WIDTH, GRID_HEIGHT));

        int num = 200;
        for (int y = 0; y < testGrid->getDimensions().y(); y++) {
            for (int x = 0; x < testGrid->getDimensions().x(); x++) {
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

    void testBoundingBox()
    {
        CoordBox<2> rect(Coord<2>(), Coord<2>(12, 13));
        Grid<int> grid(rect.dimensions);
        TS_ASSERT_EQUALS(rect, grid.boundingBox());
        TS_ASSERT_EQUALS(12, grid.getDimensions().x());
        TS_ASSERT_EQUALS(13, grid.getDimensions().y());
    }

    void testBoundingRegion()
    {
        Coord<3> origin;
        Coord<3> dim(23, 24, 25);

        CoordBox<3> rect(origin, dim);
        Grid<double, Topologies::Cube<3>::Topology> grid(dim);

        Region<3> expectedRegion;
        expectedRegion << rect;

        TS_ASSERT_EQUALS(expectedRegion, grid.boundingRegion());
    }

    void testCopyConstructorFromGridBase()
    {
        Grid<int> a;
        Grid<int> b(Coord<2>(10, 5), 5);
        GridBase<int, 2>& base = b;
        a = Grid<int>(base);
        TS_ASSERT_EQUALS(a, b);
    }

    void testConstructorDefaultInit()
    {
        Grid<double> g(Coord<2>(10, 12), 14.16, 47.11);
        TS_ASSERT_EQUALS(Coord<2>(10, 12), g.getDimensions());
        TS_ASSERT_EQUALS(14.16, g[Coord<2>( 5,  6)]);
        TS_ASSERT_EQUALS(47.11, g[Coord<2>(-1, -1)]);
    }

    void testMultiDimensionalConstructor()
    {
        Coord<3> dim(3, 4, 5);
        Grid<double, Topologies::Torus<3>::Topology > g(dim);
        g.getEdgeCell() = -1;
        g[Coord<3>( 2, 1,  1)] = 1;
        g[Coord<3>(-1, 7, -2)] = 5;

        TS_ASSERT_EQUALS(dim, g.getDimensions());
        TS_ASSERT_EQUALS(-1, g.getEdgeCell());
        TS_ASSERT_EQUALS(1, g[Coord<3>(-1, 1, 1)]);
        TS_ASSERT_EQUALS(5, g[Coord<3>( 2, 3, 3)]);
    }

    void testOperatorEqual1()
    {
        int width = testGrid->getDimensions().x();
        int height = testGrid->getDimensions().y();
        double deltaTemp = 42.0;
        Coord<2> changedCoord(0, 4);

        Grid<TestCell<2> > other(Coord<2>(width, height));
        TS_ASSERT(!other[Coord<2>(2, 4)].isValid);
        for(int x = 0; x < width; x++) {
            for(int y = 0; y < height; y++) {
                other.cellVector[y * width + x] = testGrid->cellVector[y * width + x];
            }
        }

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

    void testCopyConstructor()
    {
        Grid<int> a1(Coord<2>(200, 100));
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 200; j++) {
                a1[Coord<2>(j, i)] = i * 200 + j;
            }
        }

        Grid<int> a2;
        a2 = a1;
        TS_ASSERT_EQUALS(a1, a1);
        TS_ASSERT_EQUALS(a1, a2);

        a2[Coord<2>(0, 0)] = 4711;
        TS_ASSERT_DIFFERS(a1, a2);
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
        CoordMap<TestCell<2> > hood = testGrid->getNeighborhood(Coord<2>(GRID_WIDTH - 1, 0));

        TS_ASSERT_EQUALS(hood[Coord<2>( 0, -1)].testValue,
                         TestCell<2>::defaultValue());

        TS_ASSERT_EQUALS(hood[Coord<2>(-1,  0)].testValue, 202);
        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  0)].testValue, 203);
        TS_ASSERT_EQUALS(hood[Coord<2>( 1,  0)].testValue,
                         TestCell<2>::defaultValue());

        TS_ASSERT_EQUALS(hood[Coord<2>( 0,  1)].testValue, 207);
    }

    void testGetSetManyCells()
    {
        TestCell<2> cells[2];
        testGrid->get(Streak<2>(Coord<2>(1, 3), 3), cells);

        for (int i = 0; i < 2; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid->get(Coord<2>(i + 1, 3)));
        }

        for (int i = 0; i < 2; ++i) {
            cells[i].testValue = i + 1234;
        }
        testGrid->set(Streak<2>(Coord<2>(1, 3), 3), cells);

        for (int i = 0; i < 2; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid->get(Coord<2>(i + 1, 3)));
        }
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
            "Grid<2>(\n"
            "boundingBox: CoordBox<2>(origin: (0, 0), dimensions: (3, 2))\n"
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
            "81\n"
            ")";
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

    inline int trans(int x, int dimension)
    {
        return (x + dimension) % dimension;
    }

    void testTorusTopology()
    {
        Grid<int, Topologies::Torus<2>::Topology> g(Coord<2>(3, 4), 0, -1);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 3; ++x) {
                g[Coord<2>(x, y)] = y * 10 + x;
            }
        }

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
        TS_ASSERT_EQUALS(g[Coord<2>( 5,  5)], 12);
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

    void testLoadSaveMember()
    {
        // basic setup:
        Selector<MyDummyCell> xSelector(&MyDummyCell::x, "x");
        Selector<MyDummyCell> ySelector(&MyDummyCell::y, "y");
        Selector<MyDummyCell> zSelector(&MyDummyCell::z, "z");

        Coord<2> dim(40, 20);
        Grid<MyDummyCell, Topologies::Cube<2>::Topology> grid(dim);
        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                grid[Coord<2>(x, y)] = MyDummyCell(x, y, 13);
            }
        }

        Region<2> region;
        region << Streak<2>(Coord<2>( 0,  0), 10)
               << Streak<2>(Coord<2>(10, 10), 20)
               << Streak<2>(Coord<2>(30, 19), 40);

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
        std::vector<TestCell<2> > buffer(10);
        Region<2> region;
        region << Streak<2>(Coord<2>(0, 0), 4)
               << Streak<2>(Coord<2>(1, 1), 3)
               << Streak<2>(Coord<2>(0, 4), 4);

        testGrid->saveRegion(&buffer, region);

        Region<2>::Iterator iter = region.begin();
        for (int i = 0; i < 10; ++i) {
            TestCell<2> actual = testGrid->get(*iter);
            TestCell<2> expected(*iter, testGrid->getDimensions());
            expected.testValue = 200 + iter->y() * GRID_WIDTH + iter->x();

            TS_ASSERT_EQUALS(actual, expected);
            ++iter;
        }

        // manipulate test data:
        for (int i = 0; i < 10; ++i) {
            buffer[i].pos = Coord<2>(-i, -10);
        }

        int index = 0;
        testGrid->loadRegion(buffer, region);
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> actual = testGrid->get(*i).pos;
            Coord<2> expected = Coord<2>(index, -10);
            TS_ASSERT_EQUALS(actual, expected);

            --index;
        }
    }

    void testLoadSaveRegionWithOffset()
    {
        std::vector<TestCell<2> > buffer(10);
        Region<2> region;
        region << Streak<2>(Coord<2>(0, 0), 4)
               << Streak<2>(Coord<2>(1, 1), 3)
               << Streak<2>(Coord<2>(0, 4), 4);
        TS_ASSERT_EQUALS(region.size(), 10);

        Region<2> regionWithOffset23;
        regionWithOffset23 << Streak<2>(Coord<2>(2, 3), 6)
                           << Streak<2>(Coord<2>(3, 4), 5)
                           << Streak<2>(Coord<2>(2, 7), 6);
        TS_ASSERT_EQUALS(regionWithOffset23.size(), 10);

        Region<2> regionWithOffset59;
        regionWithOffset59 << Streak<2>(Coord<2>(5,  9), 9)
                           << Streak<2>(Coord<2>(6, 10), 8)
                           << Streak<2>(Coord<2>(5, 13), 9);
        TS_ASSERT_EQUALS(regionWithOffset59.size(), 10);

        testGrid->saveRegion(&buffer, regionWithOffset23, Coord<2>(-2, -3));

        Region<2>::Iterator iter = region.begin();
        for (int i = 0; i < 10; ++i) {
            TestCell<2> actual = testGrid->get(*iter);
            TestCell<2> expected(*iter, testGrid->getDimensions());
            expected.testValue = 200 + iter->y() * GRID_WIDTH + iter->x();

            TS_ASSERT_EQUALS(actual, expected);
            ++iter;
        }

        // manipulate test data:
        for (int i = 0; i < 10; ++i) {
            buffer[i].pos = Coord<2>(-i, -12);
        }

        int index = 0;
        testGrid->loadRegion(buffer, regionWithOffset59, Coord<2>(-5, -9));
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> actual = testGrid->get(*i).pos;
            Coord<2> expected = Coord<2>(index, -12);
            TS_ASSERT_EQUALS(actual, expected);

            --index;
        }
    }

    void testLoadSaveRegionWithBoostSerialization()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        typedef Grid<MyComplicatedCell1> GridType;

        Coord<2> dim(13, 10);
        CoordBox<2> box(Coord<2>(), dim);

        GridType sendGrid(dim);
        GridType recvGrid(dim);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            MyComplicatedCell1 cell;
            cell.cargo << i->x();
            cell.cargo << i->y();
            cell.x = i->x() * 100 + i->y();
            sendGrid.set(*i, cell);
        }

        Region<2> region;
        region << Streak<2>(Coord<2>(4001, 3001), 4012)
               << Streak<2>(Coord<2>(4001, 3002), 4002)
               << Streak<2>(Coord<2>(4011, 3002), 4012)
               << Streak<2>(Coord<2>(4001, 3003), 4002)
               << Streak<2>(Coord<2>(4011, 3003), 4012)
               << Streak<2>(Coord<2>(4001, 3004), 4002)
               << Streak<2>(Coord<2>(4011, 3004), 4012)
               << Streak<2>(Coord<2>(4001, 3005), 4002)
               << Streak<2>(Coord<2>(4011, 3005), 4012)
               << Streak<2>(Coord<2>(4001, 3006), 4002)
               << Streak<2>(Coord<2>(4011, 3006), 4012)
               << Streak<2>(Coord<2>(4001, 3007), 4002)
               << Streak<2>(Coord<2>(4011, 3007), 4012)
               << Streak<2>(Coord<2>(4001, 3008), 4012);
        TS_ASSERT_EQUALS(region.size(), 34);
        Coord<2> offset(-4000, -3000);

        std::vector<char> buffer;
        sendGrid.saveRegion(&buffer, region, offset);

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_DIFFERS(sendGrid[*i + offset], recvGrid[*i + offset]);
        }
        recvGrid.loadRegion(buffer, region, offset);
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_EQUALS(sendGrid[*i + offset], recvGrid[*i + offset]);
        }
#endif
    }

    void testCreationOfZeroSizedGrid()
    {
        Grid<int, Topologies::Torus<1>::Topology> grid1;
        TS_ASSERT_EQUALS(Coord<1>(), grid1.getDimensions());

        Grid<int, Topologies::Torus<2>::Topology> grid2;
        TS_ASSERT_EQUALS(Coord<2>(), grid2.getDimensions());

        Grid<int, Topologies::Torus<3>::Topology> grid3;
        TS_ASSERT_EQUALS(Coord<3>(), grid3.getDimensions());
    }
private:
    Grid<TestCell<2> > *testGrid;
};

}
