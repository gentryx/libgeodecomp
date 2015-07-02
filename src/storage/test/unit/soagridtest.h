#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

class SoATestCell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::Moore<3, 2> >
    {};

    explicit SoATestCell(int v = 0) :
        v(v)
    {}

    bool operator==(const SoATestCell& other)
    {
        return v == other.v;
    }

    int v;
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const SoATestCell& cell)
{
    __os << "SoATestCell(" << cell.v << ")";
    return __os;
}

LIBFLATARRAY_REGISTER_SOA(SoATestCell, ((int)(v)))

class MyDummyCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit MyDummyCell(const int x = 0, const double y = 0, const char z = 0) :
        x(x),
        y(y),
        z(z)
    {}

    int x;
    double y;
    char z;
};

LIBFLATARRAY_REGISTER_SOA(MyDummyCell, ((int)(x))((double)(y))((char)(z)) )

class CellWithArrayMember
{
public:
    class API :
          public LibGeoDecomp::APITraits::HasFixedCoordsOnlyUpdate,
          public LibGeoDecomp::APITraits::HasUpdateLineX,
          public LibGeoDecomp::APITraits::HasStencil<LibGeoDecomp::Stencils::Moore<3, 1> >,
#ifdef LIBGEODECOMP_WITH_MPI
          public LibGeoDecomp::APITraits::HasOpaqueMPIDataType<CellWithArrayMember>,
#endif
          public LibGeoDecomp::APITraits::HasTorusTopology<3>,
          public LibGeoDecomp::APITraits::HasSoA
    {};

    CellWithArrayMember(Coord<3> var0 = Coord<3>(), Coord<3> var1 = Coord<3>(), int var2 = 0, int var3 = 0)
    {
        temp[0] = var0[0];
        temp[1] = var0[1];
        temp[2] = var0[2];
        temp[3] = var1[0];
        temp[4] = var1[1];
        temp[5] = var1[2];
        temp[6] = var2;
        temp[7] = var3;

        for (int i = 8; i < 40; ++i) {
            temp[i] = -1;
        }
    }

    bool operator==(const CellWithArrayMember& other)
    {
        for (int i = 8; i < 40; ++i) {
            if (temp[i] != other.temp[i]) {
                return false;
            }
        }

        return true;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, const int nanoStep)
    {}

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_OLD& hoodOld, int indexEnd,
                            HOOD_NEW& hoodNew, int /* nanoStep */)
    {}

    double temp[40];
};


LIBFLATARRAY_REGISTER_SOA(
    CellWithArrayMember,
    ((double)(temp)(40)) )

class VoidInitializer : public SimpleInitializer<CellWithArrayMember>
{
public:
    VoidInitializer() :
        SimpleInitializer<CellWithArrayMember>(Coord<3>(30, 20, 10), 10)
    {}

    virtual void grid(LibGeoDecomp::GridBase<Cell, 3> *ret)
    {
        // intentionally left empty
    }
};

namespace LibGeoDecomp {

class CheckCellValues
{
public:
    CheckCellValues(long startOffset, long endOffset, long expected) :
        startOffset(startOffset),
        endOffset(endOffset),
        expected(expected)
    {}

    template<typename ACCESSOR>
    void operator()(ACCESSOR accessor)
    {
        accessor.index += startOffset;

        for (long offset = startOffset; offset < endOffset; ++offset) {
            TS_ASSERT_EQUALS(expected, accessor.v());
            ++accessor.index;
        }
    }

private:
    long startOffset;
    long endOffset;
    long expected;
};

class SoAGridTest : public CxxTest::TestSuite
{
public:
    typedef TestCellSoA TestCellType2;
    typedef APITraits::SelectTopology<TestCellType2>::Value Topology2;

    typedef APITraits::SelectTopology<MyDummyCell>::Value Topology3;

    void testBasic()
    {
        CoordBox<3> box(Coord<3>(10, 15, 22), Coord<3>(50, 40, 35));
        SoATestCell defaultCell(1);
        SoATestCell edgeCell(2);

        SoAGrid<SoATestCell, Topologies::Cube<3>::Topology> grid(box, defaultCell, edgeCell);
        grid.set(Coord<3>(1, 1, 1) + box.origin, SoATestCell(3));
        grid.set(Coord<3>(2, 2, 3) + box.origin, SoATestCell(4));

        TS_ASSERT_EQUALS(grid.actualDimensions, Coord<3>(54, 44, 39));
        TS_ASSERT_EQUALS(grid.boundingBox(), box);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0)), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0) + box.origin), defaultCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(1, 1, 1) + box.origin), SoATestCell(3));
        TS_ASSERT_EQUALS(grid.get(Coord<3>(2, 2, 3) + box.origin), SoATestCell(4));

        edgeCell = SoATestCell(-1);
        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0)), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0) + box.origin), defaultCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(1, 1, 1) + box.origin), SoATestCell(3));
        TS_ASSERT_EQUALS(grid.get(Coord<3>(2, 2, 3) + box.origin), SoATestCell(4));
    }

    void test2d()
    {
        CoordBox<2> box(Coord<2>(10, 15), Coord<2>(50, 40));
        SoATestCell defaultCell(1);
        SoATestCell edgeCell(2);

        SoAGrid<SoATestCell, Topologies::Cube<2>::Topology> grid(box, defaultCell, edgeCell);

        grid.set(Coord<2>(1, 1) + box.origin, SoATestCell(3));
        grid.set(Coord<2>(2, 2) + box.origin, SoATestCell(4));

        TS_ASSERT_EQUALS(grid.actualDimensions, Coord<3>(54, 44, 1));
        TS_ASSERT_EQUALS(grid.boundingBox(), box);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(0, 0)), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(0, 0) + box.origin).v, defaultCell.v);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(1, 1) + box.origin), SoATestCell(3));
        TS_ASSERT_EQUALS(grid.get(Coord<2>(2, 2) + box.origin), SoATestCell(4));
        TS_ASSERT_EQUALS(grid.get(Coord<2>(3, 3) + box.origin), SoATestCell(1));

        edgeCell = SoATestCell(-1);
        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(0, 0)), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(0, 0) + box.origin), defaultCell);
        TS_ASSERT_EQUALS(grid.get(Coord<2>(1, 1) + box.origin), SoATestCell(3));
        TS_ASSERT_EQUALS(grid.get(Coord<2>(2, 2) + box.origin), SoATestCell(4));
    }

    void testGetSetManyCells()
    {
        Coord<2> origin(20, 15);
        Coord<2> dim(30, 10);
        Coord<2> end = origin + dim;
        SoAGrid<SoATestCell> testGrid(CoordBox<2>(origin, dim));

        int num = 200;
        for (int y = origin.y(); y < end.y(); y++) {
            for (int x = origin.x(); x < end.x(); x++) {
                testGrid.set(Coord<2>(x, y), SoATestCell(num * 10000 + x * 100 + y));
            }
        }

        SoATestCell cells[5];
        testGrid.get(Streak<2>(Coord<2>(21, 18), 26), cells);

        for (int i = 0; i < 5; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid.get(Coord<2>(i + 21, 18)));
        }

        for (int i = 0; i < 5; ++i) {
            cells[i].v = i + 1234;
        }
        testGrid.set(Streak<2>(Coord<2>(21, 18), 26), cells);

        for (int i = 0; i < 5; ++i) {
            TS_ASSERT_EQUALS(cells[i], testGrid.get(Coord<2>(i + 21, 18)));
        }
    }

    void testInitialization()
    {
        CoordBox<3> box(Coord<3>(20, 25, 32), Coord<3>(51, 21, 15));
        Coord<3> topoDim(60, 50, 50);
        SoATestCell defaultCell(1);
        SoATestCell edgeCell(2);

        // next larger dimensions for x/y from 51 and 21 are 64 and 64.
        // int oppositeSideOffset = 22 * 64 + 16 * 64 * 64;
        int oppositeSideOffset0 = (21 + 4 - 1) * 64;
        int oppositeSideOffset1 = (15 + 4 - 1) * 64 * 64;
        int oppositeSideOffset2 = (15 + 4 - 1) * 64 * 64 + (21 + 4 - 1) * 64;

        SoAGrid<SoATestCell, Topologies::Cube<3>::Topology, true> grid(box, defaultCell, edgeCell, topoDim);

        // check not only first row, but also all other 3 outermost horizontal edges:
        // (width 51 + 2 edge cell layers on each side)
        grid.callback(CheckCellValues(0, 51 + 4, 2));
        grid.callback(CheckCellValues(0 + oppositeSideOffset0, 51 + 4 + oppositeSideOffset0, 2));
        grid.callback(CheckCellValues(0 + oppositeSideOffset1, 51 + 4 + oppositeSideOffset1, 2));
        grid.callback(CheckCellValues(0 + oppositeSideOffset2, 51 + 4 + oppositeSideOffset2, 2));

        grid.setEdge(SoATestCell(4));

        grid.callback(CheckCellValues(0, 51 + 4, 4));
        grid.callback(CheckCellValues(0 + oppositeSideOffset0, 51 + 4 + oppositeSideOffset0, 4));
        grid.callback(CheckCellValues(0 + oppositeSideOffset1, 51 + 4 + oppositeSideOffset1, 4));
        grid.callback(CheckCellValues(0 + oppositeSideOffset2, 51 + 4 + oppositeSideOffset2, 4));
    }

    void testDisplacementWithTopologicalCorrectness()
    {
        CoordBox<3> box(Coord<3>(20, 25, 32), Coord<3>(50, 40, 35));
        Coord<3> topoDim(60, 50, 50);
        SoATestCell defaultCell(1);
        SoATestCell edgeCell(2);

        SoAGrid<SoATestCell, Topologies::Torus<3>::Topology, true> grid(box, defaultCell, edgeCell, topoDim);
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT_EQUALS(grid.get(*i), defaultCell);
        }

        // here we check that topological correctness correctly maps
        // coordinates in the octant close to the origin to the
        // overlap of the far end of the grid delimited by topoDim.
        CoordBox<3> originOctant(Coord<3>(), box.origin + box.dimensions - topoDim);
        for (CoordBox<3>::Iterator i = originOctant.begin(); i != originOctant.end(); ++i) {
            TS_ASSERT_EQUALS(grid.get(*i), defaultCell);
        }

        SoATestCell dummy(4711);
        grid.set(Coord<3>(1, 2, 3), dummy);
        TS_ASSERT_EQUALS(grid.get(Coord<3>( 1,  2,  3)), dummy);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(61, 52, 53)), dummy);
    }


    void testSoA()
    {
        Coord<3> dim(30, 20, 10);
        SoAGrid<TestCellType2, Topology2> grid(CoordBox<3>(Coord<3>(), dim));
        Region<3> region;
        region << Streak<3>(Coord<3>(0,  0, 0), 30)
               << Streak<3>(Coord<3>(5, 11, 0), 24)
               << Streak<3>(Coord<3>(2,  5, 5), 20);
        std::vector<char> buffer(
            SoAGrid<TestCellType2, Topology2>::AGGREGATED_MEMBER_SIZE *
            region.size());

        int counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    TestCellType2 cell(c, dim, 0, counter);
                    grid.set(c, cell);
                    ++counter;
                }
            }
        }

        grid.saveRegion(&buffer[0], region);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    TestCellType2 cell = grid.get(c);
                    cell.testValue = 666;
                    grid.set(c, cell);
                }
            }
        }

        grid.loadRegion(&buffer[0], region);

        counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    double testValue = 666;
                    if (region.count(c)) {
                        testValue = counter;
                    }

                    TestCellType2 actual = grid.get(c);
                    TestCellType2 expected(c, dim, 0, testValue);

                    TS_ASSERT_EQUALS(expected, actual);
                    ++counter;
                }
            }
        }
    }

    // test load/save region with an array member
    void testSoA2()
    {
        Coord<3> dim(30, 20, 10);
        SoAGrid<CellWithArrayMember, Topology2> grid(CoordBox<3>(Coord<3>(), dim));
        Region<3> region;
        region << Streak<3>(Coord<3>(0,  0, 0), 30)
               << Streak<3>(Coord<3>(5, 11, 0), 24)
               << Streak<3>(Coord<3>(2,  5, 5), 20);
        std::vector<char> buffer(
            SoAGrid<CellWithArrayMember, Topology2>::AGGREGATED_MEMBER_SIZE *
            region.size());

        int counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    CellWithArrayMember cell(c, dim, 0, counter);
                    grid.set(c, cell);
                    ++counter;
                }
            }
        }

        grid.saveRegion(&buffer[0], region);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    CellWithArrayMember cell = grid.get(c);
                    cell.temp[7] = 666;
                    grid.set(c, cell);
                }
            }
        }

        grid.loadRegion(&buffer[0], region);

        counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    double testValue = 666;
                    if (region.count(c)) {
                        testValue = counter;
                    }

                    CellWithArrayMember actual = grid.get(c);
                    CellWithArrayMember expected(c, dim, 0, testValue);

                    TS_ASSERT_EQUALS(expected, actual);
                    ++counter;
                }
            }
        }
    }

    void testSoAWithOffset()
    {
        Coord<3> origin(5, 7, 3);
        Coord<3> dim(30, 20, 10);
        SoAGrid<TestCellType2, Topology2> grid(CoordBox<3>(origin, dim));
        Region<3> region;
        region << Streak<3>(Coord<3>(5,  7,  3), 30)
               << Streak<3>(Coord<3>(8, 11,  3), 24)
               << Streak<3>(Coord<3>(9, 10,  5), 20)
               << Streak<3>(Coord<3>(5, 27, 13), 35);
        std::vector<char> buffer(
            SoAGrid<TestCellType2, Topology2>::AGGREGATED_MEMBER_SIZE *
            region.size());

        int counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    TestCellType2 cell(c, dim, 0, counter);
                    grid.set(origin + c, cell);
                    ++counter;
                }
            }
        }

        grid.saveRegion(&buffer[0], region);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    TestCellType2 cell = grid.get(origin + c);
                    cell.testValue = 666;
                    grid.set(origin + c, cell);
                }
            }
        }

        grid.loadRegion(&buffer[0], region);

        counter = 444;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c(x, y, z);
                    double testValue = 666;
                    if (region.count(c + origin)) {
                        testValue = counter;
                    }

                    TestCellType2 actual = grid.get(origin + c);
                    TestCellType2 expected(c, dim, 0, testValue);

                    if (expected != actual) {
                        std::cout << "error at " << c << "\n"
                                  << "  expected: " << expected.toString() << "\n"
                                  << "  actual: " << actual.toString() << "\n";
                    }

                    TS_ASSERT_EQUALS(expected, actual);
                    ++counter;
                }
            }
        }
    }

    void testLoadSaveMember2D()
    {
        // basic setup:
        Selector<MyDummyCell> ySelector(&MyDummyCell::y, "y");

        Coord<2> origin(61, 62);
        Coord<2> dim(50, 40);
        double defaultValue = 111.222;

        SoAGrid<MyDummyCell, Topology3> grid(
            CoordBox<2>(origin, dim),
            MyDummyCell(0, defaultValue, 0));

        Region<2> region;
        region << Streak<2>(Coord<2>(61,  62),  70)
               << Streak<2>(Coord<2>(70,  80),  90)
               << Streak<2>(Coord<2>(65, 101), 111);

        std::vector<double> yVector(region.size(), -47);
        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(yVector[i], -47);
        }

        // test whether default grid data is accurately copied back:
        grid.saveMember(
            &yVector[0],
            MemoryLocation::HOST,
            ySelector,
            region);

        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(yVector[i], defaultValue);
        }

        // modify grid and repeat:
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, MyDummyCell(1, i->x() + i->y(), 1));
        }

        grid.saveMember(
            &yVector[0],
            MemoryLocation::HOST,
            ySelector,
            region);

        Region<2>::Iterator cursor = region.begin();
        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(yVector[i], cursor->x() + cursor->y());
            ++cursor;
        }

        // test loadMember, too:
        for (std::size_t i = 0; i < region.size(); ++i) {
            yVector[i] = i + 0.4711;
        }
        grid.loadMember(
            &yVector[0],
            MemoryLocation::HOST,
            ySelector,
            region);

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            double expected = counter + 0.4711;
            TS_ASSERT_EQUALS(grid.get(*i).y, expected);
            ++counter;
        }
    }

    void testLoadSaveMember3D()
    {
        // basic setup:
        Selector<TestCellType2> posSelector(&TestCellType2::pos, "pos");

        Coord<3> origin(601, 602, 603);
        Coord<3> dim(50, 40, 20);
        double defaultValue = 20.062011;

        SoAGrid<TestCellType2, Topology2> grid(
            CoordBox<3>(origin, dim),
            TestCellType2(Coord<3>(-1, -2, -3), Coord<3>(-4, -5, -6), 0, defaultValue));

        Region<3> region;
        region << Streak<3>(Coord<3>(610,  610,  610), 630)
               << Streak<3>(Coord<3>(610,  611,  610), 620)
               << Streak<3>(Coord<3>(630,  610,  620), 640);

        std::vector<Coord<3> > posVector(region.size(), Coord<3>(44, 55, 66));
        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(posVector[i], Coord<3>(44, 55, 66));
        }

        // test whether default grid data is accurately copied back:
        grid.saveMember(
            &posVector[0],
            MemoryLocation::HOST,
            posSelector,
            region);

        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(posVector[i], Coord<3>(-1, -2, -3));
        }

        // modify grid and repeat:
        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, TestCellType2(*i, dim, 1, 47.11));
        }

        grid.saveMember(
            &posVector[0],
            MemoryLocation::HOST,
            posSelector,
            region);

        Region<3>::Iterator cursor = region.begin();
        for (std::size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(posVector[i], *cursor);
            ++cursor;
        }

        // test loadMember, too:
        for (std::size_t i = 0; i < region.size(); ++i) {
            posVector[i] = Coord<3>(i, i * 1000, 4711);
        }
        grid.loadMember(
            &posVector[0],
            MemoryLocation::HOST,
            posSelector,
            region);

        int counter = 0;
        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<3> expected(counter, counter * 1000, 4711);
            TS_ASSERT_EQUALS(grid.get(*i).pos, expected);
            ++counter;
        }
    }

    void testSimulatorCreation()
    {
        SerialSimulator<CellWithArrayMember> sim(new VoidInitializer());
        sim.run();
    }
};

}
