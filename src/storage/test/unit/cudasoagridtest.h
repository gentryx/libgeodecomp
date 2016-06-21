#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/storage/cudasoagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDASoAGridTest : public CxxTest::TestSuite
{
public:
    void testConstructor()
    {
        TestCellSoA defaultCell(
            Coord<3>(1, 2, 3),
            Coord<3>(4, 5, 6),
            7,
            8);
        TestCellSoA edgeCell(
            Coord<3>( 9, 10, 11),
            Coord<3>(12, 13, 14),
            15,
            16);

        Coord<3> dim(51, 43, 21);
        Coord<3> origin(12, 41, 12);
        CoordBox<3> box(origin, dim);

        Region<3> region;
        region << box;
        Region<3> edgeRegion = region.expand(1) - region;

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box, defaultCell, edgeCell);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_EQUALS(defaultCell, grid.get(*i));
        }

        for (Region<3>::Iterator i = edgeRegion.begin(); i != edgeRegion.end(); ++i) {
            TS_ASSERT_EQUALS(edgeCell, grid.get(*i));
        }

        TS_ASSERT_EQUALS(grid.getEdge(), edgeCell);
    }

    void testTorus()
    {
        TestCellSoA defaultCell(
            Coord<3>(1, 2, 3),
            Coord<3>(4, 5, 6),
            7,
            8);
        TestCellSoA edgeCell(
            Coord<3>( 9, 10, 11),
            Coord<3>(12, 13, 14),
            15,
            16);

        Coord<3> dim(10, 40, 20);
        Coord<3> origin(0, 0, 0);
        CoordBox<3> box(origin, dim);
        Region<3> region;
        region << box;

        CUDASoAGrid<TestCellSoA, Topologies::Torus<3>::Topology, true> grid(
            box,
            defaultCell,
            edgeCell,
            dim);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellSoA cell = defaultCell;
            cell.pos = *i;
            grid.set(*i, cell);
        }

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellSoA cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.pos, *i);
        }

        TS_ASSERT_EQUALS(Coord<3>( 0,  0, 19), grid.get(Coord<3>( 0,  0, -1)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 3,  2, 19), grid.get(Coord<3>( 3,  2, -1)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 0,  0,  0), grid.get(Coord<3>( 0,  0, 20)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 4,  6,  0), grid.get(Coord<3>( 4,  6, 20)).pos);

        TS_ASSERT_EQUALS(Coord<3>( 9,  0,  0), grid.get(Coord<3>(-1,  0,  0)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 9,  5,  3), grid.get(Coord<3>(-1,  5,  3)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 0,  0,  0), grid.get(Coord<3>(10,  0,  0)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 0,  6,  7), grid.get(Coord<3>(10,  6,  7)).pos);

        TS_ASSERT_EQUALS(Coord<3>( 0, 39,  0), grid.get(Coord<3>( 0, -1,  0)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 2, 39,  6), grid.get(Coord<3>( 2, -1,  6)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 0,  0,  0), grid.get(Coord<3>( 0, 40,  0)).pos);
        TS_ASSERT_EQUALS(Coord<3>( 3,  0,  9), grid.get(Coord<3>( 3, 40,  9)).pos);
    }

    void testGetSetEdge()
    {
        TestCellSoA defaultCell(
            Coord<3>(101, 102, 103),
            Coord<3>(104, 105, 106),
            107,
            108);
        TestCellSoA edgeCell(
            Coord<3>(109, 110, 111),
            Coord<3>(112, 113, 114),
            115,
            116);

        Coord<3> dim(64, 32, 21);
        Coord<3> origin(12, 41, 12);
        CoordBox<3> box(origin, dim);

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box, defaultCell, edgeCell);
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(12, 41, 11)));
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(11, 41, 12)));
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(12, 40, 12)));
        TS_ASSERT_EQUALS(defaultCell, grid.get(Coord<3>(12, 41, 12)));

        edgeCell.testValue = 666;
        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(12, 41, 11)));
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(11, 41, 12)));
        TS_ASSERT_EQUALS(edgeCell,    grid.get(Coord<3>(12, 40, 12)));
        TS_ASSERT_EQUALS(defaultCell, grid.get(Coord<3>(12, 41, 12)));
    }

    void testScalarGetSet()
    {
        Coord<3> dim(100, 50, 30);
        Coord<3> origin(200, 210, 220);
        CoordBox<3> box(origin, dim);

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box);

        TestCellSoA expected0(
            Coord<3>(1, 2, 3),
            Coord<3>(4, 5, 6),
            7,
            8);
        TestCellSoA expected1(
            Coord<3>( 9, 10, 11),
            Coord<3>(12, 13, 14),
            15,
            16);
        TestCellSoA expected2(
            Coord<3>(17, 18, 19),
            Coord<3>(20, 21, 22),
            23,
            24);
        grid.set(Coord<3>(200, 210, 220), expected0);
        grid.set(Coord<3>(299, 210, 220), expected1);
        grid.set(Coord<3>(299, 259, 249), expected2);

        TestCellSoA actual;
        actual= grid.get(Coord<3>(200, 210, 220));
        TS_ASSERT_EQUALS(expected0, actual);
        actual= grid.get(Coord<3>(299, 210, 220));
        TS_ASSERT_EQUALS(expected1, actual);
        actual= grid.get(Coord<3>(299, 259, 249));
        TS_ASSERT_EQUALS(expected2, actual);
     }

    void testGetSetMultiple()
    {
        Coord<3> dim(123, 25, 63);
        Coord<3> origin(10, 10, 10);
        CoordBox<3> box(origin, dim);

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box);

        Region<3> region;
        region << Streak<3>(Coord<3>(10, 10, 10), 100)
               << Streak<3>(Coord<3>(20, 11, 10), 133)
               << Streak<3>(Coord<3>(10, 24, 72), 133);
        int counter = 0;

        for (Region<3>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            std::vector<TestCellSoA> cells;
            cells.reserve(i->length());
            for (int j = 0; j < i->length(); ++j) {
                cells << TestCellSoA(
                    Coord<3>(counter + 0,    counter + 1000, counter + 2000),
                    Coord<3>(counter + 3000, counter + 4000, counter + 5000),
                    counter + 6000,
                    counter + 7000);

                ++counter;
            }

            grid.set(*i, cells.data());
        }

        counter = 0;

        for (Region<3>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {

            std::vector<TestCellSoA> cells(i->length());
            grid.get(*i, cells.data());

            for (int j = 0; j < i->length(); ++j) {
                TestCellSoA expected(
                    Coord<3>(counter + 0,    counter + 1000, counter + 2000),
                    Coord<3>(counter + 3000, counter + 4000, counter + 5000),
                    counter + 6000,
                    counter + 7000);

                TS_ASSERT_EQUALS(cells[j], expected);
                ++counter;
            }
        }
    }

};

}
