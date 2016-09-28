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
        TS_ASSERT_EQUALS(grid.boundingBox(), box);

        Region<3> boundingRegion;
        boundingRegion << box;
        TS_ASSERT_EQUALS(boundingRegion, grid.boundingRegion());
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

    void testResize()
    {
        Coord<3> origin(20, 21, 22);
        Coord<3> dim(30, 20, 10);
        CoordBox<3> box(origin, dim);

        TestCellSoA innerCell;
        TestCellSoA edgeCell;
        innerCell.isEdgeCell = false;
        edgeCell.isEdgeCell = true;

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box, innerCell, edgeCell);

        int maxX = dim.x() + 2;
        int maxY = dim.y() + 2;
        int maxZ = dim.z() + 2;

        for (int z = 0; z < maxZ; ++z) {
            for (int y = 0; y < maxY; ++y) {
                for (int x = 0; x < maxX; ++x) {
                    TestCellSoA expected = innerCell;

                    if ((x < 1) || (x > (maxX - 2)) ||
                        (y < 1) || (y > (maxY - 2)) ||
                        (z < 1) || (z > (maxZ - 2))) {
                        expected = edgeCell;
                    }

                    TestCellSoA actual = grid.delegate.get(x, y, z);

                    TS_ASSERT_EQUALS(actual, expected);
                }
            }
        }

        origin = Coord<3>(30, 31, 32);
        dim = Coord<3>(40, 50, 60);
        box = CoordBox<3>(origin, dim);
        grid.resize(box);
        TS_ASSERT_EQUALS(box, grid.boundingBox());

        maxX = dim.x() + 2;
        maxY = dim.y() + 2;
        maxZ = dim.z() + 2;

        for (int z = 0; z < maxZ; ++z) {
            for (int y = 0; y < maxY; ++y) {
                for (int x = 0; x < maxX; ++x) {
                    if ((x < 1) || (x > (maxX - 2)) ||
                        (y < 1) || (y > (maxY - 2)) ||
                        (z < 1) || (z > (maxZ - 2))) {
                        TestCellSoA actual = grid.delegate.get(x, y, z);
                        TS_ASSERT_EQUALS(actual, edgeCell);
                    }

                }
            }
        }
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

    void testLoadSaveRegion()
    {
        Coord<3> dim(23, 25, 63);
        Coord<3> origin(10, 10, 10);
        CoordBox<3> box1(origin, dim);
        CoordBox<3> box2(
            Coord<3>( 3,  2,  1),
            Coord<3>(30, 33, 72));

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid1(box1);
        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid2(box2);

        Region<3> region;
        region << box1;

        int counter = 0;
        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            grid1.set(
                *i,
                TestCellSoA(
                    Coord<3>(counter +    1, counter + 1001, counter + 2001),
                    Coord<3>(counter + 3001, counter + 4001, counter + 5001),
                    counter + 6001,
                    counter + 7001));
            ++counter;
        }

        region.clear();
        region << Streak<3>(Coord<3>(10, 10, 10), 30)
               << Streak<3>(Coord<3>(10, 11, 10), 33)
               << Streak<3>(Coord<3>(15, 34, 72), 33);

        std::vector<char> buffer(
            SoAGrid<TestCellSoA, Topologies::Cube<3>::Topology>::AGGREGATED_MEMBER_SIZE *
            region.size());

        grid1.saveRegion(&buffer, region);
        grid2.loadRegion( buffer, region);

        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            Coord<3> relativeCoord = *i - origin;
            counter = relativeCoord.toIndex(dim);

            TestCellSoA actual = grid1.get(*i);
            TestCellSoA expected(
                Coord<3>(counter +    1, counter + 1001, counter + 2001),
                Coord<3>(counter + 3001, counter + 4001, counter + 5001),
                counter + 6001,
                counter + 7001);

            TS_ASSERT_EQUALS(expected, actual);
        }
    }

    void testLoadSaveMember()
    {
        Selector<TestCellSoA> testValSelector(&TestCellSoA::testValue, "testValue");
        Selector<TestCellSoA> posSelector(    &TestCellSoA::pos,       "pos");

        Coord<3> dim(23, 25, 63);
        Coord<3> origin(10, 10, 10);
        CoordBox<3> box1(origin, dim);
        CoordBox<3> box2(
            Coord<3>( 3,  2,  1),
            Coord<3>(30, 33, 72));

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid1(box1);
        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid2(box2);

        Region<3> region;
        region << box1;

        int counter = 0;
        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            grid1.set(
                *i,
                TestCellSoA(
                    Coord<3>(counter +    5, counter + 1005, counter + 2005),
                    Coord<3>(counter + 3005, counter + 4005, counter + 5005),
                    counter + 6005,
                    counter + 7005));
            ++counter;
        }

        region.clear();
        region << Streak<3>(Coord<3>(10, 10, 10), 30)
               << Streak<3>(Coord<3>(10, 11, 11), 33)
               << Streak<3>(Coord<3>(15, 11, 11), 33);

        // check testVal:
        std::vector<double> testValVec(region.size());

        grid1.saveMember(
            testValVec.data(),
            MemoryLocation::HOST,
            testValSelector,
            region);

        counter = 0;
        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            Coord<3> relativeCoord = *i - origin;
            int expectedTestValue = relativeCoord.toIndex(dim) + 7005;
            TS_ASSERT_EQUALS(testValVec[counter], expectedTestValue);
            ++counter;
        }
        grid2.loadMember(
            testValVec.data(),
            MemoryLocation::HOST,
            testValSelector,
            region);

        counter = 0;
        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TestCellSoA cell = grid2.get(*i);

            if (region.count(*i)) {
                TS_ASSERT_EQUALS(cell.testValue, testValVec[counter]);
                ++counter;
            } else {
                TS_ASSERT_EQUALS(cell.testValue, 666);
            }
        }

        // check pos:
        LibFlatArray::cuda_array<Coord<3> > cudaPosVec(region.size());
        std::vector<Coord<3> > posVec(region.size());

        grid1.saveMember(
            cudaPosVec.data(),
            MemoryLocation::CUDA_DEVICE,
            posSelector,
            region);
        cudaPosVec.save(posVec.data());

        counter = 0;
        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            Coord<3> relativeCoord = *i - origin;
            int index = relativeCoord.toIndex(dim);
            Coord<3> expectedPos(index + 5, index + 1005, index + 2005);
            TS_ASSERT_EQUALS(posVec[counter], expectedPos);
            ++counter;
        }
        grid2.loadMember(
            cudaPosVec.data(),
            MemoryLocation::CUDA_DEVICE,
            posSelector,
            region);

        counter = 0;
        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TestCellSoA cell = grid2.get(*i);

            if (region.count(*i)) {
                TS_ASSERT_EQUALS(cell.pos, posVec[counter]);
                ++counter;
            } else {
                TS_ASSERT_EQUALS(cell.pos, Coord<3>());
            }
        }
    }

};

}
