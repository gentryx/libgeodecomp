#include <iostream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/reorderingunstructuredgrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ReorderingUnstructuredGridTest : public CxxTest::TestSuite
{
public:
    typedef Topologies::Unstructured::Topology Topology;
    void testResize()
    {
        typedef UnstructuredTestCellSoA1 TestCell;
        typedef GridTypeSelector<TestCell, Topology, false, APITraits::TrueType>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> nodeSet;
        nodeSet << Streak<1>(Coord<1>(10), 20)
                << Streak<1>(Coord<1>(30), 35);

        GridType grid(nodeSet);
        TS_ASSERT_EQUALS(grid.boundingBox(), CoordBox<1>(Coord<1>(10), Coord<1>(25)));
        TS_ASSERT_EQUALS(grid.boundingRegion(), nodeSet);

        std::string expectedError = "Resize not supported ReorderingUnstructuredGrid";

        TS_ASSERT_THROWS_ASSERT(
            grid.resize(CoordBox<1>(Coord<1>(0), Coord<1>(100))),
            std::logic_error& exception,
            TS_ASSERT_SAME_DATA(
                expectedError.c_str(),
                exception.what(),
                expectedError.size()));
    }

    void testGetSetAoS()
    {
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(222), 301)
               << Streak<1>(Coord<1>(409), 412);

        GridType grid(region);
        TS_ASSERT_EQUALS(grid.boundingBox(), CoordBox<1>(Coord<1>(111), Coord<1>(301)));
        TS_ASSERT_EQUALS(grid.boundingRegion(), region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell(i->x(), i->x() * 1000 + 666, i->x() % 31, i->x() % 5);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell actual = grid.get(*i);
            TestCell expected(i->x(), i->x() * 1000 + 666, i->x() % 31, i->x() % 5);

            TS_ASSERT_EQUALS(actual, expected);
        }
    }

    void testGetSetSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>( 11),  16)
               << Streak<1>(Coord<1>(122), 201)
               << Streak<1>(Coord<1>(309), 310);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell(i->x(), i->x() * 1000 + 666, i->x() % 13, i->x() % 7);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell actual = grid.get(*i);
            TestCell expected(i->x(), i->x() * 1000 + 666, i->x() % 13, i->x() % 7);

            TS_ASSERT_EQUALS(actual, expected);
        }
    }

    void testGetSetStreaksAoS()
    {
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(222), 301)
               << Streak<1>(Coord<1>(409), 412);

        GridType grid(region);

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> cells;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                cells << TestCell(x, x * 1000 + 666, x % 39, x % 11);
            }
            grid.set(*i, &cells[0]);
        }

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> actual(i->length());
            grid.get(*i, &actual[0]);

            std::vector<TestCell> expected;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                expected << TestCell(x, x * 1000 + 666, x % 39, x % 11);
            }

            TS_ASSERT_EQUALS(actual, expected);
        }
    }

    void testGetSetStreaksSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(121), 126)
               << Streak<1>(Coord<1>(322), 391)
               << Streak<1>(Coord<1>(609), 662);

        GridType grid(region);

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> cells;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                cells << TestCell(x, x * 1000 + 555, x % 39, x % 11);
            }
            grid.set(*i, &cells[0]);
        }

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> actual(i->length());
            grid.get(*i, &actual[0]);

            std::vector<TestCell> expected;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                expected << TestCell(x, x * 1000 + 555, x % 39, x % 11);
            }

            TS_ASSERT_EQUALS(actual, expected);
        }
    }

    void testEdgeCellHandlingAoS()
    {
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(222), 301)
               << Streak<1>(Coord<1>(409), 412);
        GridType grid(region);

        TestCell edgeCell(4711);
        TS_ASSERT_DIFFERS(grid.getEdge(), edgeCell);

        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(grid.getEdge(), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(-13)), edgeCell);
    }

    void testEdgeCellHandlingSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(409), 412);
        GridType grid(region);

        TestCell edgeCell(54321);
        TS_ASSERT_DIFFERS(grid.getEdge(), edgeCell);

        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(grid.getEdge(), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(-32)), edgeCell);
    }

    void testLoadSaveRegionAoS()
    {
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region1;
        Region<1> region2;
        region1 << Streak<1>(Coord<1>(111), 166)
                << Streak<1>(Coord<1>(400), 490);
        region2 << Streak<1>(Coord<1>(  0), 150)
                << Streak<1>(Coord<1>(440), 500);
        Region<1> region3 = region1 & region2;

        GridType grid1(region1);
        GridType grid2(region2);

        for (Region<1>::Iterator i = region1.begin(); i != region1.end(); ++i) {
            grid1.set(*i, TestCell(-10));
        }
        for (Region<1>::Iterator i = region2.begin(); i != region2.end(); ++i) {
            grid2.set(*i, TestCell(-20));
        }

        int counter = 0;
        for (Region<1>::Iterator i = region1.begin(); i != region1.end(); ++i) {
            grid1.set(*i, TestCell(counter, 10000 + counter));
            ++counter;
        }

        std::vector<TestCell> buffer;
        SerializationBuffer<TestCell>::resize(&buffer, region3);

        grid1.saveRegion(&buffer, region3);
        grid2.loadRegion(buffer, region3);

        counter = 0;
        for (Region<1>::Iterator i = region2.begin(); i != region2.end(); ++i) {
            TestCell expected(-20);

            if (region3.count(*i)) {
                expected = TestCell(counter, 10000 + counter);
                ++counter;
                if (counter == 39) {
                    counter = 95;
                }
            }

            TestCell actual = grid2.get(*i);
            TS_ASSERT_EQUALS(expected, actual);
        }
    }

    void testLoadSaveRegionSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region1;
        Region<1> region2;
        region1 << Streak<1>(Coord<1>(111), 116)
                << Streak<1>(Coord<1>(409), 452);
        region2 << Streak<1>(Coord<1>(  0), 114)
                << Streak<1>(Coord<1>(440), 460);
        Region<1> region3 = region1 & region2;

        GridType grid1(region1);
        GridType grid2(region2);

        for (Region<1>::Iterator i = region1.begin(); i != region1.end(); ++i) {
            grid1.set(*i, TestCell(-1));
        }
        for (Region<1>::Iterator i = region2.begin(); i != region2.end(); ++i) {
            grid2.set(*i, TestCell(-2));
        }

        int counter = 0;
        for (Region<1>::Iterator i = region1.begin(); i != region1.end(); ++i) {
            grid1.set(*i, TestCell(counter, 10000 + counter));
            ++counter;
        }

        std::vector<char> buffer;
        SerializationBuffer<TestCell>::resize(&buffer, region3);

        grid1.saveRegion(&buffer, region3);
        grid2.loadRegion(buffer, region3);

        counter = 0;
        for (Region<1>::Iterator i = region2.begin(); i != region2.end(); ++i) {
            TestCell expected(-2);

            if (region3.count(*i)) {
                expected = TestCell(counter, 10000 + counter);
                ++counter;
                if (counter == 3) {
                    counter = 36;
                }
            }

            TestCell actual = grid2.get(*i);
            TS_ASSERT_EQUALS(expected, actual);
        }
    }

    void testLoadSaveMemberAoS()
    {
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(122), 177)
               << Streak<1>(Coord<1>(500), 599);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, TestCell(i->x(), 10 * i->x()));
        }

        Region<1> subset;
        subset << Streak<1>(Coord<1>(122), 150)
               << Streak<1>(Coord<1>(550), 599);

        std::vector<int> buf(subset.size(), -1);
        grid.saveMember(
            &buf[0],
            MemoryLocation::HOST,
            Selector<TestCell>(&TestCell::id, "id"),
            subset);

        int counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TS_ASSERT_EQUALS(i->x(), buf[counter]);
            ++counter;
        }

        for (int i = 0; i < buf.size(); ++i) {
            buf[i] = i;
        }

        grid.loadMember(
            (unsigned*)&buf[0],
            MemoryLocation::HOST,
            Selector<TestCell>(&TestCell::cycleCounter, "cycleCounter"),
            subset);

        counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(counter, cell.cycleCounter);
            ++counter;
        }
    }

    void testLoadSaveMemberSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(222), 277)
               << Streak<1>(Coord<1>(300), 399);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, TestCell(i->x(), 10 * i->x()));
        }

        Region<1> subset;
        subset << Streak<1>(Coord<1>(250), 277)
               << Streak<1>(Coord<1>(300), 350);

        std::vector<int> buf(subset.size(), -1);
        grid.saveMember(
            &buf[0],
            MemoryLocation::HOST,
            Selector<TestCell>(&TestCell::id, "id"),
            subset);

        int counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TS_ASSERT_EQUALS(i->x(), buf[counter]);
            ++counter;
        }

        for (int i = 0; i < buf.size(); ++i) {
            buf[i] = i;
        }

        grid.loadMember(
            (unsigned*)&buf[0],
            MemoryLocation::HOST,
            Selector<TestCell>(&TestCell::cycleCounter, "cycleCounter"),
            subset);

        counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(counter, cell.cycleCounter);
            ++counter;
        }
    }

    // fixme: also test AoS
    void testSetWeightsSoA()
    {
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value DelegateGrid;
        typedef ReorderingUnstructuredGrid<DelegateGrid> GridType;

        UnstructuredTestInitializer<TestCell> init(1234, 66);

        Region<1> region;
        region << Streak<1>(Coord<1>( 11),  44)
               << Streak<1>(Coord<1>(100), 140)
               << Streak<1>(Coord<1>(211), 214)
               << Streak<1>(Coord<1>(333), 344)
               << Streak<1>(Coord<1>(355), 366);

        GridType grid(region);
        init.grid(&grid);

        std::cout << "========================================================================================\n";
        for (int i = 0; i < 64; ++i) {
            std::vector<std::pair<int, double> > row = grid.delegate.getWeights(0).getRow(i);
            std::cout << "t: " << i << " =  " << grid.delegate.get(Coord<1>(i)).id << " -> " << row << " -- " <<  row.size() << "\n";
        }
        // fixme: test get (coord, streak), set (coord, streak), saveRegion, load Region, saveMember, loadMember after init (involves reordering)
    }
};

}
