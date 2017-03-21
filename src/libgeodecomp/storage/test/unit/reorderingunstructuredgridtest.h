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
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCell;
        typedef GridTypeSelector<TestCell, Topology, false, APITraits::TrueType>::Value GridType;

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
#endif
    }

    void testGetSetAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(222), 301)
               << Streak<1>(Coord<1>(409), 412);

        GridType grid(region);
        TS_ASSERT_EQUALS(grid.boundingBox(), CoordBox<1>(Coord<1>(111), Coord<1>(301)));
        TS_ASSERT_EQUALS(grid.boundingRegion(), region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell(
                i->x(),
                static_cast<unsigned>(i->x() * 1000 + 666),
                i->x() % 31,
                i->x() % 5);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell actual = grid.get(*i);
            TestCell expected(
                i->x(),
                static_cast<unsigned>(i->x() * 1000 + 666),
                i->x() % 31,
                i->x() % 5);

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testGetSetSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>( 11),  16)
               << Streak<1>(Coord<1>(122), 201)
               << Streak<1>(Coord<1>(309), 310);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell(
                i->x(),
                static_cast<unsigned>(i->x() * 1000 + 666),
                i->x() % 13,
                i->x() % 7);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell actual = grid.get(*i);
            TestCell expected(
                i->x(),
                static_cast<unsigned>(i->x() * 1000 + 666),
                i->x() % 13,
                i->x() % 7);

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testGetSetStreaksAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(222), 301)
               << Streak<1>(Coord<1>(409), 412);

        GridType grid(region);

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> cells;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                cells << TestCell(
                    x,
                    static_cast<unsigned>(x * 1000 + 666),
                    x % 39,
                    x % 11);
            }
            grid.set(*i, &cells[0]);
        }

        for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            std::vector<TestCell> actual(static_cast<std::size_t>(i->length()));
            grid.get(*i, &actual[0]);

            std::vector<TestCell> expected;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                expected << TestCell(
                    x,
                    static_cast<unsigned>(x * 1000 + 666),
                    x % 39,
                    x % 11);
            }

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testGetSetStreaksSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

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
            std::vector<TestCell> actual(static_cast<std::size_t>(i->length()));
            grid.get(*i, &actual[0]);

            std::vector<TestCell> expected;
            for (int x = i->origin.x(); x != i->endX; ++x) {
                expected << TestCell(
                    x,
                    static_cast<unsigned>(x * 1000 + 555),
                    x % 39,
                    x % 11);
            }

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testEdgeCellHandlingAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

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
#endif
    }

    void testEdgeCellHandlingSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(111), 116)
               << Streak<1>(Coord<1>(409), 412);
        GridType grid(region);

        TestCell edgeCell(54321);
        TS_ASSERT_DIFFERS(grid.getEdge(), edgeCell);

        grid.setEdge(edgeCell);
        TS_ASSERT_EQUALS(grid.getEdge(), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(-32)), edgeCell);
#endif
    }

    void testLoadSaveRegionAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

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
            grid1.set(*i, TestCell(counter, 10000 + counter, false, false));
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
                expected = TestCell(counter, 10000 + counter, false, false);
                ++counter;
                if (counter == 39) {
                    counter = 95;
                }
            }

            TestCell actual = grid2.get(*i);
            TS_ASSERT_EQUALS(expected, actual);
        }
#endif
    }

    void testLoadSaveRegionSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

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
            grid1.set(*i, TestCell(counter, 10000 + counter, false, false));
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
                expected = TestCell(counter, 10000 + counter, false, false);
                ++counter;
                if (counter == 3) {
                    counter = 36;
                }
            }

            TestCell actual = grid2.get(*i);
            TS_ASSERT_EQUALS(expected, actual);
        }
#endif
    }

    void testLoadSaveMemberAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(122), 177)
               << Streak<1>(Coord<1>(500), 599);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, TestCell(i->x(), 10 * i->x(), false, false));
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

        unsigned counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TS_ASSERT_EQUALS(i->x(), buf[counter]);
            ++counter;
        }

        for (std::size_t i = 0; i < buf.size(); ++i) {
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
#endif
    }

    void testLoadSaveMemberSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        Region<1> region;
        region << Streak<1>(Coord<1>(222), 277)
               << Streak<1>(Coord<1>(300), 399);

        GridType grid(region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid.set(*i, TestCell(i->x(), 10 * i->x(), false, false));
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

        unsigned counter = 0;
        for (Region<1>::Iterator i = subset.begin(); i != subset.end(); ++i) {
            TS_ASSERT_EQUALS(i->x(), buf[counter]);
            ++counter;
        }

        for (std::size_t i = 0; i < buf.size(); ++i) {
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
#endif
    }

    void testSetWeightsAoS()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCell<> TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        UnstructuredTestInitializer<TestCell> init(1234, 66);

        Region<1> region;
        region << Streak<1>(Coord<1>( 11),  44)
               << Streak<1>(Coord<1>(100), 140)
               << Streak<1>(Coord<1>(211), 214)
               << Streak<1>(Coord<1>(333), 344)
               << Streak<1>(Coord<1>(355), 450);

        GridType grid(region);
        init.grid(&grid);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.id, i->x());

            cell.cycleCounter = static_cast<unsigned>(cell.id);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.cycleCounter, i->x());
        }

        // test Streak-based get/set
        {
            std::vector<TestCell> buf;
            int counter = 42195;

            for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                buf.resize(static_cast<std::size_t>(i->length()));

                grid.get(*i, &buf[0]);
                for (std::size_t j = 0; j < buf.size(); ++j) {
                    int expectedID = i->origin.x() + static_cast<int>(j);
                    TS_ASSERT_EQUALS(expectedID, buf[j].id);

                    buf[j].id = counter;
                    ++counter;
                }

                grid.set(*i, &buf[0]);
            }

            counter = 42195;

            for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                buf.resize(static_cast<std::size_t>(i->length()));

                grid.get(*i, &buf[0]);
                for (std::size_t j = 0; j < buf.size(); ++j) {
                    TS_ASSERT_EQUALS(counter, buf[j].id);
                    ++counter;
                }
            }
        }

        // test load/save member
        {
            std::vector<unsigned> buf(region.size());
            std::size_t counter = 0;

            grid.saveMember(&buf[0], MemoryLocation::HOST, Selector<TestCell>(&TestCell::cycleCounter, "cycleCounter"), region);

            for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
                TS_ASSERT_EQUALS(i->x(), buf[counter]);
                ++counter;
            }

            counter = 2000;
            for (std::size_t i = 0; i < buf.size(); ++i) {
                buf[i] = counter;
                ++counter;
            }
            grid.loadMember((int*)&buf[0], MemoryLocation::HOST, Selector<TestCell>(&TestCell::id, "id"), region);

            counter = 2000;
            for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
                TS_ASSERT_EQUALS(grid.get(*i).id, static_cast<int>(counter));
                ++counter;
            }
        }

        // test load/save Region
        {
            Region<1> region2;
            region2 << Streak<1>(Coord<1>( 90), 150)
                    << Streak<1>(Coord<1>(200), 400);

            Region<1> intersection = region & region2;
            GridType grid2(region2);
            init.grid(&grid2);

            std::vector<TestCell> buffer;
            SerializationBuffer<TestCell>::resize(&buffer, intersection);

            grid.saveRegion(&buffer, intersection);
            grid2.loadRegion(buffer, intersection);

            for (Region<1>::Iterator i = intersection.begin(); i != intersection.end(); ++i) {
                TestCell actual = grid2.get(*i);
                TestCell expected = grid.get(*i);
                TS_ASSERT_EQUALS(actual, expected);
            }
        }

        // test remapRegion()
        Region<1> remappedRegion = grid.remapRegion(region);
        TS_ASSERT_EQUALS(remappedRegion.size(), region.size());

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            cell.id = i->x();
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = remappedRegion.begin(); i != remappedRegion.end(); ++i) {
            TestCell cell = grid.delegate.get(*i);

            std::vector<std::pair<int, double> > expectedRow;
            int numNeighbors = cell.id % 20 + 1;
            for (int j = 0; j != numNeighbors; ++j) {
                int neighborID = cell.id + 1 + j;
                double weight = neighborID + 0.1;

                std::vector<std::pair<int, int> >::const_iterator iter = std::find_if(
                    grid.logicalToPhysicalIDs.begin(),
                    grid.logicalToPhysicalIDs.end(),
                    [neighborID](const std::pair<int, double>& pair){
                        return pair.first == neighborID;
                    });

                if (iter == grid.logicalToPhysicalIDs.end()) {
                    expectedRow.clear();
                    break;
                }

                expectedRow << std::make_pair(iter->second, weight);
            }

            std::stable_sort(
                expectedRow.begin(),
                expectedRow.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.first < b.first;
                });

            std::vector<std::pair<int, double> > actualRow = grid.delegate.matrices[0].getRow(i->x());

            TS_ASSERT_EQUALS(actualRow, expectedRow);
        }

        // test expandChunksInRemappedRegion()
        remappedRegion.clear();
        remappedRegion << Streak<1>(Coord<1>(  9),  20)
                       << Streak<1>(Coord<1>( 48),  51)
                       << Streak<1>(Coord<1>( 70),  80)
                       << Streak<1>(Coord<1>( 90), 100)
                       << Streak<1>(Coord<1>(102), 119);

        Region<1> actual = grid.expandChunksInRemappedRegion(remappedRegion);

        Region<1> expected;
        expected << Streak<1>(Coord<1>(  8),  20)
                 << Streak<1>(Coord<1>( 48),  52)
                 << Streak<1>(Coord<1>( 68),  80)
                 << Streak<1>(Coord<1>( 88), 120);

        TS_ASSERT_EQUALS(expected, actual);
#endif
    }

    void testSetWeightsSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCell;
        typedef APITraits::SelectSoA<TestCell>::Value SoAFlag;
        typedef GridTypeSelector<TestCell, Topology, false, SoAFlag>::Value GridType;

        UnstructuredTestInitializer<TestCell> init(1234, 66);

        Region<1> region;
        region << Streak<1>(Coord<1>( 11),  44)
               << Streak<1>(Coord<1>(100), 140)
               << Streak<1>(Coord<1>(211), 214)
               << Streak<1>(Coord<1>(333), 344)
               << Streak<1>(Coord<1>(355), 450);

        GridType grid(region);
        init.grid(&grid);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.id, i->x());

            cell.cycleCounter = static_cast<unsigned>(cell.id);
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.cycleCounter, i->x());
        }

        // test Streak-based get/set
        {
            std::vector<TestCell> buf;
            int counter = 0;

            for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                buf.resize(static_cast<std::size_t>(i->length()));

                grid.get(*i, &buf[0]);
                for (std::size_t j = 0; j < buf.size(); ++j) {
                    int expectedID = i->origin.x() + static_cast<int>(j);
                    TS_ASSERT_EQUALS(expectedID, buf[j].id);

                    buf[j].id = counter;
                    ++counter;
                }

                grid.set(*i, &buf[0]);
            }

            counter = 0;

            for (Region<1>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                buf.resize(static_cast<std::size_t>(i->length()));

                grid.get(*i, &buf[0]);
                for (std::size_t j = 0; j < buf.size(); ++j) {
                    TS_ASSERT_EQUALS(counter, buf[j].id);
                    ++counter;
                }
            }
        }

        // test load/save member
        {
            std::vector<unsigned> buf(region.size());
            unsigned counter = 0;

            grid.saveMember(&buf[0], MemoryLocation::HOST, Selector<TestCell>(&TestCell::cycleCounter, "cycleCounter"), region);

            for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
                TS_ASSERT_EQUALS(i->x(), buf[counter]);
                ++counter;
            }

            counter = 1000;
            for (std::size_t i = 0; i < buf.size(); ++i) {
                buf[i] = counter;
                ++counter;
            }
            grid.loadMember((int*)&buf[0], MemoryLocation::HOST, Selector<TestCell>(&TestCell::id, "id"), region);

            counter = 1000;
            for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
                TS_ASSERT_EQUALS(grid.get(*i).id, static_cast<int>(counter));
                ++counter;
            }
        }

        // test load/save Region
        {
            Region<1> region2;
            region2 << Streak<1>(Coord<1>( 90), 150)
                    << Streak<1>(Coord<1>(200), 400);

            Region<1> intersection = region & region2;
            GridType grid2(region2);
            init.grid(&grid2);

            std::vector<char> buffer;
            SerializationBuffer<TestCell>::resize(&buffer, intersection);

            grid.saveRegion(&buffer, intersection);
            grid2.loadRegion(buffer, intersection);

            for (Region<1>::Iterator i = intersection.begin(); i != intersection.end(); ++i) {
                TestCell actual = grid2.get(*i);
                TestCell expected = grid.get(*i);
                TS_ASSERT_EQUALS(actual, expected);
            }
        }

        // test remapRegion()
        Region<1> remappedRegion = grid.remapRegion(region);
        TS_ASSERT_EQUALS(remappedRegion.size(), region.size());

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCell cell = grid.get(*i);
            cell.id = i->x();
            grid.set(*i, cell);
        }

        for (Region<1>::Iterator i = remappedRegion.begin(); i != remappedRegion.end(); ++i) {
            TestCell cell = grid.delegate.get(*i);

            std::vector<std::pair<int, double> > expectedRow;
            int numNeighbors = cell.id % 20 + 1;
            for (int j = 0; j != numNeighbors; ++j) {
                int neighborID = cell.id + 1 + j;
                double weight = neighborID + 0.1;

                std::vector<std::pair<int, int> >::const_iterator iter = std::find_if(
                    grid.logicalToPhysicalIDs.begin(),
                    grid.logicalToPhysicalIDs.end(),
                    [neighborID](const std::pair<int, double>& pair){
                        return pair.first == neighborID;
                    });

                if (iter == grid.logicalToPhysicalIDs.end()) {
                    expectedRow.clear();
                    break;
                }

                expectedRow << std::make_pair(iter->second, weight);
            }

            std::stable_sort(
                expectedRow.begin(),
                expectedRow.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.first < b.first;
                });

            std::vector<std::pair<int, double> > actualRow = grid.delegate.matrices[0].getRow(i->x());

            TS_ASSERT_EQUALS(actualRow, expectedRow);
        }

        // test expandChunksInRemappedRegion()
        remappedRegion.clear();
        remappedRegion << Streak<1>(Coord<1>( 10),  20)
                       << Streak<1>(Coord<1>( 48),  51)
                       << Streak<1>(Coord<1>( 70),  80)
                       << Streak<1>(Coord<1>( 90), 100)
                       << Streak<1>(Coord<1>(102), 120);

        Region<1> actual = grid.expandChunksInRemappedRegion(remappedRegion);

        Region<1> expected;
        expected << Streak<1>(Coord<1>(  8),  24)
                 << Streak<1>(Coord<1>( 48),  56)
                 << Streak<1>(Coord<1>( 64),  80)
                 << Streak<1>(Coord<1>( 88), 120);

        TS_ASSERT_EQUALS(expected, actual);
#endif
    }
};

}
