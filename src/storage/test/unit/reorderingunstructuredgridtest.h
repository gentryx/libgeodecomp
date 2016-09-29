#include <iostream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/reorderingunstructuredgrid.h>

// fixme
#include <typeinfo>
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

    // fixme: also test AoS
    void testSetWeights()
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

        // GridType grid(region);
        // init.grid(&grid);

        // fixme: test get (coord, streak), set (coord, streak), saveRegion, load Region, saveMember, loadMember after init (involves reordering)
    }
};

}
