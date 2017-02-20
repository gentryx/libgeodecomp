#include <libgeodecomp/config.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/storage/serializationbuffer.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <cxxtest/TestSuite.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdlib>

using namespace LibGeoDecomp;

#ifdef LIBGEODECOMP_WITH_CPP14
class MySoACell1
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit
    MySoACell1(int const val = 0) :
        val(val)
    {}

    inline bool operator==(const MySoACell1& other) const
    {

        return val == other.val;
    }

    inline bool operator!=(const MySoACell1& other) const
    {

        return val != other.val;
    }

    int val;
};

LIBFLATARRAY_REGISTER_SOA(MySoACell1, ((int)(val)))

class MySoACell2
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit
    MySoACell2(int x = 0, double y = 0, char z = 0) :
        x(x), y(y), z(z)
    {}

    inline bool operator==(const MySoACell2& other) const
    {
        return x == other.x &&
            y == other.y &&
            z == other.z;
    }

    inline bool operator!=(const MySoACell2& other) const
    {
        return !(*this == other);
    }

    int x;
    double y;
    char z;
};

LIBFLATARRAY_REGISTER_SOA(MySoACell2, ((int)(x))((double)(y))((char)(z)))
#endif

namespace LibGeoDecomp {

class UnstructuredSoAGridTest : public CxxTest::TestSuite
{
public:
    void testConstructor()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        MySoACell1 defaultCell(5);
        MySoACell1 edgeCell(-1);
        CoordBox<1> dim(Coord<1>(0), Coord<1>(100));

        UnstructuredSoAGrid<MySoACell1> grid(dim, defaultCell, edgeCell);

        for (int i = 0; i < 100; ++i) {
            TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), defaultCell);
        }

        TS_ASSERT_EQUALS(grid.getEdgeElement(), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(-1)), edgeCell);
#endif
    }

    void testGetAndSet()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredSoAGrid<MySoACell1> grid(CoordBox<1>(Coord<1>(), Coord<1>(10)));
        MySoACell1 elem1(1);
        MySoACell1 elem2(2);
        grid.set(Coord<1>(5), elem1);
        grid.set(Coord<1>(6), elem2);

        TS_ASSERT_EQUALS(grid.get(Coord<1>(5)), elem1);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(6)), elem2);
#endif
    }

    void testResize()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Coord<1> origin(0);
        Coord<1> dim(10);
        CoordBox<1> box(origin, dim);

        Region<1> boundingRegion;
        boundingRegion << box;
        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid(box);
        TS_ASSERT_EQUALS(box, grid.boundingBox());
        TS_ASSERT_EQUALS(boundingRegion, grid.boundingRegion());

        dim.x() = 100;
        box = CoordBox<1>(origin, dim);

        grid.resize(box);
        TS_ASSERT_EQUALS(box, grid.boundingBox());

        for (int i = 0; i < dim.x(); ++i) {
            UnstructuredTestCellSoA1 cell(i, 888, true);
            grid.set(Coord<1>(i), cell);
        }

        for (int i = origin.x(); i < (origin.x() + dim.x()); ++i) {
            UnstructuredTestCellSoA1 expected(i, 888, true);
            UnstructuredTestCellSoA1 actual = grid.get(Coord<1>(i));

            TS_ASSERT_EQUALS(expected, actual);
        }
#endif
    }

    void testLoadSaveRegionWithOffset()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Coord<1> dim(100);
        CoordBox<1> box(Coord<1>(), dim);
        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid(box);

        for (int i = 0; i < dim.x(); ++i) {
            grid.set(Coord<1>(i), UnstructuredTestCellSoA1(i, 4711, true));
        }

        Region<1> region;
        region << Streak<1>(Coord<1>( 0),   3)
               << Streak<1>(Coord<1>(10),  12)
               << Streak<1>(Coord<1>(20),  21)
               << Streak<1>(Coord<1>(96), 100);
        TS_ASSERT_EQUALS(10, region.size());

        Region<1> regionOffset10;
        regionOffset10 << Streak<1>(Coord<1>( 10),  13)
                       << Streak<1>(Coord<1>( 20),  22)
                       << Streak<1>(Coord<1>( 30),  31)
                       << Streak<1>(Coord<1>(106), 110);
        TS_ASSERT_EQUALS(10, regionOffset10.size());

        Region<1> regionOffset30;
        regionOffset30 << Streak<1>(Coord<1>( 30),  33)
                       << Streak<1>(Coord<1>( 40),  42)
                       << Streak<1>(Coord<1>( 50),  51)
                       << Streak<1>(Coord<1>(126), 130);
        TS_ASSERT_EQUALS(10, regionOffset30.size());

        std::vector<char> buffer;
        SerializationBuffer<UnstructuredTestCellSoA1>::resize(&buffer, region);

        grid.saveRegion(&buffer, regionOffset10, Coord<1>(-10));

        box.dimensions.x() = 200;
        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid2(box);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCellSoA1 actual = grid2.get(*i);
            UnstructuredTestCellSoA1 expected = grid.get(*i);

            TS_ASSERT_DIFFERS(actual, expected);
        }

        grid2.loadRegion(buffer, regionOffset30, Coord<1>(-30));

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCellSoA1 actual = grid2.get(*i);
            UnstructuredTestCellSoA1 expected = grid.get(*i);

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testLoadSaveRegion()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Coord<1> dim(100);
        CoordBox<1> box(Coord<1>(), dim);
        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid(box);

        for (int i = 0; i < dim.x(); ++i) {
            grid.set(Coord<1>(i), UnstructuredTestCellSoA1(i, 4711, true));
        }

        Region<1> region;
        region << Streak<1>(Coord<1>( 0),   3)
               << Streak<1>(Coord<1>(10),  12)
               << Streak<1>(Coord<1>(20),  21)
               << Streak<1>(Coord<1>(96), 100);
        TS_ASSERT_EQUALS(10, region.size());

        std::vector<char> buffer;
        SerializationBuffer<UnstructuredTestCellSoA1>::resize(&buffer, region);

        grid.saveRegion(&buffer, region);

        box.dimensions.x() = 200;
        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid2(box);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCellSoA1 actual = grid2.get(*i);
            UnstructuredTestCellSoA1 expected = grid.get(*i);

            TS_ASSERT_DIFFERS(actual, expected);
        }

        grid2.loadRegion(buffer, region);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCellSoA1 actual = grid2.get(*i);
            UnstructuredTestCellSoA1 expected = grid.get(*i);

            TS_ASSERT_EQUALS(actual, expected);
        }
#endif
    }

    void testSaveAndLoadMemberBasic()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Selector<MySoACell1> valSelector(&MySoACell1::val, "val");
        MySoACell1 defaultCell(5);
        MySoACell1 edgeCell(-1);
        CoordBox<1> dim(Coord<1>(), Coord<1>(100));
        UnstructuredSoAGrid<MySoACell1> grid(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 50)
               << Streak<1>(Coord<1>(50), 100);

        std::vector<int> valVector(region.size(), 0xdeadbeef);

        // copy default data back
        grid.saveMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], 5);
        }

        // modify a bit and test again
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            grid.set(Coord<1>(i), MySoACell1(i));
        }
        grid.saveMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], i);
        }

        // test load member
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            valVector[i] = -i;
        }
        grid.loadMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), MySoACell1(-i));
        }
#endif
    }

    void testSaveAndLoadMember()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Selector<MySoACell2> valSelector(&MySoACell2::y, "y");
        MySoACell2 defaultCell(5, 6, 7);
        MySoACell2 edgeCell(-1, -2, -3);
        CoordBox<1> dim(Coord<1>(), Coord<1>(100));
        UnstructuredSoAGrid<MySoACell2> grid(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 50)
               << Streak<1>(Coord<1>(50), 100);

        std::vector<double> valVector(region.size(), 0xdeadbeef);

        // copy default data back
        grid.saveMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], 6);
        }

        // modify a bit and test again
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            grid.set(Coord<1>(i), MySoACell2(i, i + 1, i + 2));
        }
        grid.saveMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], i + 1);
        }

        // test load member
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            grid.set(Coord<1>(i), defaultCell);
        }
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            valVector[i] = -i;
        }
        grid.loadMember(valVector.data(), MemoryLocation::HOST, valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), MySoACell2(5, -i, 7));
        }
#endif
    }

    void testOffset()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Coord<1> origin(1000);
        Coord<1> dim(50);
        CoordBox<1> box(origin, dim);

        UnstructuredSoAGrid<UnstructuredTestCellSoA1> grid(box);
        TS_ASSERT_EQUALS(box, grid.boundingBox());

        for (CoordBox<1>::Iterator i = box.begin(); i != box.end(); ++i) {
            UnstructuredTestCellSoA1 cell = grid.get(*i);
            cell.id = i->x();
            grid.set(*i, cell);
        }

        for (CoordBox<1>::Iterator i = box.begin(); i != box.end(); ++i) {
            UnstructuredTestCellSoA1 cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.id, i->x());
        }
#endif
    }
};

}
