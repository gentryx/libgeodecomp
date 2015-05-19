#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/misc/apitraits.h>
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
        Coord<1> dim(100);

        UnstructuredSoAGrid<MySoACell1> grid(dim, defaultCell, edgeCell);

        for (int i = 0; i < 100; ++i) {
            TS_ASSERT_EQUALS(grid[i], defaultCell);
        }

        TS_ASSERT_EQUALS(grid.getEdgeElement(), edgeCell);
        TS_ASSERT_EQUALS(grid[-1], edgeCell);
#endif
    }

    void testGetAndSet()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredSoAGrid<MySoACell1> grid(Coord<1>(10));
        MySoACell1 elem1(1);
        MySoACell1 elem2(2);
        grid.set(Coord<1>(5), elem1);
        grid.set(Coord<1>(6), elem2);

        TS_ASSERT_EQUALS(grid.get(Coord<1>(5)), elem1);
        TS_ASSERT_EQUALS(grid.get(Coord<1>(6)), elem2);
#endif
    }

    void testSaveAndLoadMemberBasic()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        Selector<MySoACell1> valSelector(&MySoACell1::val, "val");
        MySoACell1 defaultCell(5);
        MySoACell1 edgeCell(-1);
        Coord<1> dim(100);
        UnstructuredSoAGrid<MySoACell1> grid(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 50)
               << Streak<1>(Coord<1>(50), 100);

        std::vector<int> valVector(region.size(), 0xdeadbeef);

        // copy default data back
        grid.saveMember(valVector.data(), valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], 5);
        }

        // modify a bit and test again
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            grid.set(Coord<1>(i), MySoACell1(i));
        }
        grid.saveMember(valVector.data(), valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], i);
        }

        // test load member
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            valVector[i] = -i;
        }
        grid.loadMember(valVector.data(), valSelector, region);
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
        Coord<1> dim(100);
        UnstructuredSoAGrid<MySoACell2> grid(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 50)
               << Streak<1>(Coord<1>(50), 100);

        std::vector<double> valVector(region.size(), 0xdeadbeef);

        // copy default data back
        grid.saveMember(valVector.data(), valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(valVector[i], 6);
        }

        // modify a bit and test again
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            grid.set(Coord<1>(i), MySoACell2(i, i + 1, i + 2));
        }
        grid.saveMember(valVector.data(), valSelector, region);
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
        grid.loadMember(valVector.data(), valSelector, region);
        for (int i = 0; i < static_cast<int>(region.size()); ++i) {
            TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), MySoACell2(5, -i, 7));
        }
#endif
    }
};

}
