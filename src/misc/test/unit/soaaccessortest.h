#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/soaaccessor.h>
#include <libgeodecomp/misc/supervector.h>

class StupidCell
{
public:
    StupidCell(int x=0, int y=0, int z=0) :
        c(x),
        a(y),
        b(z)
    {}

    inline bool operator==(const StupidCell& other)
    {
        return
            (c == other.c) &&
            (a == other.a) &&
            (b == other.b);
    }

    double c;
    int a;
    char b;
};

LIBGEODECOMP_REGISTER_SOA(StupidCell, ((double)(c))((int)(a))((char)(b)))

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SoAAccessorTest : public CxxTest::TestSuite
{
public:
    typedef Grid<StupidCell, Topologies::Cube<3>::Topology> GridType;

    void testCopyInOut()
    {
        const int DIM_X = 32;
        const int DIM_Y = 17;
        const int DIM_Z = 43;
        SuperVector<char> storage(DIM_X * DIM_Y * DIM_Z *
                                  (sizeof(double) +
                                   sizeof(int) +
                                   sizeof(char)));
        int index = 0;
        SoAAccessor<StupidCell, DIM_X, DIM_Y, DIM_Z, 0> a(
            &storage[0], &index);

        Coord<3> dim(DIM_X, DIM_Y, DIM_Z);
        CoordBox<3> box(Coord<3>(), dim);

        GridType grid(dim);
        fillGrid(&grid);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            index = i->x() + i->y() * DIM_X + i->z() * DIM_X * DIM_Y;
            a = grid[*i];
        }

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            index = i->x() + i->y() * DIM_X + i->z() * DIM_X * DIM_Y;
            StupidCell c;
            c << a;
            TS_ASSERT_EQUALS(c, grid[*i]);
        }
    }

private:
    void fillGrid(GridType *grid) {
        CoordBox<3> box = grid->boundingBox();
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            (*grid)[*i] = StupidCell(i->x(), i->y(), i->z());
        }
    }
};

}
