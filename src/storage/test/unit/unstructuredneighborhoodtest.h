#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>

#include <map>
#include <algorithm>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyCell
{
public:
    double value;

    explicit MyCell(double val = 0.0) :
        value(val)
    {}
    bool operator==(const MyCell& rhs)
    {
        return value == rhs.value;
    }
    bool operator!=(const MyCell& rhs)
    {
        return !(*this == rhs);
    }
};

class UntructuredNeighborhoodTest : public CxxTest::TestSuite
{
public:
    void testSquareBracketsOperator()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<MyCell, 1, double, 1, 1> grid(Coord<1>(4), MyCell(), MyCell());

        // init elements
        grid[0] = MyCell(0.5);
        grid[1] = MyCell(1.5);
        grid[2] = MyCell(2.5);
        grid[3] = MyCell(3.5);

        // init neighborhood
        UnstructuredNeighborhood<MyCell, 1, double, 1, 1> nb(grid, 0);

        // test access via cell id
        TS_ASSERT_EQUALS(nb[0], MyCell(0.5));
        TS_ASSERT_EQUALS(nb[1], MyCell(1.5));
        TS_ASSERT_EQUALS(nb[2], MyCell(2.5));
        TS_ASSERT_EQUALS(nb[3], MyCell(3.5));
#endif
    }

    void testNeighborhoodSimple()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<MyCell, 1, double, 1, 1> grid(Coord<1>(4), MyCell(), MyCell());

        // init elements
        grid[0] = MyCell(0.5);
        grid[1] = MyCell(1.5);
        grid[2] = MyCell(2.5);
        grid[3] = MyCell(3.5);

        // init adjacency
        // matrix:
        // 1 0 0 0
        // 0 2 0 0
        // 0 0 3 0
        // 0 0 0 4
        std::map<Coord<2>, double> adjacency;
        adjacency[Coord<2>(0, 0)] = 1;
        adjacency[Coord<2>(1, 1)] = 2;
        adjacency[Coord<2>(2, 2)] = 3;
        adjacency[Coord<2>(3, 3)] = 4;
        grid.setAdjacency(0, 4, adjacency);

        // init neighborhood
        UnstructuredNeighborhood<MyCell, 1, double, 1, 1> nb(grid, 0);

        // every cell has one neighbor
        auto weights0 = nb.weights();
        ++nb;
        auto weights1 = nb.weights();
        ++nb;
        auto weights2 = nb.weights();
        ++nb;
        auto weights3 = nb.weights();
        TS_ASSERT_EQUALS(1, std::distance(weights0.begin(), weights0.end()));
        TS_ASSERT_EQUALS(1, std::distance(weights1.begin(), weights1.end()));
        TS_ASSERT_EQUALS(1, std::distance(weights2.begin(), weights2.end()));
        TS_ASSERT_EQUALS(1, std::distance(weights3.begin(), weights3.end()));

        // check cells and weights
        auto it0 = *weights0.begin();
        auto it1 = *weights1.begin();
        auto it2 = *weights2.begin();
        auto it3 = *weights3.begin();
        TS_ASSERT_EQUALS(it0.first, 0);
        TS_ASSERT_EQUALS(it1.first, 1);
        TS_ASSERT_EQUALS(it2.first, 2);
        TS_ASSERT_EQUALS(it3.first, 3);

        TS_ASSERT_EQUALS(it0.second, 1);
        TS_ASSERT_EQUALS(it1.second, 2);
        TS_ASSERT_EQUALS(it2.second, 3);
        TS_ASSERT_EQUALS(it3.second, 4);
#endif
    }

    void testNeighborhood()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<MyCell, 1, double, 1, 1> grid(Coord<1>(8), MyCell(), MyCell());

        // init adjacency
        // matrix:
        // 0 0 0 0 0 0 0 0
        // 0 2 0 3 7 6 4 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0
        std::map<Coord<2>, double> adjacency;
        adjacency[Coord<2>(1, 1)] = 2;
        adjacency[Coord<2>(1, 3)] = 3;
        adjacency[Coord<2>(1, 4)] = 7;
        adjacency[Coord<2>(1, 5)] = 6;
        adjacency[Coord<2>(1, 6)] = 4;
        grid.setAdjacency(0, 8, adjacency);

        // init neighborhood
        UnstructuredNeighborhood<MyCell, 1, double, 1, 1> nb(grid, 0);

        // check neighborhood for cell 0
        for (const auto& i: nb.weights()) {
            (void)i;
            // should never executed
            TS_ASSERT_EQUALS(1, 2);
        }

        // check neighborhood for cell 1
        ++nb;
        std::size_t cnt(0);
        auto weights1 = nb.weights();
        TS_ASSERT_EQUALS(5, std::distance(weights1.begin(), weights1.end()));
        for (const auto& i: nb.weights()) {
            switch (cnt) {
            case 0:
                TS_ASSERT_EQUALS(i.first, 1);
                TS_ASSERT_EQUALS(i.second, 2);
                break;
            case 1:
                TS_ASSERT_EQUALS(i.first, 3);
                TS_ASSERT_EQUALS(i.second, 3);
                break;
            case 2:
                TS_ASSERT_EQUALS(i.first, 4);
                TS_ASSERT_EQUALS(i.second, 7);
                break;
            case 3:
                TS_ASSERT_EQUALS(i.first, 5);
                TS_ASSERT_EQUALS(i.second, 6);
                break;
            case 4:
                TS_ASSERT_EQUALS(i.first, 6);
                TS_ASSERT_EQUALS(i.second, 4);
                break;
            }
            ++cnt;
        }
#endif
    }
};

}
