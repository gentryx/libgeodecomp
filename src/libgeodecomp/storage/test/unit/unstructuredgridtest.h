#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <cxxtest/TestSuite.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdlib>

using namespace LibGeoDecomp;

class MyDummyElement
{
public:
    explicit
    MyDummyElement(int const val = 0) :
        val(val)
    {}

    int& operator()(int newVal)
    {
        val = newVal;
        return val;
    }

    int operator()() const
    {
        return val;
    }

    int& operator()()
    {
        return val;
    }

    inline bool operator==(const MyDummyElement& other) const
    {

        return val == other.val;
    }

    inline bool operator!=(const MyDummyElement& other) const
    {

        return val != other.val;
    }

private:
    int val;
};

std::ostream& operator<< (std::ostream& out, const MyDummyElement& val)
{
    out << val();
    return out;
}

namespace LibGeoDecomp {

class UnstructuredGridTest : public CxxTest::TestSuite
{
#ifdef LIBGEODECOMP_WITH_CPP14
    UnstructuredGrid<MyDummyElement>* testGrid;
#endif
public:
    void setUp()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        testGrid = new UnstructuredGrid<MyDummyElement>(Coord<1>(10));

        for (int i = 0; i < testGrid->getDimensions().x(); ++i) {
            (*testGrid)[Coord<1>(i)] = MyDummyElement(i);
        }
#endif
    }

    void tearDown()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        delete testGrid;
#endif
    }


    void testDefaultConstructor()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<MyDummyElement> g;
        TS_ASSERT_EQUALS(0, (int)g.getDimensions().x());
#endif
    }

    void testConstructorDefaultInit()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<int> g(Coord<1>(10), 1, -1);
        TS_ASSERT_EQUALS(Coord<1>(10), g.getDimensions());
        TS_ASSERT_EQUALS(1,  g[4 ]);
        TS_ASSERT_EQUALS(-1, g[11]);
#endif
    }

    void testOperatorEqual1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        int dim = testGrid->getDimensions().x();
        UnstructuredGrid<MyDummyElement> other =
            UnstructuredGrid<MyDummyElement>(Coord<1>(dim));

        for (int i = 0; i < dim; ++i) {
            other[i] = testGrid->get(Coord<1>(i));
        }

        TS_ASSERT(*testGrid == other);

        other[2](-100);
        TS_ASSERT(*testGrid != other);
#endif
    }

    void testAssimentOperator()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<int> a (Coord<1>(100));
        for (int i = 0; i < 100; ++i) {
            a[Coord<1>(i)] = i * 200;
        }

        UnstructuredGrid<int> b;
        b = a;

        TS_ASSERT_EQUALS(a, a);
        TS_ASSERT_EQUALS(a, b);

        b[Coord<1>(55)] = -666;
        TS_ASSERT_DIFFERS(a, b);
#endif
    }

    void testSetterGetter()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<int> a(Coord<1>(100));
        for (int i = 0; i < 100; ++i) {
            a[Coord<1>(i)] = i * 200;
        }

        UnstructuredGrid<int> b;
        b = a;

        TS_ASSERT_EQUALS(a, a);
        TS_ASSERT_EQUALS(a, b);

        b.set(Coord<1>(55),-666);
        TS_ASSERT_DIFFERS(a, b);

        TS_ASSERT_EQUALS(-666, b.get(Coord<1>(55)));
#endif
    }

    void testEdgeCell()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<int> foo(Coord<1>(100), 1, -1);
        TS_ASSERT_EQUALS(foo.getEdge(), -1);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)], -1);

        foo[Coord<1>(-1)] = -2;
        TS_ASSERT_EQUALS(foo.getEdge(), -2);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)],-2);

        foo.setEdge(-3);
        TS_ASSERT_EQUALS(foo.getEdge(), -3);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)],-3);
#endif
    }

    void testWeightsMatrix()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 128;
        typedef UnstructuredGrid<int, 2, double, 4, 1> GridType;
        GridType *grid = new GridType(Coord<1>(DIM));
        GridType::SparseMatrix weights0;
        GridType::SparseMatrix weights1;
        GridType::SparseMatrix rawMatrix0;
        GridType::SparseMatrix rawMatrix1;
        SellCSigmaSparseMatrixContainer<double,4,1> matrix0 (DIM);
        SellCSigmaSparseMatrixContainer<double,4,1> matrix1 (DIM);

        for (int i = 0; i < DIM; ++i) {
            grid->set(Coord<1>(i), i);

            weights0   << std::make_pair(Coord<2>(i,abs(i*57)      % DIM),  i      / DIM);
            weights0   << std::make_pair(Coord<2>(i,abs(i*57 + 75) % DIM),  i * 57 / DIM);
            weights0   << std::make_pair(Coord<2>(i,abs(i*57 - 7 ) % DIM),  i *  7 / DIM);
            rawMatrix0 << std::make_pair(Coord<2>(i,abs(i*57)      % DIM),  i      / DIM);
            rawMatrix0 << std::make_pair(Coord<2>(i,abs(i*57 + 75) % DIM),  i * 57 / DIM);
            rawMatrix0 << std::make_pair(Coord<2>(i,abs(i*57 - 7 ) % DIM),  i * 7  / DIM);

            weights1   << std::make_pair(Coord<2>(i,abs(i*57)      % DIM), -i      / DIM);
            weights1   << std::make_pair(Coord<2>(i,abs(i*57 + 75) % DIM), -i * 57 / DIM);
            weights1   << std::make_pair(Coord<2>(i,abs(i*57 - 7 ) % DIM), -i *  7 / DIM);
            rawMatrix1 << std::make_pair(Coord<2>(i,abs(i*57)      % DIM), -i      / DIM);
            rawMatrix1 << std::make_pair(Coord<2>(i,abs(i*57 + 75) % DIM), -i * 57 / DIM);
            rawMatrix1 << std::make_pair(Coord<2>(i,abs(i*57 - 7 ) % DIM), -i *  7 / DIM);
        }

        matrix0.initFromMatrix(rawMatrix0);
        matrix1.initFromMatrix(rawMatrix1);
        grid->setWeights(0, weights0);
        grid->setWeights(1, weights1);

        TS_ASSERT_EQUALS(matrix0, grid->getWeights(0));
        TS_ASSERT_EQUALS(matrix1, grid->getWeights(1));

        delete grid;
#endif
    }

    void testLoadSaveRegion()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<UnstructuredTestCell<> > grid(Coord<1>(100));

        for (int i = 0; i < 100; ++i) {
            grid[Coord<1>(i)] = UnstructuredTestCell<>(i, 4711, true);
        }

        std::vector<UnstructuredTestCell<> > buffer(10);
        Region<1> region;
        region << Streak<1>(Coord<1>( 0),   3)
               << Streak<1>(Coord<1>(10),  12)
               << Streak<1>(Coord<1>(20),  21)
               << Streak<1>(Coord<1>(96), 100);
        TS_ASSERT_EQUALS(10, region.size());

        grid.saveRegion(&buffer, region);

        int index = 0;
        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCell<> actual = buffer[index];
            UnstructuredTestCell<> expected(i->x(), 4711, true);

            TS_ASSERT_EQUALS(actual, expected);
            ++index;
        }

        for (std::size_t i = 0; i < buffer.size(); ++i) {
            buffer[i].id = 1000000 + i;
            buffer[i].cycleCounter = 777;
        }

        UnstructuredGrid<UnstructuredTestCell<> > grid2(Coord<1>(100));
        grid2.loadRegion(buffer, region);

        index = 0;
        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCell<> actual = grid2.get(*i);
            UnstructuredTestCell<> expected(
                1000000 + index,
                777,
                true);

            TS_ASSERT_EQUALS(actual, expected);
            ++index;
        }
#endif
    }

    void testLoadSaveRegionWithOffset()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<UnstructuredTestCell<> > grid(Coord<1>(100));

        for (int i = 0; i < 100; ++i) {
            grid[Coord<1>(i)] = UnstructuredTestCell<>(i, 4711, true);
        }

        std::vector<UnstructuredTestCell<> > buffer(10);
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

        Region<1> regionOffset20;
        regionOffset20 << Streak<1>(Coord<1>( 20),  23)
                       << Streak<1>(Coord<1>( 30),  32)
                       << Streak<1>(Coord<1>( 40),  41)
                       << Streak<1>(Coord<1>(116), 120);
        TS_ASSERT_EQUALS(10, regionOffset20.size());

        grid.saveRegion(&buffer, regionOffset10, Coord<1>(-10));

        int index = 0;
        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCell<> actual = buffer[index];
            UnstructuredTestCell<> expected(i->x(), 4711, true);

            TS_ASSERT_EQUALS(actual, expected);
            ++index;
        }

        for (std::size_t i = 0; i < buffer.size(); ++i) {
            buffer[i].id = 1010101 + i;
            buffer[i].cycleCounter = 8252;
        }

        UnstructuredGrid<UnstructuredTestCell<> > grid2(Coord<1>(100));
        grid2.loadRegion(buffer, regionOffset20, Coord<1>(-20));

        index = 0;
        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            UnstructuredTestCell<> actual = grid2.get(*i);
            UnstructuredTestCell<> expected(
                1010101 + index,
                8252,
                true);

            TS_ASSERT_EQUALS(actual, expected);
            ++index;
        }
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
        UnstructuredGrid<UnstructuredTestCellSoA1> grid(dim);
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

    void testResizeFromZero()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredGrid<UnstructuredTestCellSoA1> grid;
        grid.resize(CoordBox<1>(Coord<1>(0), Coord<1>(10)));
#endif
    }
};

}
