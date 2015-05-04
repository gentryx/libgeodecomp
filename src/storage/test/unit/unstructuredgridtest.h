#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/misc/apitraits.h>
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
    class API :
        public APITraits::HasSoA
    {};

    explicit
    MyDummyElement(int const val = 0) :
        val(val)
    {}

    int& operator()(const int& newVal)
    {
        val = newVal;
        return val;
    }

    const int operator()() const
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

    int val;
};

std::ostream& operator<< (std::ostream& out, MyDummyElement const & val){
    out << val();
    return out;
}

LIBFLATARRAY_REGISTER_SOA(MyDummyElement, ((int)(val)))

class MySoACell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit
    MySoACell(int x = 0, double y = 0, char z = 0) :
        x(x), y(y), z(z)
    {}

    inline bool operator==(const MySoACell& other) const
    {
        return x == other.x &&
            y == other.y &&
            z == other.z;
    }

    inline bool operator!=(const MySoACell& other) const
    {
        return !(*this == other);
    }

    int x;
    double y;
    char z;
};

LIBFLATARRAY_REGISTER_SOA(MySoACell, ((int)(x))((double)(y))((char)(z)))

namespace LibGeoDecomp {

class UnstructuredGridTest : public CxxTest::TestSuite
{
    UnstructuredGrid<MyDummyElement>* testGrid;
public:
    void setUp(){
        testGrid = new UnstructuredGrid<MyDummyElement>(Coord<1>(10));

        for (int i=0; i < testGrid->getDimensions().x(); ++i){
            (*testGrid)[Coord<1>(i)] = MyDummyElement(i);
        }
    }

    void tearDown(){
        delete testGrid;
    }


    void testDefaultConstructor(){
        UnstructuredGrid<MyDummyElement> g;
        TS_ASSERT_EQUALS(0, (int)g.getDimensions().x());
    }

    void testConstructorDefaultInit(){
        UnstructuredGrid<int> g (Coord<1>(10), 1, -1);
        TS_ASSERT_EQUALS(Coord<1>(10), g.getDimensions());
        TS_ASSERT_EQUALS(1,  g[4 ]);
        TS_ASSERT_EQUALS(-1, g[11]);
    }

    void testOperatorEqual1(){
        int dim = testGrid->getDimensions().x();
        UnstructuredGrid<MyDummyElement> other =
            UnstructuredGrid<MyDummyElement>(Coord<1>(dim));

        for(int i = 0; i < dim; ++i) {
            other[i] = testGrid->get(Coord<1>(i));
        }

        TS_ASSERT(*testGrid == other);

        other[2](-100);
        TS_ASSERT(*testGrid != other);
    }

    void testAssimentOperator(){
        UnstructuredGrid<int> a (Coord<1>(100));
        for (int i = 0; i<100; i++){
            a[Coord<1>(i)] = i * 200;
        }

        UnstructuredGrid<int> b;
        b = a;

        TS_ASSERT_EQUALS(a,a);
        TS_ASSERT_EQUALS(a,b);

        b[Coord<1>(55)] = -666;
        TS_ASSERT_DIFFERS(a,b);
    }

    void testSetterGetter(){
        UnstructuredGrid<int> a (Coord<1>(100));
        for (int i = 0; i<100; i++){
            a[Coord<1>(i)] = i * 200;
        }

        UnstructuredGrid<int> b;
        b = a;

        TS_ASSERT_EQUALS(a,a);
        TS_ASSERT_EQUALS(a,b);

        b.set(Coord<1>(55),-666);
        TS_ASSERT_DIFFERS(a,b);

        TS_ASSERT_EQUALS(-666,b.get(Coord<1>(55)));

    }

    void testEdgeCell(){
        UnstructuredGrid<int> foo (Coord<1>(100), 1, -1);
        TS_ASSERT_EQUALS(foo.getEdgeElement(), -1);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)], -1);

        foo[Coord<1>(-1)] = -2;
        TS_ASSERT_EQUALS(foo.getEdgeElement(), -2);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)],-2);

        foo.setEdge(-3);
        TS_ASSERT_EQUALS(foo.getEdge(), -3);
        TS_ASSERT_EQUALS(foo[Coord<1>(-1)],-3);
    }

    void testAdjacencyMatrix(){
        const int DIM = 128;
        UnstructuredGrid<int, 2, double, 4, 1> *grid =
            new UnstructuredGrid<int, 2, double, 4, 1>(Coord<1>(DIM));
        std::map<Coord<2>,int> adjacenc0;
        std::map<Coord<2>,int> adjacenc1;
        SellCSigmaSparseMatrixContainer<double,4,1> matrix0 (DIM);
        SellCSigmaSparseMatrixContainer<double,4,1> matrix1 (DIM);

        for (int i=0; i<DIM; ++i){
            grid->set(Coord<1>(i), i);

            adjacenc0 [Coord<2>(i,abs(i*57)     %DIM)] = i   /DIM;
            adjacenc0 [Coord<2>(i,abs(i*57 + 75)%DIM)] = i*57/DIM;
            adjacenc0 [Coord<2>(i,abs(i*57 - 7 )%DIM)] = i*7 /DIM;
            matrix0.addPoint   (i,abs(i*57)     %DIM   , i   /DIM);
            matrix0.addPoint   (i,abs(i*57 + 75)%DIM   , i*57/DIM);
            matrix0.addPoint   (i,abs(i*57 - 7 )%DIM   , i*7 /DIM);

            adjacenc1 [Coord<2>(i,abs(i*57)     %DIM)] = -i   /DIM;
            adjacenc1 [Coord<2>(i,abs(i*57 + 75)%DIM)] = -i*57/DIM;
            adjacenc1 [Coord<2>(i,abs(i*57 - 7 )%DIM)] = -i*7 /DIM;
            matrix1.addPoint   (i,abs(i*57)     %DIM   , -i   /DIM);
            matrix1.addPoint   (i,abs(i*57 + 75)%DIM   , -i*57/DIM);
            matrix1.addPoint   (i,abs(i*57 - 7 )%DIM   , -i*7 /DIM);
        }

        grid->setAdjacency(0, adjacenc0.begin(), adjacenc0.end());
        grid->setAdjacency(1, adjacenc1.begin(), adjacenc1.end());

        TS_ASSERT_EQUALS(matrix0, grid->getAdjacency(0));
        TS_ASSERT_EQUALS(matrix1, grid->getAdjacency(1));

        delete grid;
    }

    void testUnstructuredGridSoABasic()
    {
        // test constructor
        {
            MyDummyElement defaultCell(5);
            MyDummyElement edgeCell(-1);
            Coord<1> dim(100);

            UnstructuredGridSoA<MyDummyElement> grid(dim, defaultCell, edgeCell);

            for (int i = 0; i < 100; ++i) {
                TS_ASSERT_EQUALS(grid[i], defaultCell);
            }

            TS_ASSERT_EQUALS(grid.getEdgeElement(), edgeCell);
            TS_ASSERT_EQUALS(grid[-1], edgeCell);
        }

        // test set and get
        {
            UnstructuredGridSoA<MyDummyElement> grid(Coord<1>(10));
            MyDummyElement elem1(1);
            MyDummyElement elem2(2);
            grid.set(Coord<1>(5), elem1);
            grid.set(Coord<1>(6), elem2);

            TS_ASSERT_EQUALS(grid.get(Coord<1>(5)), elem1);
            TS_ASSERT_EQUALS(grid.get(Coord<1>(6)), elem2);
        }

        // test save and load member with one member
        {
            Selector<MyDummyElement> valSelector(&MyDummyElement::val, "val");
            MyDummyElement defaultCell(5);
            MyDummyElement edgeCell(-1);
            Coord<1> dim(100);
            UnstructuredGridSoA<MyDummyElement> grid(dim, defaultCell, edgeCell);

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
                grid.set(Coord<1>(i), MyDummyElement(i));
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
                TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), MyDummyElement(-i));
            }
        }

        // test save and load member with a little more complex cell
        {
            Selector<MySoACell> valSelector(&MySoACell::y, "y");
            MySoACell defaultCell(5, 6, 7);
            MySoACell edgeCell(-1, -2, -3);
            Coord<1> dim(100);
            UnstructuredGridSoA<MySoACell> grid(dim, defaultCell, edgeCell);

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
                grid.set(Coord<1>(i), MySoACell(i, i + 1, i + 2));
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
                TS_ASSERT_EQUALS(grid.get(Coord<1>(i)), MySoACell(5, -i, 7));
            }
        }
    }
};

}
