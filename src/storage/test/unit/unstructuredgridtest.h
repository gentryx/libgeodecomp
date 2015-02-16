#include <libgeodecomp/storage/unstructuredgrid.h>
#include <cxxtest/TestSuite.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdlib>

class MyDummyElement
{
public:
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

private:
    int val;
};

std::ostream& operator<< (std::ostream& out, MyDummyElement const & val){
    out << val();
    return out;
}


using namespace LibGeoDecomp;

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

};

}
