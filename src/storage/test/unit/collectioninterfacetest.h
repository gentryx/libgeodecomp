#include <libgeodecomp/storage/collectioninterface.h>
#include <libgeodecomp/storage/containercell.h>
#include <libgeodecomp/storage/multicontainercell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ArrayCell
{
public:
    typedef double value_type;
    typedef double* iterator;
    typedef const double* const_iterator;

    const double *begin() const
    {
        return data + 0;
    }

    double *begin()
    {
        return data + 0;
    }

    const double *end() const
    {
        return data + 10;
    }

    double *end()
    {
        return data + 10;
    }

    std::size_t size() const
    {
        return 10;
    }

private:
    double data[10];
};

class SimpleCellA
{
public:
    SimpleCellA() :
        foo(++counter)
    {}

    int foo;
    static int counter;
};

class SimpleCellB
{
public:
    SimpleCellB() :
        bar(++counter + 0.5)
    {}

    double bar;
    static int counter;
};

int SimpleCellA::counter = 0;
int SimpleCellB::counter = 0;

DECLARE_MULTI_CONTAINER_CELL(MultiCell,                         \
                             ((SimpleCellA)(30)(cellA))         \
                             ((SimpleCellB)(50)(cellB)) )

class CollectionInterfaceTest : public CxxTest::TestSuite
{
public:

    void testPassThrough1()
    {
        ArrayCell cell;
        CollectionInterface::PassThrough<ArrayCell> interface;

        TS_ASSERT_EQUALS(cell.begin(),    interface.begin(cell));
        TS_ASSERT_EQUALS(cell.end(),      interface.end(cell));
        TS_ASSERT_EQUALS(std::size_t(10), interface.size(cell));
    }

    void testPassThrough2()
    {
        typedef ContainerCell<SimpleCellA, 20, int> ContainerCellType;

        ContainerCellType cell;
        cell.insert( 1, SimpleCellA());
        cell.insert( 2, SimpleCellA());
        cell.insert( 3, SimpleCellA());
        cell.insert( 5, SimpleCellA());
        cell.insert( 7, SimpleCellA());
        cell.insert(11, SimpleCellA());

        CollectionInterface::PassThrough<ContainerCellType> interface;

        // foo values are not 1 and 6, but 21 and 26, as the container
        // cell will init its internal array with 20 cells upon creation.
        TS_ASSERT_EQUALS(21,             interface.begin(cell)->foo);
        TS_ASSERT_EQUALS(26,            (interface.end(cell) - 1)->foo);
        TS_ASSERT_EQUALS(std::size_t(6), interface.size(cell));
    }

    void testDelegate()
    {
        MultiCell cell;
        cell.cellA.insert(20, SimpleCellA());
        cell.cellA.insert(30, SimpleCellA());
        cell.cellA.insert(40, SimpleCellA());
        cell.cellA.insert(50, SimpleCellA());

        cell.cellB.insert(40, SimpleCellB());
        cell.cellB.insert(50, SimpleCellB());
        cell.cellB.insert(60, SimpleCellB());

        CollectionInterface::Delegate<MultiCell, ContainerCell<SimpleCellA, 30, int> > interfaceA(&MultiCell::cellA);
        CollectionInterface::Delegate<MultiCell, ContainerCell<SimpleCellB, 50, int> > interfaceB(&MultiCell::cellB);

        TS_ASSERT_EQUALS(57,             interfaceA.begin(cell)->foo);
        TS_ASSERT_EQUALS(60,            (interfaceA.end(cell) - 1)->foo);
        TS_ASSERT_EQUALS(std::size_t(4), interfaceA.size(cell));

        TS_ASSERT_EQUALS(51.5, interfaceB.begin(cell)->bar);
        TS_ASSERT_EQUALS(53.5, (interfaceB.end(cell) - 1)->bar);
        TS_ASSERT_EQUALS(3,  interfaceB.size(cell));
    }
};

}
