#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

DECLARE_MULTI_CONTAINER_CELL(DummyContainer,            \
                             ((std::string)(5)(labels)) \
                             ((double)(7)(prices))      \
                             )

class SimpleNode
{
public:
    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        std::cout << "SimpleNode::update()\n";
    }
};

class SimpleElement
{
public:
    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        std::cout << "SimpleElement::update()\n";
    }
};

DECLARE_MULTI_CONTAINER_CELL(SimpleContainer,                   \
                             ((SimpleNode)(30)(nodes))          \
                             ((SimpleElement)(10)(elements))    \
                             )

class MultiContainerCellTest : public CxxTest::TestSuite
{
public:
    void testConstructionAndAccess()
    {
        DummyContainer cell;
        cell.labels.insert(10, "foo");
        cell.labels.insert(11, "bar");
        cell.labels.insert(15, "goo");

        cell.prices.insert(10, -666);
        cell.prices.insert(11, -0.11);
        cell.prices.insert(12, -0.12);
        cell.prices.insert(13, -0.13);
        cell.prices.insert(10, -0.10);

        TS_ASSERT_EQUALS(cell.labels.size(), std::size_t(3));
        TS_ASSERT_EQUALS(*cell.labels[10], "foo");
        TS_ASSERT_EQUALS(*cell.labels[11], "bar");
        TS_ASSERT_EQUALS(*cell.labels[15], "goo");

        TS_ASSERT_EQUALS(cell.prices.size(), std::size_t(4));
        TS_ASSERT_EQUALS(*cell.prices[10], -0.10);
        TS_ASSERT_EQUALS(*cell.prices[11], -0.11);
        TS_ASSERT_EQUALS(*cell.prices[12], -0.12);
        TS_ASSERT_EQUALS(*cell.prices[13], -0.13);
    }

    void testUpdate()
    {
        Coord<2> dim(10, 5);
        Grid<SimpleContainer> gridOld(dim);
        Grid<SimpleContainer> gridNew(dim);

        SimpleContainer c;
        c.nodes.insert(1, SimpleNode());
        c.nodes.insert(5, SimpleNode());
        gridOld[Coord<2>(3, 4)] = c;

        SimpleContainer d;
        d.nodes.insert(6, SimpleNode());
        d.elements.insert(1, SimpleElement());
        d.elements.insert(7, SimpleElement());
        d.elements.insert(9, SimpleElement());
        gridOld[Coord<2>(2, 4)] = d;

        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        UpdateFunctor<SimpleContainer>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);
    }
};

}
