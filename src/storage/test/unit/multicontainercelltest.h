#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

DECLARE_MULTI_CONTAINER_CELL(
    DummyContainer,
    DummyContainer,
    (((ContainerCell<std::string, 5>))(labels))
    (((ContainerCell<double,      7>))(prices)) )

typedef std::vector<std::pair<std::string, std::string> > LogType;
LogType multiContainerCellTestLog;

class SimpleElement;

class SimpleNode
{
public:
    explicit SimpleNode(const std::string& cargo = "") :
        cargo(cargo)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep);

    std::string cargo;
};

class SimpleElement
{
public:
    explicit SimpleElement(const std::string& cargo = "") :
        cargo(cargo)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        for (int i = 0; i < 20; ++i) {
            try {
                const SimpleNode& node = hood.nodes[i];
                multiContainerCellTestLog << std::make_pair(cargo, node.cargo);
            } catch(const std::logic_error& exception) {
                // intentionally left blank
            }
        }

        for (int i = 0; i < 20; ++i) {
            try {
                const SimpleElement& element = hood.elements[i];
                multiContainerCellTestLog << std::make_pair(cargo, element.cargo);
            } catch(const std::logic_error& exception) {
                // intentionally left blank
            }
        }
    }

    std::string cargo;
};

template<typename NEIGHBORHOOD>
void SimpleNode::update(const NEIGHBORHOOD& hood, int nanoStep)
{
    for (int i = 0; i < 20; ++i) {
        try {
            const SimpleNode& node = hood.nodes[i];
            multiContainerCellTestLog << std::make_pair(cargo, node.cargo);
        } catch(const std::logic_error& exception) {
            // intentionally left blank
        }
    }

    for (int i = 0; i < 20; ++i) {
        try {
            const SimpleElement& element = hood.elements[i];
            multiContainerCellTestLog << std::make_pair(cargo, element.cargo);
        } catch(const std::logic_error& exception) {
            // intentionally left blank
        }
    }
}

DECLARE_MULTI_CONTAINER_CELL(
    SimpleContainer,
    SimpleContainer,
    (((ContainerCell<SimpleNode,    30>))(nodes))
    (((ContainerCell<SimpleElement, 10>))(elements)) )


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
        c.nodes.insert(1, SimpleNode("Node1"));
        c.nodes.insert(5, SimpleNode("Node5a"));
        gridOld[Coord<2>(3, 3)] = c;

        SimpleContainer d;
        d.nodes.insert(6, SimpleNode("Node6"));
        d.elements.insert(1, SimpleElement("Element1"));
        d.elements.insert(7, SimpleElement("Element7"));
        d.elements.insert(9, SimpleElement("Element9"));
        gridOld[Coord<2>(3, 4)] = d;

        SimpleContainer e;
        e.nodes.insert(10, SimpleNode("Node10"));
        e.nodes.insert(11, SimpleNode("Node11"));
        e.elements.insert(5, SimpleElement("Element5b"));
        gridOld[Coord<2>(8, 2)] = e;

        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        UpdateFunctor<SimpleContainer>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);

        LogType expectedLog;
        expectedLog << std::make_pair("Node10", "Node10")
                    << std::make_pair("Node10", "Node11")
                    << std::make_pair("Node10", "Element5b");

        expectedLog << std::make_pair("Node11", "Node10")
                    << std::make_pair("Node11", "Node11")
                    << std::make_pair("Node11", "Element5b");

        expectedLog << std::make_pair("Element5b", "Node10")
                    << std::make_pair("Element5b", "Node11")
                    << std::make_pair("Element5b", "Element5b");

        expectedLog << std::make_pair("Node1",  "Node1")
                    << std::make_pair("Node1",  "Node5a")
                    << std::make_pair("Node1",  "Node6")
                    << std::make_pair("Node1",  "Element1")
                    << std::make_pair("Node1",  "Element7")
                    << std::make_pair("Node1",  "Element9");

        expectedLog << std::make_pair("Node5a", "Node1")
                    << std::make_pair("Node5a", "Node5a")
                    << std::make_pair("Node5a", "Node6")
                    << std::make_pair("Node5a", "Element1")
                    << std::make_pair("Node5a", "Element7")
                    << std::make_pair("Node5a", "Element9");

        expectedLog << std::make_pair("Node6",  "Node1")
                    << std::make_pair("Node6",  "Node5a")
                    << std::make_pair("Node6",  "Node6")
                    << std::make_pair("Node6",  "Element1")
                    << std::make_pair("Node6",  "Element7")
                    << std::make_pair("Node6",  "Element9");

        expectedLog << std::make_pair("Element1", "Node1")
                    << std::make_pair("Element1", "Node5a")
                    << std::make_pair("Element1", "Node6")
                    << std::make_pair("Element1", "Element1")
                    << std::make_pair("Element1", "Element7")
                    << std::make_pair("Element1", "Element9");

        expectedLog << std::make_pair("Element7", "Node1")
                    << std::make_pair("Element7", "Node5a")
                    << std::make_pair("Element7", "Node6")
                    << std::make_pair("Element7", "Element1")
                    << std::make_pair("Element7", "Element7")
                    << std::make_pair("Element7", "Element9");

        expectedLog << std::make_pair("Element9", "Node1")
                    << std::make_pair("Element9", "Node5a")
                    << std::make_pair("Element9", "Node6")
                    << std::make_pair("Element9", "Element1")
                    << std::make_pair("Element9", "Element7")
                    << std::make_pair("Element9", "Element9");

        TS_ASSERT_EQUALS(expectedLog, multiContainerCellTestLog);
    }
};

}
