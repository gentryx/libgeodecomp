#include <libgeodecomp/storage/containercell.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/meshlessadapter.h>
#include <libgeodecomp/misc/testhelper.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MockCell
{
public:
    typedef Topologies::Cube<2>::Topology Topology;

    explicit MockCell(int id=0, std::vector<int> *ids=0) :
        id(id),
        ids(ids)
    {}

    template<class NEIGHBORHOOD>
    void update(NEIGHBORHOOD neighbors, int nanoStep)
    {
        for (std::vector<int>::iterator i = ids->begin();
             i != ids->end();
             ++i) {
            TS_ASSERT_EQUALS(*i, neighbors[*i].id);
        }

        int buffer = 0;
        TS_ASSERT_THROWS(buffer = neighbors[4711].id, std::logic_error&);
        if (buffer != 0) {
            id -= 10000;
        }
        id -= 4000;
    }

    int id;
    std::vector<int> *ids;
};

class ContainerCellTest : public CxxTest::TestSuite
{
public:
    void testCopy()
    {
        ContainerCell<MockCell, 5> container1;
        ContainerCell<MockCell, 5> container2;

        container1.insert(10, MockCell(10));
        container1.insert(11, MockCell(11));
        container1.insert(12, MockCell(12));

        TS_ASSERT_EQUALS(container1.size(), std::size_t(3));
        TS_ASSERT_EQUALS(container1[10]->id, 10);
        TS_ASSERT_EQUALS(container1[11]->id, 11);
        TS_ASSERT_EQUALS(container1[12]->id, 12);

        container2.insert(50, MockCell(50));
        container2.insert(51, MockCell(51));
        container2.insert(52, MockCell(52));
        container2.insert(53, MockCell(53));
        container2.insert(54, MockCell(54));

        TS_ASSERT_EQUALS(container2.size(), std::size_t(5));
        TS_ASSERT_EQUALS(container2[50]->id, 50);
        TS_ASSERT_EQUALS(container2[51]->id, 51);
        TS_ASSERT_EQUALS(container2[52]->id, 52);
        TS_ASSERT_EQUALS(container2[53]->id, 53);
        TS_ASSERT_EQUALS(container2[54]->id, 54);

        container2 = container1;

        container1.remove(10);
        container1.remove(11);
        container1.remove(12);
        container1.insert(50, MockCell(50));
        container1.insert(51, MockCell(51));
        container1.insert(52, MockCell(52));
        container1.insert(53, MockCell(53));
        container1.insert(54, MockCell(54));

        TS_ASSERT_EQUALS(container2.size(), std::size_t(3));
        TS_ASSERT_EQUALS(container2[10]->id, 10);
        TS_ASSERT_EQUALS(container2[11]->id, 11);
        TS_ASSERT_EQUALS(container2[12]->id, 12);

        TS_ASSERT_EQUALS(container1.size(), std::size_t(5));
        MockCell *nilCell = 0;
        TS_ASSERT_EQUALS(container1[10], nilCell);
    }

    void testInsertAndSearch()
    {
        ContainerCell<MockCell, 5> container;
        std::vector<int> ids;
        ids << 1 << 2 << 4 << 5 << 6;

        container.insert(2, MockCell(2, &ids));
        container.insert(1, MockCell(1, &ids));
        container.insert(6, MockCell(6, &ids));
        container.insert(5, MockCell(5, &ids));
        container.insert(4, MockCell(4, &ids));
        container.insert(4, MockCell(4, &ids));

        TS_ASSERT_EQUALS(std::size_t(5), container.size());
        TS_ASSERT_THROWS(container.insert(47, MockCell(47, 0)), std::logic_error&);
        TS_ASSERT_THROWS(container.insert(3,  MockCell(3 , 0)), std::logic_error&);


        for (int i = 0; i < 5; ++i) {
            TS_ASSERT_EQUALS(ids[i], container.ids[i]);
            TS_ASSERT_EQUALS(ids[i], container.cells[i].id);
        }

        TS_ASSERT_EQUALS(container[-1], (void*)0);
        TS_ASSERT_EQUALS(container[ 3], (void*)0);
        TS_ASSERT_EQUALS(container[ 9], (void*)0);

        for (int i = 0; i < 5; ++i)
            TS_ASSERT_EQUALS(container.cells + i, container[ids[i]]);
    }

    void testInsertAtEnd()
    {
        ContainerCell<MockCell, 5> container;
        std::vector<int> ids;
        ids << 1 << 2 << 4 << 5 << 6;

        container.insert(2, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(1), container.size());

        container.insert(4, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(2), container.size());
        container.insert(4, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(2), container.size());
        container.insert(4, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(2), container.size());

        container.insert(5, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(3), container.size());

        container.insert(6, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(4), container.size());
        container.insert(7, MockCell(2, &ids));
        TS_ASSERT_EQUALS(std::size_t(5), container.size());

        TS_ASSERT_THROWS(container.insert(8, MockCell(2, &ids)), std::logic_error&);
    }

    void testRemove()
    {
        ContainerCell<MockCell, 5> container;
        std::vector<int> ids;
        ids << 1 << 2 << 6 << 7;

        container.insert(2, MockCell(2, &ids));
        container.insert(1, MockCell(1, &ids));
        container.insert(6, MockCell(6, &ids));
        container.insert(5, MockCell(5, &ids));
        container.insert(7, MockCell(7, &ids));
        container.remove(5);

        for (int i = 0; i < 4; ++i) {
            TS_ASSERT_EQUALS(ids[i], container.ids[i]);
            TS_ASSERT_EQUALS(ids[i], container.cells[i].id);
        }
    }

    void testClear()
    {
        ContainerCell<MockCell, 5> container;
        std::vector<int> ids;

        container.insert(2, MockCell(2, &ids));
        container.insert(1, MockCell(1, &ids));
        container.insert(6, MockCell(6, &ids));
        TS_ASSERT_EQUALS(container.size(), std::size_t(3));

        container.clear();
        TS_ASSERT_EQUALS(container.size(), std::size_t(0));
    }

    void testUpdate()
    {
        std::vector<int> ids;
        DisplacedGrid<ContainerCell<MockCell, 9> > grid(
            CoordBox<2>(Coord<2>(-1, -1), Coord<2>(3, 3)));

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                for (int i = 0; i < ((x + 1) * (y + 1)); ++i) {
                    int id = 9000 + x * 100 + y * 10 + i;
                    ids << id;
                    grid[Coord<2>(x - 1, y - 1)].insert(id, MockCell(id, &ids));
                }
            }
        }

        DisplacedGrid<ContainerCell<MockCell, 9> > gridOld = grid;
        grid[Coord<2>(0, 0)].update(gridOld, 0);

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                for (int i = 0; i < ((x + 1) * (y + 1)); ++i) {
                    int id = 9000 + x * 100 + y * 10 + i;
                    Coord<2> c(x - 1, y - 1);
                    int expectedID = id;
                    if (c == Coord<2>(0, 0))
                        expectedID -= 4000;
                    TS_ASSERT_EQUALS(expectedID, grid[c][id]->id);
                }
            }
        }
    }
};

}
