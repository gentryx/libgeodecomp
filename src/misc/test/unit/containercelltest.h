#include <cxxtest/TestSuite.h>
#include "../../displacedgrid.h"
#include "../../meshlessadapter.h"
#include "../../containercell.h"
#include "../../testhelper.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class MockCell
{
public:
    typedef Topologies::Cube<2>::Topology Topology;

    MockCell(int _id=0, SuperVector<int> *_ids=0) :
        id(_id),
        ids(_ids)
    {}

    template<class NEIGHBORHOOD>
    void update(NEIGHBORHOOD neighbors, int nanoStep)
    {
        for (SuperVector<int>::iterator i = ids->begin();
             i != ids->end(); ++i) {
            TS_ASSERT_EQUALS(*i, neighbors[*i].id);
        }

        TS_ASSERT_THROWS(neighbors[4711].id, std::logic_error);
        id -= 4000;
    }

    int id;
    SuperVector<int> *ids;
};

class ContainerCellTest : public CxxTest::TestSuite 
{
public:

    void testInsertAndSearch()
    {
        ContainerCell<MockCell, 5> container;
        SuperVector<int> ids;
        ids << 1 << 2 << 4 << 5 << 6;
        
        container.insert(2, MockCell(2, &ids));
        container.insert(1, MockCell(1, &ids));
        container.insert(6, MockCell(6, &ids));
        container.insert(5, MockCell(5, &ids));
        container.insert(4, MockCell(4, &ids));
        container.insert(4, MockCell(4, &ids));

        TS_ASSERT_EQUALS(5, container.size);
        TS_ASSERT_THROWS(container.insert(47, MockCell(47, 0)), std::logic_error);
        TS_ASSERT_THROWS(container.insert(3,  MockCell(3 , 0)), std::logic_error);


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
    
    void testRemove()
    {
        ContainerCell<MockCell, 5> container;
        SuperVector<int> ids;
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

    void testUpdate()
    {
        SuperVector<int> ids;
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
                
        grid[Coord<2>(0, 0)].update(grid, 0);

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
