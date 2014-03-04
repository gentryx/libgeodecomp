#include <libgeodecomp/geometry/voronoimesher.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/containercell.h>
#include <libgeodecomp/storage/grid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DummyCell
{
public:
    DummyCell(const FloatCoord<2>& center = FloatCoord<2>(0, 0)) :
        center(center)
    {}

    FloatCoord<2> center;
};

typedef ContainerCell<DummyCell, 100> ContainerCellType;

class MockMesher : public VoronoiMesher<ContainerCellType>
{
public:
    MockMesher(const Coord<2>& gridDim, const FloatCoord<2>& quadrantSize, double minCellDistance) :
        VoronoiMesher(gridDim, quadrantSize, minCellDistance),
        cellCounter(0)
    {}

    virtual void addCell(ContainerCellType *container, const FloatCoord<DIM>& center)
    {
        container->insert(cellCounter++, DummyCell(center));
    }

    int cellCounter;
};

class VoronoiMesherTest : public CxxTest::TestSuite
{
public:

    void testFillGeometryData()
    {
        Coord<2> dim(5, 10);
        FloatCoord<2> quadrantSize(100, 100);
        double minCellDistance = 20;
        Grid<ContainerCellType> grid(dim);
        MockMesher mesher(dim, quadrantSize, minCellDistance);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);

                for (int subY = 0; subY < 10; ++subY) {
                    for (int subX = 0; subX < 10; ++subX) {
                        FloatCoord<2> realPos(
                            x * quadrantSize[0] + subX,
                            y * quadrantSize[1] + subY);
                        mesher.addCell(&grid[c], realPos);
                    }
                }
            }
        }
    }

    void testAddRandomCells()
    {
        Coord<2> dim(7, 3);
        Coord<2> loc(5, 2);
        FloatCoord<2> quadrantSize(100, 200);
        double minCellDistance = 1.0;
        Grid<ContainerCellType> grid(dim);
        MockMesher mesher(dim, quadrantSize, minCellDistance);

        TS_ASSERT_EQUALS(grid[loc].size(), std::size_t(0));
        mesher.addRandomCells(&grid, loc, 10);
        TS_ASSERT_EQUALS(grid[loc].size(), std::size_t(10));
        mesher.addRandomCells(&grid, loc, 100);
        // ensure that 10 cells were rejected b/c they were placed on
        // the same positions as the first 10 cells:
        TS_ASSERT_EQUALS(grid[loc].size(), std::size_t(100));

        std::vector<int> xCoords;
        std::vector<int> yCoords;

        const ContainerCellType& cell = grid[loc];
        for (ContainerCellType::const_iterator i = cell.begin(); i != cell.end(); ++i) {
            xCoords << i->center[0];
            yCoords << i->center[1];
        }

        double minX = loc[0] * quadrantSize[0];
        double minY = loc[1] * quadrantSize[1];
        double maxX = minX + quadrantSize[0];
        double maxY = minY + quadrantSize[1];

        TS_ASSERT((min)(xCoords) >= minX + 0);
        TS_ASSERT((min)(xCoords) <  minX + 10);
        TS_ASSERT((max)(xCoords) >= maxX - 10);
        TS_ASSERT((max)(xCoords) <  maxX + 0);

        TS_ASSERT((min)(yCoords) >= minY + 0);
        TS_ASSERT((min)(yCoords) <  minY + 10);
        TS_ASSERT((max)(yCoords) >= maxY - 10);
        TS_ASSERT((max)(yCoords) <  maxY + 0);
    }

};

}
