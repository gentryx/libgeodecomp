#include <libgeodecomp/geometry/voronoimesher.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/containercell.h>
#include <libgeodecomp/storage/grid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DummyCell
{
public:
    explicit DummyCell(const FloatCoord<2>& center = FloatCoord<2>(0, 0), int id = -1) :
        center(center),
        id(id),
        area(0)
    {}

    void setArea(const double newArea)
    {
        area = newArea;
    }

    void setShape(const std::vector<FloatCoord<2> > newShape)
    {
        shape = newShape;
    }

    void pushNeighbor(const int id, const double boundaryLength, const FloatCoord<2> dir)
    {
        neighborIDs << id;
        neighborBoundaryLengths << boundaryLength;
        neighborDirections << dir;
    }

    std::size_t numberOfNeighbors() const
    {
        return neighborIDs.size();
    }

    FloatCoord<2> center;
    int id;
    double area;
    std::vector<FloatCoord<2> > shape;
    std::vector<int> neighborIDs;
    std::vector<double> neighborBoundaryLengths;
    std::vector<FloatCoord<2> > neighborDirections;
};

typedef ContainerCell<DummyCell, 100> ContainerCellType;

class MockMesher : public VoronoiMesher<ContainerCellType>
{
public:
    MockMesher(const Coord<2>& gridDim, const FloatCoord<2>& quadrantSize, double minCellDistance) :
        VoronoiMesher<ContainerCellType>(gridDim, quadrantSize, minCellDistance),
        cellCounter(0)
    {}

    virtual ~MockMesher()
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
    void testElementDoesntRemoveLimitsAfterDuplicateInsertion()
    {
        FloatCoord<2> center(1.87819, 58.5767);
        ConvexPolytope<FloatCoord<2> > element(
            center,
            FloatCoord<2>(400, 400),
            FloatCoord<2>(4000, 1600),
            10,
            0);

        FloatCoord<2> origin = FloatCoord<2>(0.939096, 29.2884);
        Plane<FloatCoord<2> > equation(origin, (center - origin) * 2, 1);
        element << equation;
        TS_ASSERT_EQUALS(element.getShape().size(), std::size_t(5));

        element << equation;
        TS_ASSERT_EQUALS(element.getShape().size(), std::size_t(5));
    }

    void testFillGeometryData()
    {
        Coord<2> dim(5, 3);
        CoordBox<2> box(Coord<2>(), dim);
        FloatCoord<2> quadrantSize(100, 100);
        double minCellDistance = 20;
        Grid<ContainerCellType> grid(dim);
        MockMesher mesher(dim, quadrantSize, minCellDistance);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);

                for (int subY = 0; subY < 4; ++subY) {
                    for (int subX = 0; subX < 4; ++subX) {
                        FloatCoord<2> realPos(
                            x * quadrantSize[0] + subX * 15.5,
                            y * quadrantSize[1] + subY * 16.5);
                        mesher.addCell(&grid[c], realPos);
                    }
                }
            }
        }

        mesher.fillGeometryData(&grid);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            ContainerCellType cell = grid[*i];
            TS_ASSERT_EQUALS(cell.size(), std::size_t(16));

            for (ContainerCellType::Iterator j = cell.begin(); j != cell.end(); ++j) {
                if (j->shape.size() < 4) {
                    std::cout << "center: " << j->center << "\n"
                              << "  shape: " << j->shape << "\n";
                }
                TS_ASSERT_EQUALS(j->shape.size(), std::size_t(4));
                TS_ASSERT(j->area > 0);
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

    void testCollision()
    {
        Coord<2> dim(1, 1);
        std::size_t numCells = 100;
        double quadrantSize = 400;
        double minDistance = 40;

        MockMesher mesher(dim, FloatCoord<2>(quadrantSize, quadrantSize), minDistance);
        Grid<ContainerCellType> grid(dim);
        mesher.addCell(&grid[Coord<2>(0, 0)], FloatCoord<2>(0, 0));
        mesher.addRandomCells(&grid, Coord<2>(0, 0), numCells);
    }

};

}
