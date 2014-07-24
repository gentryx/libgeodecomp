#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/boxcell.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

/**
 * A simple test class which counts the surrounding particles within a
 * certain distance.
 */
template<int DIM>
class SimpleParticle
{
public:
    SimpleParticle(
        const FloatCoord<DIM>& pos = FloatCoord<DIM>(),
        const double positionFactor = 1.0,
        const double maxDistance = 0) :
        pos(pos),
        positionFactor(positionFactor),
        maxDistance2(maxDistance * maxDistance)
    {}

    template<class ITERATOR>
    inline void update(const ITERATOR& begin, const ITERATOR& end, const int nanoStep)
    {
        neighbors = 0;

        for (ITERATOR i = begin; i != end; ++i) {
            FloatCoord<DIM> delta = i->pos - pos;
            double distance2 = delta * delta;
            if (distance2 < maxDistance2) {
                ++neighbors;
            }
        }

        pos *= positionFactor;
    }

    int getNeighbors() const
    {
        return neighbors;
    }

    inline const FloatCoord<DIM>& getPos() const
    {
        return pos;
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "SimpleParticle(pos: " << pos << ", maxDistance2: " << maxDistance2 << ", neighbors: " << neighbors << ")\n";
        return buf.str();
    }

private:
    FloatCoord<DIM> pos;
    double positionFactor;
    double maxDistance2;
    int neighbors;
};

class BoxCellTest : public CxxTest::TestSuite
{
public:
    typedef BoxCell<FixedArray<SimpleParticle<2>, 30> > CellType;

    void setUp()
    {
        gridDim = Coord<2> (10, 5);
        cellDim = FloatCoord<2>(2.0, 3.0);
        box = CoordBox<2>(Coord<2>(0, 0), gridDim);
        region.clear();
        region << box;

        grid1 = Grid<CellType>(gridDim);
        grid2 = Grid<CellType>(gridDim);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            FloatCoord<2> origin00 = cellDim.scale(*i);
            FloatCoord<2> origin01 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.0, 0.5));
            FloatCoord<2> origin10 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.5, 0.0));
            FloatCoord<2> origin11 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.5, 0.5));

            double posFactor = 0.95;
            double maxDistance = 2.9;

            grid1[*i] = CellType(origin00, cellDim);
            grid1[*i].insert(SimpleParticle<2>(origin00, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<2>(origin01, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<2>(origin10, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<2>(origin11, posFactor, maxDistance));
        }

    }

    // fixme: add performance tests (both, regular and cuda)
    // fixme: add 3d test

    void testBasic2D()
    {
        UpdateFunctor<CellType>()(
            region,
            Coord<2>(),
            Coord<2>(),
            grid1,
            &grid2,
            0);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT_EQUALS(grid2[*i].size(), std::size_t(4));
        }

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            for (int j = 0; j < 4; ++j) {
                // a particle should be able to see a field of 5x3
                // particles (including itself), unless it's situated at
                // the border of the simulation space:
                int fieldDimX = 5;
                int fieldDimY = 3;

                if (i->x() == 0) {
                    if ((j == 0) || (j == 1)) {
                        fieldDimX = 3;
                    } else {
                        fieldDimX = 4;
                    }
                }

                if (i->x() == (box.dimensions.x() - 1)) {
                    if ((j == 0) || (j == 1)) {
                        fieldDimX = 4;
                    } else {
                        fieldDimX = 3;
                    }
                }

                if (i->y() == 0) {
                    if ((j == 0) || (j == 2)) {
                        fieldDimY= 2;
                    }
                }

                if (i->y() == (box.dimensions.y() - 1)) {
                    if ((j == 1) || (j == 3)) {
                        fieldDimY= 2;
                    }
                }

                int expected = fieldDimX * fieldDimY;
                TS_ASSERT_EQUALS(grid2[*i].particles[j].getNeighbors(), expected);
            }
        }
    }

    void test2DCellTransport()
    {
        Coord<2> dim = box.dimensions;

        UpdateFunctor<CellType>()(
            region,
            Coord<2>(),
            Coord<2>(),
            grid1,
            &grid2,
            0);

        // we assume that after the first update step all four
        // particles still reside in their original cell:
        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                int expected = 4;

                TS_ASSERT_EQUALS(grid2[Coord<2>(x, y)].size(), expected);

            }
        }

        UpdateFunctor<CellType>()(
            region,
            Coord<2>(),
            Coord<2>(),
            grid2,
            &grid1,
            0);

        // after the second iteration we assume that those particles
        // on the left and upper cell boundaries have transitioned to
        // the corresponding neighbor cells:
        for (int y = 0; y < dim.y(); ++y) {
            int fieldDimY = 2;
            if (y == 0) {
                fieldDimY = 3;
            }
            if (y == (dim.y() - 1)) {
                fieldDimY = 1;
            }

            for (int x = 0; x < dim.x(); ++x) {
                int fieldDimX = 2;
                if (x == 0) {
                    fieldDimX = 3;
                }
                if (x == (dim.x() - 1)) {
                    fieldDimX = 1;
                }

                int expected = fieldDimX * fieldDimY;

                TS_ASSERT_EQUALS(grid1[Coord<2>(x, y)].size(), expected);
            }
        }
    }

private:
    Coord<2> gridDim;
    FloatCoord<2> cellDim;
    CoordBox<2> box;
    Region<2> region;

    Grid<CellType> grid1;
    Grid<CellType> grid2;
};

}
