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
        const FloatCoord<DIM>& pos = FloatCoord<DIM>(), double maxDistance = 0) :
        pos(pos),
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

    }

    int getNeighbors() const
    {
        return neighbors;
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "SimpleParticle(pos: " << pos << ", maxDistance2: " << maxDistance2 << ", neighbors: " << neighbors << ")\n";
        return buf.str();
    }

private:
    FloatCoord<DIM> pos;
    double maxDistance2;
    int neighbors;
};

class BoxCellTest : public CxxTest::TestSuite
{
public:

    // fixme: add performance tests (both, regular and cuda)

    void test2D()
    {
        Coord<2> gridDim(10, 5);
        FloatCoord<2> cellDim(2.0, 3.0);
        CoordBox<2> box(Coord<2>(0, 0), gridDim);
        Region<2> region;
        region << box;

        typedef BoxCell<FixedArray<SimpleParticle<2>, 30> > CellType;
        Grid<CellType> grid1(gridDim);
        Grid<CellType> grid2(gridDim);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            FloatCoord<2> origin00 = cellDim.scale(*i);
            FloatCoord<2> origin01 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.0, 0.5));
            FloatCoord<2> origin10 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.5, 0.0));
            FloatCoord<2> origin11 = cellDim.scale(*i) + cellDim.scale(FloatCoord<2>(0.5, 0.5));

            grid1[*i] = CellType(origin00, cellDim);
            grid1[*i].insert(SimpleParticle<2>(origin00, 2.9));
            grid1[*i].insert(SimpleParticle<2>(origin01, 2.9));
            grid1[*i].insert(SimpleParticle<2>(origin10, 2.9));
            grid1[*i].insert(SimpleParticle<2>(origin11, 2.9));
        }

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

};

}
