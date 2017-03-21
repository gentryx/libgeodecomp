#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/boxcell.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/misc/apitraits.h>
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
    class API : public APITraits::HasCubeTopology<DIM>
    {};

    explicit SimpleParticle(
        const FloatCoord<DIM>& pos = FloatCoord<DIM>(),
        const double positionFactor = 1.0,
        const double maxDistance = 0) :
        pos(pos),
        positionFactor(positionFactor),
        maxDistance2(maxDistance * maxDistance)
    {}

    template<typename HOOD>
    inline void update(const HOOD& hood, const unsigned nanoStep)
    {
        neighbors = 0;

        for (typename HOOD::Iterator i = hood.begin(); i != hood.end(); ++i) {
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

/**
 * Another simple test particle which simply spawns new particles
 */
class SpawningParticle
{
public:
    class API : public APITraits::HasCubeTopology<3>
    {};

    explicit SpawningParticle(
        const FloatCoord<3>& pos = FloatCoord<3>(),
        const int numParticlesToBeSpawned = 0) :
        pos(pos),
        numParticlesToBeSpawned(numParticlesToBeSpawned)
    {}

    inline const FloatCoord<3>& getPos() const
    {
        return pos;
    }

    template<typename HOOD>
    inline void update(HOOD& hood, const unsigned nanoStep)
    {
        for (int i = 0; i < numParticlesToBeSpawned; ++i) {
            hood << SpawningParticle(pos, 0);
        }
    }

private:
    FloatCoord<3> pos;
    int numParticlesToBeSpawned;
};

class BoxCellTest : public CxxTest::TestSuite
{
public:
    typedef BoxCell<FixedArray<SimpleParticle<2>, 30> > CellType;

    void setUp()
    {
        gridDim = Coord<2>(10, 5);
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
                std::size_t expected = 4;

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

                std::size_t expected = fieldDimX * fieldDimY;

                TS_ASSERT_EQUALS(grid1[Coord<2>(x, y)].size(), expected);
            }
        }
    }

    void test3D()
    {
        typedef BoxCell<FixedArray<SimpleParticle<3>, 30> > CellType;
        typedef APITraits::SelectTopology<CellType>::Value Topology;

        Coord<3> gridDim(10, 5, 4);
        FloatCoord<3> cellDim(2.0, 3.0, 5.0);
        CoordBox<3> box(Coord<3>(0, 0), gridDim);
        Region<3> region;
        region << box;

        Grid<CellType, Topology> grid1(gridDim);
        Grid<CellType, Topology> grid2(gridDim);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            FloatCoord<3> origin000 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.0, 0.0, 0.0));
            FloatCoord<3> origin001 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.0, 0.0, 0.5));
            FloatCoord<3> origin010 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.0, 0.5, 0.0));
            FloatCoord<3> origin011 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.0, 0.5, 0.5));
            FloatCoord<3> origin100 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.5, 0.0, 0.0));
            FloatCoord<3> origin101 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.5, 0.0, 0.5));
            FloatCoord<3> origin110 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.5, 0.5, 0.0));
            FloatCoord<3> origin111 = cellDim.scale(*i) + cellDim.scale(FloatCoord<3>(0.5, 0.5, 0.5));

            double posFactor = 0.95;
            double maxDistance = 2.9;

            grid1[*i] = CellType(origin000, cellDim);
            grid1[*i].insert(SimpleParticle<3>(origin000, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin001, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin010, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin011, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin100, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin101, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin110, posFactor, maxDistance));
            grid1[*i].insert(SimpleParticle<3>(origin111, posFactor, maxDistance));
        }

        UpdateFunctor<CellType>()(
            region,
            Coord<3>(),
            Coord<3>(),
            grid1,
            &grid2,
            0);

        // we assume that after the first update step all eight
        // particles still reside in their original cell:
        for (int z = 0; z < gridDim.z(); ++z) {
            for (int y = 0; y < gridDim.y(); ++y) {
                for (int x = 0; x < gridDim.x(); ++x) {
                    std::size_t expected = 8;

                    TS_ASSERT_EQUALS(grid2[Coord<3>(x, y, z)].size(), expected);
                }
            }
        }

        UpdateFunctor<CellType>()(
            region,
            Coord<3>(),
            Coord<3>(),
            grid2,
            &grid1,
            0);

        // after the second step we assume that particles on the
        // boundaries have transitioned to neighboring cells:
        for (int z = 0; z < gridDim.z(); ++z) {
            for (int y = 0; y < gridDim.y(); ++y) {
                for (int x = 0; x < gridDim.x(); ++x) {

                    Coord<3> expectedCubeDim(2, 2, 2);
                    if (x == 0) {
                        expectedCubeDim.x() += 1;
                    }
                    if (y == 0) {
                        expectedCubeDim.y() += 1;
                    }
                    if (z == 0) {
                        expectedCubeDim.z() += 1;
                    }

                    if (x == (gridDim.x() - 1)) {
                        expectedCubeDim.x() -= 1;
                    }
                    if (y == (gridDim.y() - 1)) {
                        expectedCubeDim.y() -= 1;
                    }
                    if (z == (gridDim.z() - 1)) {
                        expectedCubeDim.z() -= 1;
                    }

                    std::size_t expected = expectedCubeDim.prod();

                    TS_ASSERT_EQUALS(grid1[Coord<3>(x, y, z)].size(), expected);
                }
            }
        }
    }

    void testParticleSpawn()
    {
        // we need space for (6 * 5 * 4 + 1) particles
        typedef BoxCell<FixedArray<SpawningParticle, 121> > CellType;
        typedef APITraits::SelectTopology<CellType>::Value Topology;

        Coord<3> gridDim(7, 6, 5);
        FloatCoord<3> cellDim(2.0, 3.0, 5.0);
        CoordBox<3> box(Coord<3>(0, 0), gridDim);
        Region<3> region;
        region << box;

        Grid<CellType, Topology> grid1(gridDim);
        Grid<CellType, Topology> grid2(gridDim);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            FloatCoord<3> origin = cellDim.scale(*i);
            FloatCoord<3> particlePos = origin + cellDim.scale(FloatCoord<3>(0.5, 0.5, 0.5));
            int numParticlesToBeSpawned = i->prod();

            CellType cell(origin, cellDim);
            cell.insert(SpawningParticle(particlePos, numParticlesToBeSpawned));

            grid1[*i] = cell;
        }

        UpdateFunctor<CellType>()(
            region,
            Coord<3>(),
            Coord<3>(),
            grid1,
            &grid2,
            0);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            int expectedParticles = 1 + i->prod();

            TS_ASSERT_EQUALS(grid2[*i].size(), expectedParticles);
        }
    }

    void testRemove()
    {
        typedef BoxCell<FixedArray<SpawningParticle, 222> > CellType;

        FloatCoord<3> origin(3.0, 3.0, 3.0);
        FloatCoord<3> cellDim(2.0, 2.0, 2.0);

        CellType cell(origin, cellDim);
        cell.insert(SpawningParticle(FloatCoord<3>(3.5, 3.5, 3.5), 100));

        TS_ASSERT_EQUALS(cell.size(), 1);
        cell.remove(0);
        TS_ASSERT_EQUALS(cell.size(), 0);
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
