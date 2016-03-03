#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/geometry/convexpolytope.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/partitions/hindexingpartition.h>
#include <libgeodecomp/geometry/partitions/hilbertpartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/linepointerassembly.h>
#include <libgeodecomp/storage/linepointerupdatefunctor.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/parallelization/openmpsimulator.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>
#include <libgeodecomp/storage/unstructuredupdatefunctor.h>

#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>
#include <libflatarray/api_traits.hpp>
#include <libflatarray/macros.hpp>

#include <emmintrin.h>
#ifdef __AVX__
#include <immintrin.h>
#endif

#include <iomanip>
#include <iostream>
#include <stdio.h>

using namespace LibGeoDecomp;
using namespace LibFlatArray;

class RegionCount : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionCount";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int sum = 0;
        Region<3> r;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int z = 0; z < dim.z(); z += 4) {
                for (int y = 0; y < dim.y(); y += 4) {
                    for (int x = 0; x < dim.x(); x += 4) {
                        sum += r.count(Coord<3>(x, y, z));
                    }
                }
            }
        }

        if (sum == 31) {
            std::cout << "pure debug statement to prevent the compiler from optimizing away the previous loop";
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionInsert : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionInsert";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r;
            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionIntersect : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionIntersect";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r1;
            Region<3> r2;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            for (int z = 1; z < (dim.z() - 1); ++z) {
                for (int y = 1; y < (dim.y() - 1); ++y) {
                    r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
                }
            }

            Region<3> r3 = r1 & r2;
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionSubtract : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionSubtract";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r1;
            Region<3> r2;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            for (int z = 1; z < (dim.z() - 1); ++z) {
                for (int y = 1; y < (dim.y() - 1); ++y) {
                    r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
                }
            }

            Region<3> r3 = r1 - r2;
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionUnion : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionUnion";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r1;
            Region<3> r2;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            for (int z = 1; z < (dim.z() - 1); ++z) {
                for (int y = 1; y < (dim.y() - 1); ++y) {
                    r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
                }
            }

            Region<3> r3 = r1 + r2;
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionAppend : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionAppend";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            Region<3> r1;
            Region<3> r2;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            for (int z = 1; z < (dim.z() - 1); ++z) {
                for (int y = 1; y < (dim.y() - 1); ++y) {
                    r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
                }
            }

            Region<3> akku1 = r1;
            Region<3> akku2 = r2;
            int sum = 0;

            {
                ScopedTimer t(&seconds);
                akku1 += r2;
                akku2 += r1;

                sum += akku1.size();
                sum += akku2.size();
                sum += (r1 + r2).size();
                sum += (r2 + r1).size();
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionExpand : public CPUBenchmark
{
public:
    explicit RegionExpand(int expansionWidth) :
        expansionWidth(expansionWidth)
    {}

    std::string family()
    {
        std::stringstream buf;
        buf << "RegionExpand" << expansionWidth;
        return buf.str();
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r1;
            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            Region<3> r2 = r1.expand(expansionWidth);
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }

private:
    int expansionWidth;
};

class RegionExpandWithAdjacency : public CPUBenchmark
{
public:
    explicit RegionExpandWithAdjacency(
        std::map<int, ConvexPolytope<FloatCoord<2> > > cells) :
        rawCells(cells)
    {}

    std::string family()
    {
        std::stringstream buf;
        // we don't name this RegionExpandWithAdjacency so users can
        // still selectively run RegionExpand sans this test.
        buf << "RegionExpWithAdjacency";
        return buf.str();
    }

    std::string species()
    {
        return "gold";
    }

    static std::map<int, ConvexPolytope<FloatCoord<2> > > genGrid(int numCells)
    {
        int elementsPerChunk = 5;
        int numChunks = numCells / elementsPerChunk;
        Coord<2> gridSize = Coord<2>::diagonal(sqrt(numChunks));
        Coord<2> chunkDim(100, 100);
        Coord<2> globalDim = chunkDim.scale(gridSize);
        double minDistance = 10;
        int counter = 0;

        Grid<std::map<int, Coord<2> >, Topologies::Torus<2>::Topology> grid(gridSize);
        for (int y = 0; y < gridSize.y(); ++y) {
            for (int x = 0; x < gridSize.x(); ++x) {
                Coord<2> gridIndex(x, y);
                fillChunk(&grid, gridIndex, &counter, elementsPerChunk, chunkDim, minDistance);
            }
        }

        std::map<int, ConvexPolytope<FloatCoord<2> > > cells;
        Coord<2> relativeCoords[] = {
            Coord<2>( 0,  0),
            Coord<2>(-1,  0),
            Coord<2>( 1,  0),
            Coord<2>( 0, -1),
            Coord<2>( 0,  1),
            Coord<2>(-1, -1),
            Coord<2>( 1, -1),
            Coord<2>(-1,  1),
            Coord<2>( 1,  1)
        };

        for (int y = 0; y < gridSize.y(); ++y) {
            for (int x = 0; x < gridSize.x(); ++x) {
                Coord<2> gridIndex(x, y);
                const std::map<int, Coord<2> >& chunk = grid[gridIndex];

                for (std::map<int, Coord<2> >::const_iterator i = chunk.begin(); i != chunk.end(); ++i) {
                    ConvexPolytope<FloatCoord<2> > element(i->second, globalDim);

                    for (int c = 0; c < 9; ++c) {
                        Coord<2> currentIndex = gridIndex + relativeCoords[c];
                        const std::map<int, Coord<2> >& neighbors = grid[currentIndex];
                        for (std::map<int, Coord<2> >::const_iterator j = neighbors.begin(); j != neighbors.end(); ++j) {
                            if (j->second == i->second) {
                                continue;
                            }

                            element << std::make_pair(j->second, j->first);
                        }

                        if (c == 0) {
                            element.updateGeometryData(true);
                        }
                    }

                    cells[i->first] = element;
                }
            }
        }

        return cells;
    }

    double performance(std::vector<int> dim)
    {
        double seconds = 0;

        // I. Adapt Voronio Mesh (i.e. Set of Cells)
        int skipCells = dim[1];
        int expansionWidth = dim[2];
        int idStreakLength = dim[3];

        std::map<int, ConvexPolytope<FloatCoord<2> > > cells = mapIDs(rawCells, idStreakLength);

        // II. Extract Adjacency List from Cells
        RegionBasedAdjacency adjacency;

        for (std::map<int, ConvexPolytope<FloatCoord<2> > >::iterator  i = cells.begin(); i != cells.end(); ++i) {
            int id = i->first;
            const ConvexPolytope<FloatCoord<2> > element = i->second;

            addNeighbors(adjacency, id, element.getLimits());
        }

        // III. Fill Region
        typedef std::map<int, std::vector<int> > MapAdjacency;

        MapAdjacency mapAdjacency;
        for (Region<2>::Iterator it = adjacency.getRegion().begin(); it != adjacency.getRegion().end(); ++it) {
            mapAdjacency[it->x()].push_back(it->y());
        }

        Region<1> r;
        int counter = 0;
        bool select = true;
        for (MapAdjacency::iterator i = mapAdjacency.begin(); i != mapAdjacency.end(); ++i) {
            ++counter;
            if (counter >= skipCells) {
                counter = 0;
                select = !select;
            }

            if (select) {
                r << Coord<1>(i->first);
            }
        }

        // IV. Performance Measurement
        {
            ScopedTimer t(&seconds);

            Region<1> q = r.expandWithTopology(expansionWidth, Coord<1>(), Topologies::Unstructured::Topology(), adjacency);

            if (q.size() == 4711) {
                std::cout << "pure debug statement to prevent the compiler from optimizing away the previous function";
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }

private:
    std::map<int, ConvexPolytope<FloatCoord<2> > > rawCells;

    static std::map<int, ConvexPolytope<FloatCoord<2> > > mapIDs(
        const std::map<int, ConvexPolytope<FloatCoord<2> > >& rawCells, int idStreakLength)
    {
        std::map<int, ConvexPolytope<FloatCoord<2> > > ret;
        for (std::map<int, ConvexPolytope<FloatCoord<2> > >::const_iterator i = rawCells.begin(); i != rawCells.end(); ++i) {
            ConvexPolytope<FloatCoord<2> > element = i->second;
            mapLimitIDs(&element.getLimits(), idStreakLength);
            ret[mapID(i->first, idStreakLength)] = element;
        }

        return ret;
    }

    template<typename LIMITS_CONTAINER>
    static void mapLimitIDs(LIMITS_CONTAINER *limits, int idStreakLength)
    {
        for (typename LIMITS_CONTAINER::iterator i = limits->begin(); i != limits->end(); ++i) {
            i->neighborID = mapID(i->neighborID, idStreakLength);
        }
    }

    static int mapID(int id, int idStreakLength)
    {
        if (idStreakLength == -1) {
            return id;
        }

        return id / idStreakLength * 2 * idStreakLength + id % idStreakLength;
    }

    template<typename GRID>
    static void fillChunk(GRID *grid, const Coord<2>& gridIndex, int *counter, int elementsPerChunk, const Coord<2>& chunkDim, double minDistance)
    {
        Coord<2> chunkOffset = gridIndex.scale(chunkDim);

        for (int i = 0; i < elementsPerChunk; ++i) {
            Coord<2> randomCoord = Coord<2>(Random::gen_u(chunkDim.x()), Random::gen_u(chunkDim.y()));
            randomCoord += chunkOffset;

            if (doesNotCollide(randomCoord, *grid, gridIndex, minDistance)) {
                int id = (*counter)++;
                (*grid)[gridIndex][id] = randomCoord;
            }
        }
    }

    template<typename COORD, typename GRID>
    static bool doesNotCollide(COORD position, const GRID& grid, Coord<2> gridIndex, double minDistance)
    {
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                Coord<2> currentIndex = gridIndex + Coord<2>(x, y);
                bool valid = positionMaintainsMinDistanceToOthers(
                    position,
                    grid[currentIndex].begin(),
                    grid[currentIndex].end(),
                    minDistance);

                if (!valid) {
                    return false;
                }
            }
        }

        return true;
    }

    template<typename COORD, typename ITERATOR1, typename ITERATOR2>
    static bool positionMaintainsMinDistanceToOthers(
        const COORD& position, const ITERATOR1& begin, const ITERATOR2& end, double minDistance)
    {
        for (ITERATOR1 i = begin; i != end; ++i) {
            COORD delta = i->second - position;
            if (delta.abs().maxElement() < minDistance) {
                return false;
            }
        }

        return true;
    }

    template<typename LIMITS>
    void addNeighbors(Adjacency& adjacency, int from, const LIMITS& limits)
    {
        for (typename LIMITS::const_iterator i = limits.begin(); i != limits.end(); ++i) {
            adjacency.insert(from, i->neighborID);
        }
    }
};


class CoordEnumerationVanilla : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    for (int x = 0; x < dim.x(); ++x) {
                        sum += Coord<3>(x, y, z);
                    }
                }
            }
            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }

        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationBronze : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;
            for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
                sum += *i;
            }

            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationGold : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;
            for (Region<3>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                Coord<3> c = i->origin;
                for (; c.x() < i->endX; c.x() += 1) {
                    sum += c;
                }
            }
            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class FloatCoordAccumulationGold : public CPUBenchmark
{
public:
    std::string family()
    {
        return "FloatCoordAccumulation";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            FloatCoord<3> sum;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    for (int x = 0; x < dim.x(); ++x) {
                        FloatCoord<3> addent(x, y, z);
                        sum += addent;
                    }
                }
            }

            // trick the compiler to not optimize away the loop above
            if (sum == FloatCoord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }

        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class Jacobi3DVanilla : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int dimX = dim.x();
        int dimY = dim.y();
        int dimZ = dim.z();
        int offsetZ = dimX * dimY;
        int maxT = 20;

        double *gridOld = new double[dimX * dimY * dimZ];
        double *gridNew = new double[dimX * dimY * dimZ];

        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimY; ++x) {
                    gridOld[z * offsetZ + y * dimY + x] = x + y + z;
                    gridNew[z * offsetZ + y * dimY + x] = x + y + z;
                }
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                for (int z = 1; z < (dimZ - 1); ++z) {
                    for (int y = 1; y < (dimY - 1); ++y) {
                        for (int x = 1; x < (dimX - 1); ++x) {
                            gridNew[z * offsetZ + y * dimY + x] =
                                (gridOld[z * offsetZ + y * dimX + x - offsetZ] +
                                 gridOld[z * offsetZ + y * dimX + x - dimX] +
                                 gridOld[z * offsetZ + y * dimX + x - 1] +
                                 gridOld[z * offsetZ + y * dimX + x + 0] +
                                 gridOld[z * offsetZ + y * dimX + x + 1] +
                                 gridOld[z * offsetZ + y * dimX + x + dimX] +
                                 gridOld[z * offsetZ + y * dimX + x + offsetZ]) * (1.0 / 7.0);
                        }
                    }
                }
            }
        }

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        delete[] gridOld;
        delete[] gridNew;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DSSE : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "pepper";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int dimX = dim.x();
        int dimY = dim.y();
        int dimZ = dim.z();
        int offsetZ = dimX * dimY;
        int maxT = 20;

        double *gridOld = new double[dimX * dimY * dimZ];
        double *gridNew = new double[dimX * dimY * dimZ];

        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimY; ++x) {
                    gridOld[z * offsetZ + y * dimY + x] = x + y + z;
                    gridNew[z * offsetZ + y * dimY + x] = x + y + z;
                }
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                for (int z = 1; z < (dimZ - 1); ++z) {
                    for (int y = 1; y < (dimY - 1); ++y) {
                        updateLine(&gridNew[z * offsetZ + y * dimY + 0],
                                   &gridOld[z * offsetZ + y * dimX - offsetZ],
                                   &gridOld[z * offsetZ + y * dimX - dimX],
                                   &gridOld[z * offsetZ + y * dimX + 0],
                                   &gridOld[z * offsetZ + y * dimX + dimX],
                                   &gridOld[z * offsetZ + y * dimX + offsetZ],
                                   1,
                                   dimX - 1);

                    }
                }
            }
        }

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        delete[] gridOld;
        delete[] gridNew;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }

private:
    void updateLine(
        double *target,
        double *south,
        double *top,
        double *same,
        double *bottom,
        double *north,
        int startX,
        int endX)
    {
        int x = startX;

        if ((x % 2) == 1) {
            updatePoint(target, south, top, same, bottom, north, x);
            ++x;
        }

        __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);
        __m128d same1 = _mm_load_pd(same + x + 0);
        __m128d odds0 = _mm_loadu_pd(same + x - 1);

        for (; x < (endX - 7); x += 8) {
            __m128d same2 = _mm_load_pd(same + x + 2);
            __m128d same3 = _mm_load_pd(same + x + 4);
            __m128d same4 = _mm_load_pd(same + x + 6);
            __m128d same5 = _mm_load_pd(same + x + 8);

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same4, same5, (1 << 0) | (0 << 2));

            // load south neighbors
            __m128d buf0 =  _mm_load_pd(south + x + 0);
            __m128d buf1 =  _mm_load_pd(south + x + 2);
            __m128d buf2 =  _mm_load_pd(south + x + 4);
            __m128d buf3 =  _mm_load_pd(south + x + 6);

            // add left neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // add right neighbors
            same1 = _mm_add_pd(same1, odds1);
            same2 = _mm_add_pd(same2, odds2);
            same3 = _mm_add_pd(same3, odds3);
            same4 = _mm_add_pd(same4, odds4);

            // load top neighbors
            odds0 = _mm_load_pd(top + x + 0);
            odds1 = _mm_load_pd(top + x + 2);
            odds2 = _mm_load_pd(top + x + 4);
            odds3 = _mm_load_pd(top + x + 6);

            // add south neighbors
            same1 = _mm_add_pd(same1, buf0);
            same2 = _mm_add_pd(same2, buf1);
            same3 = _mm_add_pd(same3, buf2);
            same4 = _mm_add_pd(same4, buf3);

            // load bottom neighbors
            buf0 =  _mm_load_pd(bottom + x + 0);
            buf1 =  _mm_load_pd(bottom + x + 2);
            buf2 =  _mm_load_pd(bottom + x + 4);
            buf3 =  _mm_load_pd(bottom + x + 6);

            // add top neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // load north neighbors
            odds0 = _mm_load_pd(north + x + 0);
            odds1 = _mm_load_pd(north + x + 2);
            odds2 = _mm_load_pd(north + x + 4);
            odds3 = _mm_load_pd(north + x + 6);

            // add bottom neighbors
            same1 = _mm_add_pd(same1, buf0);
            same2 = _mm_add_pd(same2, buf1);
            same3 = _mm_add_pd(same3, buf2);
            same4 = _mm_add_pd(same4, buf3);

            // add north neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // scale by 1/7
            same1 = _mm_mul_pd(same1, oneSeventh);
            same2 = _mm_mul_pd(same2, oneSeventh);
            same3 = _mm_mul_pd(same3, oneSeventh);
            same4 = _mm_mul_pd(same4, oneSeventh);

            // store results
            _mm_store_pd(target + x + 0, same1);
            _mm_store_pd(target + x + 2, same2);
            _mm_store_pd(target + x + 4, same3);
            _mm_store_pd(target + x + 6, same4);

            odds0 = odds4;
            same1 = same5;
        }

        for (; x < endX; ++x) {
            updatePoint(target, south, top, same, bottom, north, x);
        }
    }

    void updatePoint(
        double *target,
        double *south,
        double *top,
        double *same,
        double *bottom,
        double *north,
        int x)
    {
        target[x] =
            (south[x] +
             top[x] +
             same[x - 1] + same[x + 0] + same[x + 1] +
             bottom[x] +
             north[x]) * (1.0 / 7.0);
    }

};

template<typename CELL>
class NoOpInitializer : public SimpleInitializer<CELL>
{
public:
    typedef typename SimpleInitializer<CELL>::Topology Topology;

    NoOpInitializer(
        const Coord<3>& dimensions,
        unsigned steps) :
        SimpleInitializer<CELL>(dimensions, steps)
    {}

    virtual void grid(GridBase<CELL, Topology::DIM> *target)
    {}
};

class JacobiCellClassic
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    explicit JacobiCellClassic(double t = 0) :
        temp(t)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        temp = (hood[Coord<3>( 0,  0, -1)].temp +
                hood[Coord<3>( 0, -1,  0)].temp +
                hood[Coord<3>(-1,  0,  0)].temp +
                hood[Coord<3>( 0,  0,  0)].temp +
                hood[Coord<3>( 1,  0,  0)].temp +
                hood[Coord<3>( 0,  1,  0)].temp +
                hood[Coord<3>( 0,  0,  1)].temp) * (1.0 / 7.0);
    }

    double temp;
};

class Jacobi3DClassic : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int maxT = 5;
        SerialSimulator<JacobiCellClassic> sim(
            new NoOpInitializer<JacobiCellClassic>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class JacobiCellFixedHood
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    explicit JacobiCellFixedHood(double t = 0) :
        temp(t)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        temp = (hood[FixedCoord< 0,  0, -1>()].temp +
                hood[FixedCoord< 0, -1,  0>()].temp +
                hood[FixedCoord<-1,  0,  0>()].temp +
                hood[FixedCoord< 0,  0,  0>()].temp +
                hood[FixedCoord< 1,  0,  0>()].temp +
                hood[FixedCoord< 0,  1,  0>()].temp +
                hood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 7.0);
    }

    double temp;
};

class Jacobi3DFixedHood : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "silver";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int maxT = 20;

        SerialSimulator<JacobiCellFixedHood> sim(
            new NoOpInitializer<JacobiCellFixedHood>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class QuadM128
{
public:
    __m128d a;
    __m128d b;
    __m128d c;
    __m128d d;
};

class PentaM128
{
public:
    __m128d a;
    __m128d b;
    __m128d c;
    __m128d d;
    __m128d e;
};

class JacobiCellStreakUpdate
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasSoA
    {};

    explicit JacobiCellStreakUpdate(double t = 0) :
        temp(t)
    {}

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateSingle(HOOD_OLD& hoodOld, HOOD_NEW& hoodNew)
    {
        hoodNew.temp() =
            (hoodOld[FixedCoord<0,  0, -1>()].temp() +
             hoodOld[FixedCoord<0, -1,  0>()].temp() +
             hoodOld[FixedCoord<1,  0,  0>()].temp() +
             hoodOld[FixedCoord<0,  0,  0>()].temp() +
             hoodOld[FixedCoord<1,  0,  0>()].temp() +
             hoodOld[FixedCoord<0,  1,  0>()].temp() +
             hoodOld[FixedCoord<0,  0,  1>()].temp()) * (1.0 / 7.0);
    }

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_OLD& hoodOld, int indexEnd,
                            HOOD_NEW& hoodNew, int /* nanoStep */)
    {
        if (hoodOld.index() % 2 == 1) {
            updateSingle(hoodOld, hoodNew);
            ++hoodOld.index();
            ++hoodNew.index;
        }

        __m128d oneSeventh = _mm_set1_pd(1.0 / 7.0);
        PentaM128 same;
        same.a = _mm_load_pd( &hoodOld[FixedCoord< 0, 0, 0>()].temp());
        __m128d odds0 = _mm_loadu_pd(&hoodOld[FixedCoord<-1, 0, 0>()].temp());

        for (; hoodOld.index() < (indexEnd - 8 + 1); hoodOld.index() += 8, hoodNew.index += 8) {

            load(&same, hoodOld, FixedCoord<0, 0, 0>());

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same.a, same.b, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same.b, same.c, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same.c, same.d, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same.d, same.e, (1 << 0) | (0 << 2));

            // load south neighbors
            QuadM128 buf;
            load(&buf, hoodOld, FixedCoord<0, 0, -1>());

            // add left neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // add right neighbors
            same.a = _mm_add_pd(same.a, odds1);
            same.b = _mm_add_pd(same.b, odds2);
            same.c = _mm_add_pd(same.c, odds3);
            same.d = _mm_add_pd(same.d, odds4);

            // load top neighbors
            odds0 = load<0>(hoodOld, FixedCoord< 0, -1, 0>());
            odds1 = load<2>(hoodOld, FixedCoord< 0, -1, 0>());
            odds2 = load<4>(hoodOld, FixedCoord< 0, -1, 0>());
            odds3 = load<6>(hoodOld, FixedCoord< 0, -1, 0>());

            // add south neighbors
            same.a = _mm_add_pd(same.a, buf.a);
            same.b = _mm_add_pd(same.b, buf.b);
            same.c = _mm_add_pd(same.c, buf.c);
            same.d = _mm_add_pd(same.d, buf.d);

            // load bottom neighbors
            load(&buf, hoodOld, FixedCoord<0, 1, 0>());

            // add top neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // load north neighbors
            odds0 = load<0>(hoodOld, FixedCoord< 0, 0, 1>());
            odds1 = load<2>(hoodOld, FixedCoord< 0, 0, 1>());
            odds2 = load<4>(hoodOld, FixedCoord< 0, 0, 1>());
            odds3 = load<6>(hoodOld, FixedCoord< 0, 0, 1>());

            // add bottom neighbors
            same.a = _mm_add_pd(same.a, buf.a);
            same.b = _mm_add_pd(same.b, buf.b);
            same.c = _mm_add_pd(same.c, buf.c);
            same.d = _mm_add_pd(same.d, buf.d);

            // add north neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // scale by 1/7
            same.a = _mm_mul_pd(same.a, oneSeventh);
            same.b = _mm_mul_pd(same.b, oneSeventh);
            same.c = _mm_mul_pd(same.c, oneSeventh);
            same.d = _mm_mul_pd(same.d, oneSeventh);

            // store results
            _mm_store_pd(&hoodNew[LibFlatArray::coord<0, 0, 0>()].temp(), same.a);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<2, 0, 0>()].temp(), same.b);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<4, 0, 0>()].temp(), same.c);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<6, 0, 0>()].temp(), same.d);

            // cycle members
            odds0 = odds4;
            same.a = same.e;
        }

        for (; hoodOld.index() < indexEnd; ++hoodOld.index(), ++hoodNew.index) {
            updateSingle(hoodOld, hoodNew);
        }
    }

    template<typename NEIGHBORHOOD, int X, int Y, int Z>
    static void load(QuadM128 *q, const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        q->a = load<0>(hood, coord);
        q->b = load<2>(hood, coord);
        q->c = load<4>(hood, coord);
        q->d = load<6>(hood, coord);
    }

    template<typename NEIGHBORHOOD, int X, int Y, int Z>
    static void load(PentaM128 *q, const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        q->b = load<2>(hood, coord);
        q->c = load<4>(hood, coord);
        q->d = load<6>(hood, coord);
        q->e = load<8>(hood, coord);
    }

    template<int OFFSET, typename NEIGHBORHOOD, int X, int Y, int Z>
    static __m128d load(const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        return load<OFFSET>(&hood[coord].temp());
    }

    template<int OFFSET>
    static __m128d load(const double *p)
    {
        return _mm_load_pd(p + OFFSET);
    }

    double temp;
};

LIBFLATARRAY_REGISTER_SOA(
    JacobiCellStreakUpdate,
    ((double)(temp))
                          )

class Jacobi3DStreakUpdate : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        using std::swap;
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        typedef SoAGrid<
            JacobiCellStreakUpdate,
            APITraits::SelectTopology<JacobiCellStreakUpdate>::Value> GridType;
        CoordBox<3> box(Coord<3>(), dim);
        GridType gridA(box, JacobiCellStreakUpdate(1.0));
        GridType gridB(box, JacobiCellStreakUpdate(2.0));
        GridType *gridOld = &gridA;
        GridType *gridNew = &gridB;

        int maxT = 20;

        Region<3> region;
        region << box;

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                typedef UpdateFunctorHelpers::Selector<
                    JacobiCellStreakUpdate>::SoARegionUpdateHelper<
                        UpdateFunctorHelpers::ConcurrencyNoP, APITraits::SelectThreadedUpdate<void>::Value> Updater;

                Coord<3> offset(1, 1, 1);
                Updater updater(&region, &offset, &offset, &box.dimensions, 0, 0, 0);
                gridNew->callback(gridOld, updater);
                swap(gridOld, gridNew);
            }
        }

        if (gridA.get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DStreakUpdateFunctor : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "platinum";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int maxT = 20;
        SerialSimulator<JacobiCellStreakUpdate> sim(
            new NoOpInitializer<JacobiCellStreakUpdate>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class LBMCell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

#define C 0
#define N 1
#define E 2
#define W 3
#define S 4
#define T 5
#define B 6

#define NW 7
#define SW 8
#define NE 9
#define SE 10

#define TW 11
#define BW 12
#define TE 13
#define BE 14

#define TN 15
#define BN 16
#define TS 17
#define BS 18

    inline explicit LBMCell(double v = 1.0, State s = LIQUID) :
        state(s)
    {
        comp[C] = v;
        for (int i = 1; i < 19; ++i) {
            comp[i] = 0.0;
        }
        density = 1.0;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned /* nanoStep */)
    {
        *this = neighborhood[FixedCoord<0, 0>()];

        switch (state) {
        case LIQUID:
            updateFluid(neighborhood);
            break;
        case WEST_NOSLIP:
            updateWestNoSlip(neighborhood);
            break;
        case EAST_NOSLIP:
            updateEastNoSlip(neighborhood);
            break;
        case TOP :
            updateTop(neighborhood);
            break;
        case BOTTOM:
            updateBottom(neighborhood);
            break;
        case NORTH_ACC:
            updateNorthAcc(neighborhood);
            break;
        case SOUTH_NOSLIP:
            updateSouthNoSlip(neighborhood);
            break;
        }
    }

    template<typename COORD_MAP>
    void updateFluid(const COORD_MAP& neighborhood)
    {
#define GET_COMP(X, Y, Z, COMP) neighborhood[Coord<3>(X, Y, Z)].comp[COMP]
#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
        const int x = 0;
        const int y = 0;
        const int z = 0;
        double velX, velY, velZ;

        velX  =
            GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
            GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
            GET_COMP(x-1,y,z+1,BE);
        velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
            GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
        velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
            GET_COMP(x+1,y,z-1,TW);

        const double rho =
            GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
            GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
            GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
            GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
        velX  = velX
            - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
            - GET_COMP(x+1,y,z+1,BW);
        velY  = velY
            + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
            - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
        velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

        density = rho;
        velocityX = velX;
        velocityX = velX;
        velocityY = velY;
        velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        comp[C]=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        comp[NW]=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        comp[SE]=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        comp[NE]=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        comp[SW]=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        comp[TW]=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        comp[BE]=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        comp[TE]=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        comp[BW]=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        comp[TS]=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        comp[BN]=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        comp[TN]=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        comp[BS]=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        comp[N]=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        comp[S]=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        comp[E]=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        comp[W]=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        comp[T]=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        comp[B]=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));

    }

    template<typename COORD_MAP>
    void updateWestNoSlip(const COORD_MAP& neighborhood)
    {
        comp[E ]=GET_COMP(1, 0,  0, W);
        comp[NE]=GET_COMP(1, 1,  0, SW);
        comp[SE]=GET_COMP(1,-1,  0, NW);
        comp[TE]=GET_COMP(1, 0,  1, BW);
        comp[BE]=GET_COMP(1, 0, -1, TW);
    }

    template<typename COORD_MAP>
    void updateEastNoSlip(const COORD_MAP& neighborhood)
    {
        comp[W ]=GET_COMP(-1, 0, 0, E);
        comp[NW]=GET_COMP(-1, 0, 1, SE);
        comp[SW]=GET_COMP(-1,-1, 0, NE);
        comp[TW]=GET_COMP(-1, 0, 1, BE);
        comp[BW]=GET_COMP(-1, 0,-1, TE);
    }

    template<typename COORD_MAP>
    void updateTop(const COORD_MAP& neighborhood)
    {
        comp[B] =GET_COMP(0,0,-1,T);
        comp[BE]=GET_COMP(1,0,-1,TW);
        comp[BW]=GET_COMP(-1,0,-1,TE);
        comp[BN]=GET_COMP(0,1,-1,TS);
        comp[BS]=GET_COMP(0,-1,-1,TN);
    }

    template<typename COORD_MAP>
    void updateBottom(const COORD_MAP& neighborhood)
    {
        comp[T] =GET_COMP(0,0,1,B);
        comp[TE]=GET_COMP(1,0,1,BW);
        comp[TW]=GET_COMP(-1,0,1,BE);
        comp[TN]=GET_COMP(0,1,1,BS);
        comp[TS]=GET_COMP(0,-1,1,BN);
    }

    template<typename COORD_MAP>
    void updateNorthAcc(const COORD_MAP& neighborhood)
    {
        const double w_1 = 0.01;
        comp[S] =GET_COMP(0,-1,0,N);
        comp[SE]=GET_COMP(1,-1,0,NW)+6*w_1*0.1;
        comp[SW]=GET_COMP(-1,-1,0,NE)-6*w_1*0.1;
        comp[TS]=GET_COMP(0,-1,1,BN);
        comp[BS]=GET_COMP(0,-1,-1,TN);
    }

    template<typename COORD_MAP>
    void updateSouthNoSlip(const COORD_MAP& neighborhood)
    {
        comp[N] =GET_COMP(0,1,0,S);
        comp[NE]=GET_COMP(1,1,0,SW);
        comp[NW]=GET_COMP(-1,1,0,SE);
        comp[TN]=GET_COMP(0,1,1,BS);
        comp[BN]=GET_COMP(0,1,-1,TS);
    }

    double comp[19];
    double density;
    double velocityX;
    double velocityY;
    double velocityZ;
    State state;

#undef C
#undef N
#undef E
#undef W
#undef S
#undef T
#undef B

#undef NW
#undef SW
#undef NE
#undef SE

#undef TW
#undef BW
#undef TE
#undef BE

#undef TN
#undef BN
#undef TS
#undef BS

#undef GET_COMP
#undef SQR
};

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

class ShortVec1xSSE
{
public:
    static const int ARITY = 2;

    inline ShortVec1xSSE() :
        val(_mm_set1_pd(0))
    {}

    inline ShortVec1xSSE(const double *addr) :
        val(_mm_loadu_pd(addr))
    {}

    inline ShortVec1xSSE(const double val) :
        val(_mm_set1_pd(val))
    {}

    inline ShortVec1xSSE(__m128d val) :
        val(val)
    {}

    inline ShortVec1xSSE operator+(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_add_pd(val, a.val));
    }

    inline ShortVec1xSSE operator-(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_sub_pd(val, a.val));
    }

    inline ShortVec1xSSE operator*(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_mul_pd(val, a.val));
    }

    inline void store(double *a) const
    {
        _mm_store_pd(a + 0, val);
    }

    __m128d val;
};

class ShortVec2xSSE
{
public:
    static const int ARITY = 4;

    inline ShortVec2xSSE() :
        val1(_mm_set1_pd(0)),
        val2(_mm_set1_pd(0))
    {}

    inline ShortVec2xSSE(const double *addr) :
        val1(_mm_loadu_pd(addr + 0)),
        val2(_mm_loadu_pd(addr + 2))
    {}

    inline ShortVec2xSSE(const double val) :
        val1(_mm_set1_pd(val)),
        val2(_mm_set1_pd(val))
    {}

    inline ShortVec2xSSE(__m128d val1, __m128d val2) :
        val1(val1),
        val2(val2)
    {}

    inline ShortVec2xSSE operator+(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_add_pd(val1, a.val1),
            _mm_add_pd(val2, a.val2));
    }

    inline ShortVec2xSSE operator-(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_sub_pd(val1, a.val1),
            _mm_sub_pd(val2, a.val2));

    }

    inline ShortVec2xSSE operator*(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_mul_pd(val1, a.val1),
            _mm_mul_pd(val2, a.val2));
    }

    inline void store(double *a) const
    {
        _mm_store_pd(a + 0, val1);
        _mm_store_pd(a + 2, val2);
    }

    __m128d val1;
    __m128d val2;
};

class ShortVec4xSSE
{
public:
    static const int ARITY = 8;

    inline ShortVec4xSSE() :
        val1(_mm_set1_pd(0)),
        val2(_mm_set1_pd(0)),
        val3(_mm_set1_pd(0)),
        val4(_mm_set1_pd(0))
    {}

    inline ShortVec4xSSE(const double *addr) :
        val1(_mm_loadu_pd(addr + 0)),
        val2(_mm_loadu_pd(addr + 2)),
        val3(_mm_loadu_pd(addr + 4)),
        val4(_mm_loadu_pd(addr + 6))
    {}

    inline ShortVec4xSSE(const double val) :
        val1(_mm_set1_pd(val)),
        val2(_mm_set1_pd(val)),
        val3(_mm_set1_pd(val)),
        val4(_mm_set1_pd(val))
    {}

    inline ShortVec4xSSE(__m128d val1, __m128d val2, __m128d val3, __m128d val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline ShortVec4xSSE operator+(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_add_pd(val1, a.val1),
            _mm_add_pd(val2, a.val2),
            _mm_add_pd(val3, a.val3),
            _mm_add_pd(val4, a.val4));
    }

    inline ShortVec4xSSE operator-(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_sub_pd(val1, a.val1),
            _mm_sub_pd(val2, a.val2),
            _mm_sub_pd(val3, a.val3),
            _mm_sub_pd(val4, a.val4));

    }

    inline ShortVec4xSSE operator*(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_mul_pd(val1, a.val1),
            _mm_mul_pd(val2, a.val2),
            _mm_mul_pd(val3, a.val3),
            _mm_mul_pd(val4, a.val4));
    }

    inline void store(double *a) const
    {
        _mm_storeu_pd(a + 0, val1);
        _mm_storeu_pd(a + 2, val2);
        _mm_storeu_pd(a + 4, val3);
        _mm_storeu_pd(a + 6, val4);
    }

private:
    __m128d val1;
    __m128d val2;
    __m128d val3;
    __m128d val4;
};

#ifdef __AVX__

class ShortVec1xAVX
{
public:
    static const int ARITY = 4;

    inline ShortVec1xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec1xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0))
    {}

    inline ShortVec1xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec1xAVX(__m256d val1) :
        val1(val1)
    {}

    inline ShortVec1xAVX operator+(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_add_pd(val1, a.val1));
    }

    inline ShortVec1xAVX operator-(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_sub_pd(val1, a.val1));

    }

    inline ShortVec1xAVX operator*(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_mul_pd(val1, a.val1));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
    }

private:
    __m256d val1;
};

class ShortVec2xAVX
{
public:
    static const int ARITY = 8;

    inline ShortVec2xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0)),
        val2(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec2xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0)),
        val2(_mm256_loadu_pd(addr +  4))
    {}

    inline ShortVec2xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val)),
        val2(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec2xAVX(__m256d val1, __m256d val2) :
        val1(val1),
        val2(val2)
    {}

    inline ShortVec2xAVX operator+(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_add_pd(val1, a.val1),
            _mm256_add_pd(val2, a.val2));
    }

    inline ShortVec2xAVX operator-(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_sub_pd(val1, a.val1),
            _mm256_sub_pd(val2, a.val2));

    }

    inline ShortVec2xAVX operator*(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_mul_pd(val1, a.val1),
            _mm256_mul_pd(val2, a.val2));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
        _mm256_storeu_pd(a +  4, val2);
    }

private:
    __m256d val1;
    __m256d val2;
};

class ShortVec4xAVX
{
public:
    static const int ARITY = 16;

    inline ShortVec4xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0)),
        val2(_mm256_set_pd(0, 0, 0, 0)),
        val3(_mm256_set_pd(0, 0, 0, 0)),
        val4(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec4xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0)),
        val2(_mm256_loadu_pd(addr +  4)),
        val3(_mm256_loadu_pd(addr +  8)),
        val4(_mm256_loadu_pd(addr + 12))
    {}

    inline ShortVec4xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val)),
        val2(_mm256_set_pd(val, val, val, val)),
        val3(_mm256_set_pd(val, val, val, val)),
        val4(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec4xAVX(__m256d val1, __m256d val2, __m256d val3, __m256d val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline ShortVec4xAVX operator+(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_add_pd(val1, a.val1),
            _mm256_add_pd(val2, a.val2),
            _mm256_add_pd(val3, a.val3),
            _mm256_add_pd(val4, a.val4));
    }

    inline ShortVec4xAVX operator-(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_sub_pd(val1, a.val1),
            _mm256_sub_pd(val2, a.val2),
            _mm256_sub_pd(val3, a.val3),
            _mm256_sub_pd(val4, a.val4));

    }

    inline ShortVec4xAVX operator*(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_mul_pd(val1, a.val1),
            _mm256_mul_pd(val2, a.val2),
            _mm256_mul_pd(val3, a.val3),
            _mm256_mul_pd(val4, a.val4));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
        _mm256_storeu_pd(a +  4, val2);
        _mm256_storeu_pd(a +  8, val3);
        _mm256_storeu_pd(a + 12, val4);
    }

private:
    __m256d val1;
    __m256d val2;
    __m256d val3;
    __m256d val4;
};

#endif

#ifdef __ICC
#pragma warning pop
#endif

template<typename VEC>
void store(double *a, VEC v)
{
    v.store(a);
}

class LBMSoACell
{
public:
    // typedef ShortVec1xSSE Double;
    // typedef ShortVec2xSSE Double;
    typedef ShortVec4xSSE Double;
    // typedef ShortVec1xAVX Double;
    // typedef ShortVec2xAVX Double;
    // typedef ShortVec4xAVX Double;

    class API : public APITraits::HasFixedCoordsOnlyUpdate,
                public APITraits::HasSoA,
                public APITraits::HasUpdateLineX,
                public APITraits::HasStencil<Stencils::Moore<3, 1> >,
                public APITraits::HasCubeTopology<3>
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

    inline explicit LBMSoACell(double v=1.0, const State& s=LIQUID) :
        C(v),
        N(0),
        E(0),
        W(0),
        S(0),
        T(0),
        B(0),

        NW(0),
        SW(0),
        NE(0),
        SE(0),

        TW(0),
        BW(0),
        TE(0),
        BE(0),

        TN(0),
        BN(0),
        TS(0),
        BS(0),

        density(1.0),
        velocityX(0),
        velocityY(0),
        velocityZ(0),
        state(s)
    {
    }

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineX(
        ACCESSOR1& hoodOld, int indexEnd,
        ACCESSOR2& hoodNew, int nanoStep)
    {
        updateLineXFluid(hoodOld, indexEnd, hoodNew);
    }



//     template<typename COORD_MAP>
//     void update(const COORD_MAP& neighborhood, unsigned nanoStep)
//     {
//         *this = neighborhood[FixedCoord<0, 0>()];

//         switch (state) {
//         case LIQUID:
//             updateFluid(neighborhood);
//             break;
//         case WEST_NOSLIP:
//             updateWestNoSlip(neighborhood);
//             break;
//         case EAST_NOSLIP:
//             updateEastNoSlip(neighborhood);
//             break;
//         case TOP :
//             updateTop(neighborhood);
//             break;
//         case BOTTOM:
//             updateBottom(neighborhood);
//             break;
//         case NORTH_ACC:
//             updateNorthAcc(neighborhood);
//             break;
//         case SOUTH_NOSLIP:
//             updateSouthNoSlip(neighborhood);
//             break;
//         }
//     }

// private:
    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineXFluid(
        ACCESSOR1& hoodOld, int indexEnd,
        ACCESSOR2& hoodNew)
    {
#define GET_COMP(X, Y, Z, COMP) Double(&hoodOld[FixedCoord<X, Y, Z>()].COMP())
#define SQR(X) ((X)*(X))
        const Double omega = 1.0/1.7;
        const Double omega_trm = Double(1.0) - omega;
        const Double omega_w0 = Double(3.0 * 1.0 / 3.0) * omega;
        const Double omega_w1 = Double(3.0*1.0/18.0)*omega;
        const Double omega_w2 = Double(3.0*1.0/36.0)*omega;
        const Double one_third = 1.0 / 3.0;
        const Double one_half = 0.5;
        const Double one_point_five = 1.5;

        const int x = 0;
        const int y = 0;
        const int z = 0;
        Double velX, velY, velZ;

        for (; hoodOld.index() < indexEnd; hoodNew.index += Double::ARITY, hoodOld.index() += Double::ARITY) {
            velX  =
                GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
                GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
                GET_COMP(x-1,y,z+1,BE);
            velY  =
                GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
                GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
            velZ  =
                GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
                GET_COMP(x+1,y,z-1,TW);

            const Double rho =
                GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
                GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
                GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
                GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
            velX  = velX
                - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
                - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
                - GET_COMP(x+1,y,z+1,BW);
            velY  = velY
                + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
                - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
                - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
            velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

            store(&hoodNew.density(), rho);
            store(&hoodNew.velocityX(), velX);
            store(&hoodNew.velocityY(), velY);
            store(&hoodNew.velocityZ(), velZ);

            const Double dir_indep_trm = one_third*rho - one_half *( velX*velX + velY*velY + velZ*velZ );

            store(&hoodNew.C(), omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm ));

            store(&hoodNew.NW(), omega_trm * GET_COMP(x+1,y-1,z,NW) + omega_w2*( dir_indep_trm - ( velX-velY ) + one_point_five * SQR( velX-velY ) ));
            store(&hoodNew.SE(), omega_trm * GET_COMP(x-1,y+1,z,SE) + omega_w2*( dir_indep_trm + ( velX-velY ) + one_point_five * SQR( velX-velY ) ));
            store(&hoodNew.NE(), omega_trm * GET_COMP(x-1,y-1,z,NE) + omega_w2*( dir_indep_trm + ( velX+velY ) + one_point_five * SQR( velX+velY ) ));
            store(&hoodNew.SW(), omega_trm * GET_COMP(x+1,y+1,z,SW) + omega_w2*( dir_indep_trm - ( velX+velY ) + one_point_five * SQR( velX+velY ) ));

            store(&hoodNew.TW(), omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + one_point_five * SQR( velX-velZ ) ));
            store(&hoodNew.BE(), omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + one_point_five * SQR( velX-velZ ) ));
            store(&hoodNew.TE(), omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + one_point_five * SQR( velX+velZ ) ));
            store(&hoodNew.BW(), omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + one_point_five * SQR( velX+velZ ) ));

            store(&hoodNew.TS(), omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + one_point_five * SQR( velY-velZ ) ));
            store(&hoodNew.BN(), omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + one_point_five * SQR( velY-velZ ) ));
            store(&hoodNew.TN(), omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + one_point_five * SQR( velY+velZ ) ));
            store(&hoodNew.BS(), omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + one_point_five * SQR( velY+velZ ) ));

            store(&hoodNew.N(), omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + one_point_five * SQR(velY)));
            store(&hoodNew.S(), omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + one_point_five * SQR(velY)));
            store(&hoodNew.E(), omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + one_point_five * SQR(velX)));
            store(&hoodNew.W(), omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + one_point_five * SQR(velX)));
            store(&hoodNew.T(), omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + one_point_five * SQR(velZ)));
            store(&hoodNew.B(), omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + one_point_five * SQR(velZ)));
        }
    }

//     template<typename ACCESSOR1, typename ACCESSOR2>
//     static void updateLineXFluid(
//         ACCESSOR1 hoodOld, int hoodOld.index(), int indexEnd, ACCESSOR2 hoodNew, int *indexNew)
//     {
// #define GET_COMP(X, Y, Z, COMP) hoodOld[FixedCoord<X, Y, Z>()].COMP()
// #define SQR(X) ((X)*(X))
//         const double omega = 1.0/1.7;
//         const double omega_trm = 1.0 - omega;
//         const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
//         const double omega_w1 = 3.0*1.0/18.0*omega;
//         const double omega_w2 = 3.0*1.0/36.0*omega;
//         const double one_third = 1.0 / 3.0;
//         const int x = 0;
//         const int y = 0;
//         const int z = 0;
//         double velX, velY, velZ;

//         for (; hoodOld.index() < indexEnd; ++hoodOld.index()) {
//             velX  =
//                 GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
//                 GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
//                 GET_COMP(x-1,y,z+1,BE);
//             velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
//                 GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
//             velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
//                 GET_COMP(x+1,y,z-1,TW);

//             const double rho =
//                 GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
//                 GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
//                 GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
//                 GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
//             velX  = velX
//                 - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
//                 - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
//                 - GET_COMP(x+1,y,z+1,BW);
//             velY  = velY
//                 + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
//                 - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
//                 - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
//             velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

//             hoodNew.density() = rho;
//             hoodNew.velocityX() = velX;
//             hoodNew.velocityY() = velY;
//             hoodNew.velocityZ() = velZ;

//             const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

//             hoodNew.C()=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

//             hoodNew.NW()=omega_trm * GET_COMP(x+1,y-1,z,NW) +
//                 omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             hoodNew.SE()=omega_trm * GET_COMP(x-1,y+1,z,SE) +
//                 omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             hoodNew.NE()=omega_trm * GET_COMP(x-1,y-1,z,NE) +
//                 omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
//             hoodNew.SW()=omega_trm * GET_COMP(x+1,y+1,z,SW) +
//                 omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

//             hoodNew.TW()=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             hoodNew.BE()=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             hoodNew.TE()=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
//             hoodNew.BW()=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

//             hoodNew.TS()=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             hoodNew.BN()=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             hoodNew.TN()=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
//             hoodNew.BS()=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

//             hoodNew.N()=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
//             hoodNew.S()=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
//             hoodNew.E()=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
//             hoodNew.W()=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
//             hoodNew.T()=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
//             hoodNew.B()=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));

//             ++*indexNew;
//         }
//     }

//     template<typename COORD_MAP>
//     void updateWestNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[E ]=GET_COMP(1, 0,  0, W);
//         comp[NE]=GET_COMP(1, 1,  0, SW);
//         comp[SE]=GET_COMP(1,-1,  0, NW);
//         comp[TE]=GET_COMP(1, 0,  1, BW);
//         comp[BE]=GET_COMP(1, 0, -1, TW);
//     }

//     template<typename COORD_MAP>
//     void updateEastNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[W ]=GET_COMP(-1, 0, 0, E);
//         comp[NW]=GET_COMP(-1, 0, 1, SE);
//         comp[SW]=GET_COMP(-1,-1, 0, NE);
//         comp[TW]=GET_COMP(-1, 0, 1, BE);
//         comp[BW]=GET_COMP(-1, 0,-1, TE);
//     }

//     template<typename COORD_MAP>
//     void updateTop(const COORD_MAP& neighborhood)
//     {
//         comp[B] =GET_COMP(0,0,-1,T);
//         comp[BE]=GET_COMP(1,0,-1,TW);
//         comp[BW]=GET_COMP(-1,0,-1,TE);
//         comp[BN]=GET_COMP(0,1,-1,TS);
//         comp[BS]=GET_COMP(0,-1,-1,TN);
//     }

//     template<typename COORD_MAP>
//     void updateBottom(const COORD_MAP& neighborhood)
//     {
//         comp[T] =GET_COMP(0,0,1,B);
//         comp[TE]=GET_COMP(1,0,1,BW);
//         comp[TW]=GET_COMP(-1,0,1,BE);
//         comp[TN]=GET_COMP(0,1,1,BS);
//         comp[TS]=GET_COMP(0,-1,1,BN);
//     }

//     template<typename COORD_MAP>
//     void updateNorthAcc(const COORD_MAP& neighborhood)
//     {
//         const double w_1 = 0.01;
//         comp[S] =GET_COMP(0,-1,0,N);
//         comp[SE]=GET_COMP(1,-1,0,NW)+6*w_1*0.1;
//         comp[SW]=GET_COMP(-1,-1,0,NE)-6*w_1*0.1;
//         comp[TS]=GET_COMP(0,-1,1,BN);
//         comp[BS]=GET_COMP(0,-1,-1,TN);
//     }

//     template<typename COORD_MAP>
//     void updateSouthNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[N] =GET_COMP(0,1,0,S);
//         comp[NE]=GET_COMP(1,1,0,SW);
//         comp[NW]=GET_COMP(-1,1,0,SE);
//         comp[TN]=GET_COMP(0,1,1,BS);
//         comp[BN]=GET_COMP(0,1,-1,TS);
//     }

//     double comp[19];
//     double density;
//     double velocityX;
//     double velocityY;
//     double velocityZ;
//     State state;


    double C;
    double N;
    double E;
    double W;
    double S;
    double T;
    double B;

    double NW;
    double SW;
    double NE;
    double SE;

    double TW;
    double BW;
    double TE;
    double BE;

    double TN;
    double BN;
    double TS;
    double BS;

    double density;
    double velocityX;
    double velocityY;
    double velocityZ;
    State state;

};

LIBFLATARRAY_REGISTER_SOA(
    LBMSoACell,
    ((double)(C))
    ((double)(N))
    ((double)(E))
    ((double)(W))
    ((double)(S))
    ((double)(T))
    ((double)(B))
    ((double)(NW))
    ((double)(SW))
    ((double)(NE))
    ((double)(SE))
    ((double)(TW))
    ((double)(BW))
    ((double)(TE))
    ((double)(BE))
    ((double)(TN))
    ((double)(BN))
    ((double)(TS))
    ((double)(BS))
    ((double)(density))
    ((double)(velocityX))
    ((double)(velocityY))
    ((double)(velocityZ))
    ((LBMSoACell::State)(state))
                          )

class LBMClassic : public CPUBenchmark
{
public:
    std::string family()
    {
        return "LBM";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int maxT = 20;
        SerialSimulator<LBMCell> sim(
            new NoOpInitializer<LBMCell>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).density == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class LBMSoA : public CPUBenchmark
{
public:
    std::string family()
    {
        return "LBM";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        int maxT = 20;
        OpenMPSimulator<LBMSoACell> sim(
            new NoOpInitializer<LBMSoACell>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).density == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

template<class PARTITION>
class PartitionBenchmark : public CPUBenchmark
{
public:
    explicit PartitionBenchmark(const std::string& name) :
        name(name)
    {}

    std::string species()
    {
        return "gold";
    }

    std::string family()
    {
        return name;
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        double duration = 0;
        Coord<2> accu;
        Coord<2> realDim(dim.x(), dim.y());

        {
            ScopedTimer t(&duration);

            PARTITION h(Coord<2>(100, 200), realDim);
            typename PARTITION::Iterator end = h.end();
            for (typename PARTITION::Iterator i = h.begin(); i != end; ++i) {
                accu += *i;
            }
        }

        if (accu == Coord<2>()) {
            throw std::runtime_error("oops, partition iteration went bad!");
        }

        return duration;
    }

    std::string unit()
    {
        return "s";
    }

private:
    std::string name;
};

#ifdef LIBGEODECOMP_WITH_CPP14
typedef double ValueType;
static const std::size_t MATRICES = 1;
static const int C = 4;         // AVX
static const int SIGMA = 1;
typedef short_vec<ValueType, C> ShortVec;

class SPMVMCell
{
public:
    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<ValueType>,
        public APITraits::HasSellMatrices<MATRICES>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {};

    inline explicit SPMVMCell(double v = 8.0) :
        value(v), sum(0)
    {}

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    double value;
    double sum;
};

class SPMVMCellStreak
{
public:
    class API :
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<ValueType>,
        public APITraits::HasSellMatrices<MATRICES>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {};

    inline explicit SPMVMCellStreak(double v = 8.0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            hoodNew[i].sum = 0.;
            for (const auto& j: hoodOld.weights(0)) {
                hoodNew[i].sum += hoodOld[j.first].value * j.second;
            }
        }
    }

    double value;
    double sum;
};

class SPMVMSoACell
{
public:
    class API :
        public APITraits::HasSoA,
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<ValueType>,
        public APITraits::HasSellMatrices<MATRICES>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {
    public:
        // uniform sizes lead to std::bad_alloc,
        // since UnstructuredSoAGrid uses (dim.x(), 1, 1)
        // as dimension (DIM = 1)
        LIBFLATARRAY_CUSTOM_SIZES(
            (16)(32)(64)(128)(256)(512)(1024)(2048)(4096)(8192)(16384)(32768)
            (65536)(131072)(262144)(524288)(1048576),
            (1),
            (1))
    };

    inline explicit SPMVMSoACell(double v = 8.0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (int i = hoodOld.index(); i < indexEnd / C; ++i, ++hoodOld) {
            ShortVec tmp;
            tmp.load_aligned(&hoodNew->sum() + i * C);
            for (const auto& j: hoodOld.weights(0)) {
                ShortVec weights, values;
                weights.load_aligned(j.second);
                values.gather(&hoodOld->value(), j.first);
                tmp += values * weights;
            }
            tmp.store_aligned(&hoodNew->sum() + i * C);
        }
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    double value;
    double sum;
};

LIBFLATARRAY_REGISTER_SOA(SPMVMSoACell, ((double)(sum))((double)(value)))

class SPMVMSoACellInf
{
public:
    using REAL = ShortVec;

    class API :
        public APITraits::HasSoA,
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<ValueType>,
        public APITraits::HasSellMatrices<MATRICES>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {
    public:
        // uniform sizes lead to std::bad_alloc,
        // since UnstructuredSoAGrid uses (dim.x(), 1, 1)
        // as dimension (DIM = 1)
        LIBFLATARRAY_CUSTOM_SIZES(
            (16)(32)(64)(128)(256)(512)(1024)(2048)(4096)(8192)(16384)(32768)
            (65536)(131072)(262144)(524288)(1048576),
            (1),
            (1))
    };

    inline explicit SPMVMSoACellInf(double v = 8.0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        REAL tmp, weights, values;
        for (int i = hoodOld.index(); i < (indexEnd / C); ++i, ++hoodOld) {
            tmp = &hoodNew->sum() + i * C;
            for (const auto& j: hoodOld.weights(0)) {
                weights = j.second;
                values.gather(&hoodOld->value(), j.first);
                tmp += values * weights;
            }
            (&hoodNew->sum() + i * C) << tmp;
        }
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    double value;
    double sum;
};

LIBFLATARRAY_REGISTER_SOA(SPMVMSoACellInf, ((double)(sum))((double)(value)))

// setup a sparse matrix
template<typename CELL, typename GRID>
class SparseMatrixInitializer : public SimpleInitializer<CELL>
{
private:
    int size;

public:
    inline
    SparseMatrixInitializer(const Coord<3>& dim, int maxT) :
        SimpleInitializer<CELL>(Coord<1>(dim.x()), maxT),
        size(dim.x())
    {}

    virtual void grid(GridBase<CELL, 1> *grid)
    {
        // setup sparse matrix
        std::map<Coord<2>, ValueType> weights;

        // setup matrix: ~1 % non zero entries
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size / 100; ++col) {
                weights[Coord<2>(row, col * 100)] = 5.0;
            }
        }

        grid->setWeights(0, weights);

        // setup rhs: not needed, since the grid is intialized with default cells
        // default value of SPMVMCell is 8.0
    }
};

class SellMatrixInitializer : public CPUBenchmark
{
    public:
    std::string family()
    {
        return "SELLInit";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        const Coord<1> dim1d(dim.x());
        const int size = dim.x();
        UnstructuredGrid<SPMVMCell, MATRICES, ValueType, C, SIGMA> grid(dim1d);
        std::map<Coord<2>, ValueType> weights;

        // setup matrix: ~1 % non zero entries
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size / 100; ++col) {
                weights[Coord<2>(row, col * 100)] = 5.0;
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            grid.setWeights(0, weights);
        }

        if (grid.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class SparseMatrixVectorMultiplication : public CPUBenchmark
{
private:
    template<typename CELL, typename GRID>
    void updateFunctor(const Streak<1>& streak, const GRID& gridOld,
                       GRID *gridNew, unsigned nanoStep)
    {
        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodOld(gridOld, streak.origin.x());
        CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodNew(*gridNew);

        // call update()
        for (int i = hoodOld.index(); i < streak.endX; ++i, ++hoodOld) {
            hoodNew[i].update(hoodOld, nanoStep);
        }
    }

public:
    std::string family()
    {
        return "SPMVM";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        // 1. create grids
        typedef UnstructuredGrid<SPMVMCell, MATRICES, ValueType, C, SIGMA> Grid;
        const Coord<1> size(dim.x());
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializer<SPMVMCell, Grid> init(dim, maxT);
        init.grid(&gridOld);

        // 3. call updateFunctor()
        double seconds = 0;
        Streak<1> streak(Coord<1>(0), size.x());
        {
            ScopedTimer t(&seconds);
            updateFunctor<SPMVMCell, Grid>(streak, gridOld, &gridNew, 0);
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * (size.x() / 100) * (size.x());
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};

class SparseMatrixVectorMultiplicationVectorized : public CPUBenchmark
{
private:
    template<typename CELL, typename GRID>
    void updateFunctor(
        const Region<1>& region,
        const GRID& gridOld,
        GRID *gridNew,
        unsigned nanoStep)
    {
        gridOld.callback(
            gridNew,
            UnstructuredUpdateFunctorHelpers::UnstructuredGridSoAUpdateHelper<CELL>(
                gridOld, gridNew, region, nanoStep));
    }

public:
    std::string family()
    {
        return "SPMVM";
    }

    std::string species()
    {
        return "platinum";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        // 1. create grids
        typedef UnstructuredSoAGrid<SPMVMSoACell, MATRICES, ValueType, C, SIGMA> Grid;
        const CoordBox<1> size(Coord<1>(0), Coord<1>(dim.x()));
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializer<SPMVMSoACell, Grid> init(dim, maxT);
        init.grid(&gridOld);

        // 3. call updateFunctor()
        double seconds = 0;
        Region<1> region;
        region << Streak<1>(Coord<1>(0), size.dimensions.x());
        {
            ScopedTimer t(&seconds);
            updateFunctor<SPMVMSoACell, Grid>(region, gridOld, &gridNew, 0);
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * (size.dimensions.x() / 100) * (size.dimensions.x());
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};

class SparseMatrixVectorMultiplicationVectorizedInf : public CPUBenchmark
{
private:
    template<typename CELL, typename GRID>
    void updateFunctor(const Region<1>& region, const GRID& gridOld,
                       GRID *gridNew, unsigned nanoStep)
    {
        gridOld.callback(
            gridNew,
            UnstructuredUpdateFunctorHelpers::UnstructuredGridSoAUpdateHelper<CELL>(
                gridOld,
                gridNew,
                region,
                nanoStep));
    }

public:
    std::string family()
    {
        return "SPMVM";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        // 1. create grids
        typedef UnstructuredSoAGrid<SPMVMSoACellInf, MATRICES, ValueType, C, SIGMA> Grid;
        const CoordBox<1> size(Coord<1>(0), Coord<1>(dim.x()));
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializer<SPMVMSoACellInf, Grid> init(dim, maxT);
        init.grid(&gridOld);

        // 3. call updateFunctor()
        double seconds = 0;
        Region<1> region;
        region << Streak<1>(Coord<1>(0), size.dimensions.x());
        {
            ScopedTimer t(&seconds);
            updateFunctor<SPMVMSoACellInf, Grid>(region, gridOld, &gridNew, 0);
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * (size.dimensions.x() / 100) * (size.dimensions.x());
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};

#ifdef __AVX__
class SparseMatrixVectorMultiplicationNative : public CPUBenchmark
{
private:
    // callback to get cell's member pointer
    template<typename CELL, typename VALUE_TYPE>
    class GetPointer
    {
    private:
        VALUE_TYPE **sumPtr;
        VALUE_TYPE **valuePtr;
    public:
        GetPointer(VALUE_TYPE **sumPtr, VALUE_TYPE **valuePtr) :
            sumPtr(sumPtr), valuePtr(valuePtr)
        {}

        template<
            typename CELL1, long MY_DIM_X1, long MY_DIM_Y1, long MY_DIM_Z1, long INDEX1,
            typename CELL2, long MY_DIM_X2, long MY_DIM_Y2, long MY_DIM_Z2, long INDEX2>
        void operator()(
            LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& oldAccessor,
            LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& newAccessor) const
        {
            *sumPtr = &newAccessor.sum();
            *valuePtr = &oldAccessor.value();
        }
    };

public:
    std::string family()
    {
        return "SPMVM";
    }

    std::string species()
    {
        return "pepper";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
        // 1. create grids
        typedef UnstructuredSoAGrid<SPMVMSoACell, MATRICES, ValueType, C, SIGMA> Grid;
        typedef SellCSigmaSparseMatrixContainer<ValueType, C, SIGMA> Matrix;
        const CoordBox<1> size(Coord<1>(0), Coord<1>(dim.x()));
        Grid gridOld(size);
        Grid gridNew(size);

        // 2. init grid old
        const int maxT = 1;
        SparseMatrixInitializer<SPMVMSoACell, Grid> init(dim, maxT);
        init.grid(&gridOld);

        // 3. native kernel
        const Matrix& matrix = gridOld.getWeights(0);
        const ValueType *values = matrix.valuesVec().data();
        const int *cl = matrix.chunkLengthVec().data();
        const int *cs = matrix.chunkOffsetVec().data();
        const int *col = matrix.columnVec().data();
        ValueType *rhsPtr; // = hoodOld.valuePtr;
        ValueType *resPtr; // = hoodNew.sumPtr;
        gridOld.callback(&gridNew, GetPointer<SPMVMSoACell, ValueType>(&resPtr, &rhsPtr));
        const int rowsPadded = ((size.dimensions.x() - 1) / C + 1) * C;
        double seconds = 0;
        {
            ScopedTimer t(&seconds);
            for (int i = 0; i < rowsPadded / C; ++i) {
                int offs = cs[i];
                __m256d tmp = _mm256_load_pd(resPtr + i*C);
                for (int j = 0; j < cl[i]; ++j) {
                    __m256d rhs;
                    __m256d val;
                    rhs = _mm256_set_pd(
                        *(rhsPtr + col[offs + 3]),
                        *(rhsPtr + col[offs + 2]),
                        *(rhsPtr + col[offs + 1]),
                        *(rhsPtr + col[offs + 0]));
                    val    = _mm256_load_pd(values + offs);
                    tmp    = _mm256_add_pd(tmp, _mm256_mul_pd(val, rhs));
                    offs += 4;
                }
                _mm256_store_pd(resPtr + i*C, tmp);
            }
        }

        if (gridNew.get(Coord<1>(1)).sum == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        const double numOps = 2. * (size.dimensions.x() / 100) * (size.dimensions.x());
        const double gflops = 1.0e-9 * numOps / seconds;
        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};
#endif
#endif

#ifdef LIBGEODECOMP_WITH_CUDA
void cudaTests(std::string name, std::string revision, int cudaDevice);
#endif

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc == 4) || (argc > 5)) {
        std::cerr << "usage: " << argv[0] << " [-n,--name SUBSTRING] REVISION CUDA_DEVICE \n"
                  << "  - optional: only run tests whose name contains a SUBSTRING,\n"
                  << "  - REVISION is purely for output reasons,\n"
                  << "  - CUDA_DEVICE causes CUDA tests to run on the device with the given ID.\n";
        return 1;
    }

    std::string name = "";
    int argumentIndex = 1;
    if (argc == 5) {
        if ((std::string(argv[1]) == "-n") ||
            (std::string(argv[1]) == "--name")) {
            name = std::string(argv[2]);
        }
        argumentIndex = 3;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    LibFlatArray::evaluate eval(name, revision);
    eval.print_header();

    std::vector<Coord<3> > sizes;

#ifdef LIBGEODECOMP_WITH_CPP14
    sizes << Coord<3>(10648 , 1, 1)
          << Coord<3>(35937 , 1, 1)
          << Coord<3>(85184 , 1, 1);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(SellMatrixInitializer(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(SparseMatrixVectorMultiplication(), toVector(sizes[i]));
    }

#ifdef __AVX__
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(SparseMatrixVectorMultiplicationNative(), toVector(sizes[i]));
    }
#endif

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(SparseMatrixVectorMultiplicationVectorized(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(SparseMatrixVectorMultiplicationVectorizedInf(), toVector(sizes[i]));
    }
    sizes.clear();
#endif

    eval(RegionCount(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionCount(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionCount(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionInsert(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionInsert(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionInsert(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionIntersect(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionIntersect(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionIntersect(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionSubtract(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionSubtract(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionSubtract(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionUnion(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionUnion(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionUnion(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionAppend(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionAppend(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionAppend(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionExpand(1), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionExpand(1), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionExpand(1), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionExpand(5), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionExpand(5), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionExpand(5), toVector(Coord<3>(2048, 2048, 2048)));

    {
        std::vector<int> params(4);
        int numCells = 2000000;
        std::map<int, ConvexPolytope<FloatCoord<2> > > cells = RegionExpandWithAdjacency::genGrid(numCells);
        params[0] = numCells;
        params[1] = 0; // skip cells
        params[2] = 1; // expansion width
        params[3] = -1; // id streak lenght
        eval(RegionExpandWithAdjacency(cells), params);
        params[1] = 5000; // skip cells
        params[3] = 500; // id streak lenght
        eval(RegionExpandWithAdjacency(cells), params);

        params[1] = 100;
        params[2] = 50;
        params[3] = 100;
        eval(RegionExpandWithAdjacency(cells), params);

        params[2] = 20;
        params[3] = 10;
        eval(RegionExpandWithAdjacency(cells), params);

        params[2] = 10;
        params[3] = 2;
        eval(RegionExpandWithAdjacency(cells), params);
    }

    eval(CoordEnumerationVanilla(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationVanilla(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationVanilla(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(CoordEnumerationBronze(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationBronze(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationBronze(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(CoordEnumerationGold(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationGold(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationGold(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(FloatCoordAccumulationGold(), toVector(Coord<3>(2048, 2048, 2048)));

    sizes << Coord<3>(22, 22, 22)
          << Coord<3>(64, 64, 64)
          << Coord<3>(68, 68, 68)
          << Coord<3>(106, 106, 106)
          << Coord<3>(128, 128, 128)
          << Coord<3>(150, 150, 150)
          << Coord<3>(512, 512, 32)
          << Coord<3>(518, 518, 32);

    sizes << Coord<3>(1024, 1024, 32)
          << Coord<3>(1026, 1026, 32);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DVanilla(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DSSE(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DClassic(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DFixedHood(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DStreakUpdate(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DStreakUpdateFunctor(), toVector(sizes[i]));
    }

    sizes.clear();

    sizes << Coord<3>(22, 22, 22)
          << Coord<3>(64, 64, 64)
          << Coord<3>(68, 68, 68)
          << Coord<3>(106, 106, 106)
          << Coord<3>(128, 128, 128);

    sizes << Coord<3>(160, 160, 160);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(LBMClassic(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(LBMSoA(), toVector(sizes[i]));
    }

    std::vector<int> dim = toVector(Coord<3>(32 * 1024, 32 * 1024, 1));
    eval(PartitionBenchmark<HIndexingPartition   >("PartitionHIndexing"), dim);
    eval(PartitionBenchmark<StripingPartition<2> >("PartitionStriping"),  dim);
    eval(PartitionBenchmark<HilbertPartition     >("PartitionHilbert"),   dim);
    eval(PartitionBenchmark<ZCurvePartition<2>   >("PartitionZCurve"),    dim);

#ifdef LIBGEODECOMP_WITH_CUDA
    cudaTests(name, revision, cudaDevice);
#endif

    return 0;
}
