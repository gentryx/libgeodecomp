#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/storage/grid.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <unistd.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPIIOTest : public CxxTest::TestSuite
{
public:
    void testBasicReadWrite2D()
    {
        int width = 5;
        int height = 3;
        unsigned step = 42;
        unsigned maxSteps = 4711;
        std::string filename = TempFile::parallel("mpiio");

        Grid<double> grid1(Coord<2>(width, height));
        grid1.getEdgeCell() = -1.245;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                grid1[Coord<2>(x, y)] = y * 10 + x;
        Region<2> region;
        for (int y = 0; y < height; ++y)
            region << Streak<2>(Coord<2>(0, y), width);
        MPIIO<double, Topologies::Cube<2>::Topology>::writeRegion(
            grid1, grid1.getDimensions(), step, maxSteps, filename, region);

        Coord<2> dimensions;
        unsigned s;
        unsigned ms;
        MPIIO<double, Topologies::Cube<2>::Topology>::readMetadata(
            &dimensions, &s, &ms, filename);
        TS_ASSERT_EQUALS(Coord<2>(width, height), dimensions);
        TS_ASSERT_EQUALS(s, step);
        TS_ASSERT_EQUALS(ms, maxSteps);

        Grid<double> grid2(dimensions);
        TS_ASSERT_DIFFERS(grid1, grid2);
        MPIIO<double, Topologies::Cube<2>::Topology>::readRegion(&grid2, filename, region);
        TS_ASSERT_EQUALS(grid1, grid2);
    }

    void testBasicReadWrite3D()
    {
        int width = 5;
        int height = 3;
        int depth = 7;
        unsigned maxSteps = 47;
        unsigned step = 11;
        std::string filename = TempFile::parallel("mpiio");

        Grid<double, Topologies::Cube<3>::Topology> grid1(
            Coord<3>(width, height, depth));
        for (int z = 0; z < depth; ++z)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    grid1[Coord<3>(x, y, z)] = z * 100 + y * 10 + x;

        Region<3> region;
        for (int z = 0; z < depth; ++z)
            for (int y = 0; y < height; ++y)
                region << Streak<3>(Coord<3>(0, y, z), width);

        MPIIO<double, Topologies::Cube<3>::Topology>::writeRegion(
            grid1, grid1.getDimensions(), step, maxSteps, filename, region);

        Coord<3> dimensions;
        unsigned s;
        unsigned ms;
        MPIIO<double, Topologies::Cube<3>::Topology>::readMetadata(
            &dimensions, &s, &ms, filename);
        TS_ASSERT_EQUALS(Coord<3>(width, height, depth), dimensions);
        TS_ASSERT_EQUALS(step, s);
        TS_ASSERT_EQUALS(maxSteps, ms);

        Grid<double, Topologies::Cube<3>::Topology> grid2(dimensions);
        TS_ASSERT_DIFFERS(grid1, grid2);
        MPIIO<double, Topologies::Cube<3>::Topology>::readRegion(&grid2, filename, region);
        TS_ASSERT_EQUALS(grid1, grid2);
    }

    void testAdvancedReadWrite()
    {
        int width = 9;
        int height = 3;
        int depth = 4;
        unsigned maxSteps = 31;
        unsigned step = 7;
        std::string filename = TempFile::parallel("mpiio");

        Grid<double, Topologies::Cube<3>::Topology> grid1(
            Coord<3>(width, height, depth));
        for (int z = 0; z < depth; ++z)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    grid1[Coord<3>(x, y, z)] = z * 100 + y * 10 + x;

        Region<3> region;
        for (int z = 0; z < depth; ++z)
            for (int y = 0; y < height; ++y)
                region << Streak<3>(Coord<3>(y + z, y, z), width);

        MPIIO<double, Topologies::Cube<3>::Topology>::writeRegion(
            grid1, grid1.getDimensions(), step, maxSteps, filename, region);

        Coord<3> dimensions;
        unsigned s;
        unsigned ms;
        MPIIO<double, Topologies::Cube<3>::Topology>::readMetadata(
            &dimensions, &s, &ms, filename);
        TS_ASSERT_EQUALS(Coord<3>(width, height, depth), dimensions);
        TS_ASSERT_EQUALS(step, s);
        TS_ASSERT_EQUALS(maxSteps, ms);

        Grid<double, Topologies::Cube<3>::Topology> grid2(dimensions, -1);
        TS_ASSERT_DIFFERS(grid1, grid2);
        MPIIO<double, Topologies::Cube<3>::Topology>::readRegion(&grid2, filename, region);

        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double expected = (x < (y + z)) ?
                        -1 : (z * 100 + y * 10 + x);
                    TS_ASSERT_EQUALS(expected, grid2[Coord<3>(x, y, z)]);
                }
            }
        }
    }
};

}
