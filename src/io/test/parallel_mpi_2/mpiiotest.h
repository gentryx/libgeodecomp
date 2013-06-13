#include <boost/date_time/posix_time/posix_time.hpp>
#include <unistd.h>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/mpiio.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPIIOTest : public CxxTest::TestSuite
{
public:
    void testReadWrite()
    {
        int width = 5;
        int height = 3;
        int depth = 7;
        unsigned step = 4;
        unsigned maxSteps = 11;
        int rank = MPILayer().rank();
        int sz1, sz2, ez1, ez2;

        if (rank == 0) {
            sz1 = 0;
            ez1 = 3;

            sz2 = 5;
            ez2 = 7;
        } else {
            sz1 = 3;
            ez1 = 7;

            sz2 = 0;
            ez2 = 5;
        }

        std::string filename = TempFile::parallel("mpiio");

        Grid<double, Topologies::Cube<3>::Topology> grid1(
            Coord<3>(width, height, depth), -2);
        for (int z = 0; z < depth; ++z)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    grid1[Coord<3>(x, y, z)] = z * 100 + y * 10 + x;

        Region<3> region;
        for (int z = sz1; z < ez1; ++z)
            for (int y = 0; y < height; ++y)
                region << Streak<3>(Coord<3>(0, y, z), width);
        MPIIO<double, Topologies::Cube<3>::Topology>::writeRegion(
            grid1, grid1.getDimensions(), step, maxSteps, filename, region);

        Grid<double, Topologies::Cube<3>::Topology> grid2(
            Coord<3>(width, height, depth), -1);

        region.clear();
        for (int z = sz2; z < ez2; ++z)
            for (int y = 0; y < height; ++y)
                region << Streak<3>(Coord<3>(0, y, z), width);
        MPIIO<double, Topologies::Cube<3>::Topology>::readRegion(&grid2, filename, region);

        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    double expected = (z >= sz2) && (z < ez2) ?
                        z * 100 + y * 10 + x :
                        -1;
                    TS_ASSERT_EQUALS(expected, grid2[Coord<3>(x, y, z)]);
                }
            }
        }
    }
};

}
