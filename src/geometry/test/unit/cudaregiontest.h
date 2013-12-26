#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/cudaregion.h>
#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/displacedgrid.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

__global__
void setIDs2D(CoordBox<2> boundingBox, double *gridData, int *coords, int regionSize)
{
    int regionIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (regionIndex >= regionSize) {
        return;
    }

    int x = coords[regionIndex + 0 * regionSize] - boundingBox.origin.x();
    int y = coords[regionIndex + 1 * regionSize] - boundingBox.origin.y();
    int gridIndex = y * boundingBox.dimensions.x() + x;

    gridData[gridIndex] = regionIndex + 0.123;
}

__global__
void setIDs3D(CoordBox<3> boundingBox, double *gridData, int *coords, int regionSize)
{
    int regionIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (regionIndex >= regionSize) {
        return;
    }

    int x = coords[regionIndex + 0 * regionSize] - boundingBox.origin.x();
    int y = coords[regionIndex + 1 * regionSize] - boundingBox.origin.y();
    int z = coords[regionIndex + 2 * regionSize] - boundingBox.origin.z();
    int gridIndex =
        z * boundingBox.dimensions.x() * boundingBox.dimensions.y() +
        y * boundingBox.dimensions.x() +
        x;

    gridData[gridIndex] = regionIndex + 0.666;
}

class CUDARegionTest : public CxxTest::TestSuite
{
public:

    void test2D()
    {
        CoordBox<2> box(Coord<2>(10, 40), Coord<2>(90, 60));
        Region<2> gridRegion;
        gridRegion << box;

        Region<2> region;
        region << Streak<2>(Coord<2>(10, 50),  50)
               << Streak<2>(Coord<2>(15, 51),  25)
               << Streak<2>(Coord<2>(20, 53), 100);
        CUDARegion<2> cudaRegion(region);

        CUDAGrid<double> deviceGrid(box);
        DisplacedGrid<double> hostGrid(box);

        dim3 gridDim(10);
        dim3 blockDim(32);
        setIDs2D<<<gridDim, blockDim>>>(
            deviceGrid.boundingBox(), deviceGrid.data(),
            cudaRegion.data(), region.size());
        deviceGrid.saveRegion(&hostGrid, gridRegion);

        int counter = 0;
        for (Region<2>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            double expected = counter++ + 0.123;
            TS_ASSERT_EQUALS(expected, hostGrid[*i]);
        }
    }

    void test3D()
    {
        CoordBox<3> box(Coord<3>(10, 40, 20), Coord<3>(90, 60, 80));
        Region<3> gridRegion;
        gridRegion << box;

        Region<3> region;
        region << Streak<3>(Coord<3>(10, 50, 20),  50)
               << Streak<3>(Coord<3>(15, 51, 40),  25)
               << Streak<3>(Coord<3>(15, 52, 60), 100)
               << Streak<3>(Coord<3>(20, 53, 99), 100);
        CUDARegion<3> cudaRegion(region);

        typedef Topologies::Cube<3>::Topology Topology;
        CUDAGrid<double, Topology> deviceGrid(box);
        DisplacedGrid<double, Topology> hostGrid(box);

        dim3 gridDim(10);
        dim3 blockDim(32);
        setIDs3D<<<gridDim, blockDim>>>(
            deviceGrid.boundingBox(), deviceGrid.data(),
            cudaRegion.data(), region.size());
        deviceGrid.saveRegion(&hostGrid, gridRegion);

        int counter = 0;
        for (Region<3>::Iterator i = region.begin();
             i != region.end();
             ++i) {
            double expected = counter++ + 0.666;
            TS_ASSERT_EQUALS(expected, hostGrid[*i]);
        }
    }

};

}
