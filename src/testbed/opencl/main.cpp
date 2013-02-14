#include <iostream>
#include <libgeodecomp/libgeodecomp.h>

using namespace LibGeoDecomp;

template<typename CELL>
class MyFutureOpenCLStepper
{
public:
    const static int DIM = CELL::Topology::DIMENSIONS;

    typedef typename CELL::Topology Topology;
    typedef DisplacedGrid<CELL, Topology>  GridType;

    MyFutureOpenCLStepper(const Coord<DIM> dim) :
        dim(dim),
        hostGrid(dim)
    {
        // todo: allocate deviceGridOld, deviceGridNew via OpenCL on device
        // todo: specify OpenCL platform, device via constructor
    }

    void regionToVec(const Region<DIM>& region, SuperVector<int> *coordsX, SuperVector<int> *coordsY, SuperVector<int> *coordsZ)
    {
        // todo: iterate through region and add all coordinates to the corresponding vectors
    }

    template<typename GRID>
    void setGridRegion(const GRID& grid, const Region<DIM>& region)
    {
        // todo: copy all coords in region from grid (on host) to deviceGridOld
    }

    template<typename GRID>
    void getGridRegion(const GRID *grid, const Region<DIM>& region)
    {
        // todo: copy all coords in region from deviceGridOld (on host) to grid
    }

private:
    Coord<DIM> dim;
    GridType hostGrid;
    // fixme deviceGridOld;
    // fixme deviceGridNew;
};

int main(int argc, char **argv)
{

    return 0;
}
