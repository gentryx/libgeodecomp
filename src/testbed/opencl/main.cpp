#include <iostream>
#include <libgeodecomp/libgeodecomp.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    typedef Stencils::VonNeumann<3, 1> Stencil;

    class API :
        public CellAPITraits::Fixed,
        public CellAPITraitsFixme::HasFixedCoordsOnlyUpdate,
        public CellAPITraitsFixme::HasStencil<Stencils::VonNeumann<3, 1> >,
        public CellAPITraitsFixme::HasCubeTopology<3>
    {};

    static inline unsigned nanoSteps()
    {
        return 1;
    }

    inline explicit Cell(const double& v=0) : temp(v)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[FixedCoord< 0,  0, -1>()].temp +
                neighborhood[FixedCoord< 0, -1,  0>()].temp +
                neighborhood[FixedCoord<-1,  0,  0>()].temp +
                neighborhood[FixedCoord< 1,  0,  0>()].temp +
                neighborhood[FixedCoord< 0,  1,  0>()].temp +
                neighborhood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 6.0);
    }

    double temp;
};

template<typename CELL>
class MyFutureOpenCLStepper
{
public:
    typedef typename CellAPITraitsFixme::SelectTopology<CELL>::Value Topology;
    typedef DisplacedGrid<CELL, Topology>  GridType;
    const static int DIM = Topology::DIM;

    MyFutureOpenCLStepper(const CoordBox<DIM> box) :
        box(box),
        hostGrid(box)
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
    CoordBox<DIM> box;
    GridType hostGrid;
    // fixme deviceGridOld;
    // fixme deviceGridNew;
};

int main(int argc, char **argv)
{
    MyFutureOpenCLStepper<Cell> stepper(
        CoordBox<3>(
            Coord<3>(10, 10, 10),
            Coord<3>(20, 30, 40)));
    std::cout << "test: " << sizeof(stepper) << "\n";
    return 0;
}
