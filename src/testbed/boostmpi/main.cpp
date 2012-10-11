#include <boost/mpi.hpp>
#include <iostream>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <list>
#include <stdexcept>

using namespace LibGeoDecomp;

class DemoCell
{
public:
    bool operator==(const DemoCell& other) const
    {
        return cargo == other.cargo;
    }

    std::list<int> cargo;
};

void initGrid(Grid<DemoCell> *grid, int startY, int endY)
{
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < grid->getDimensions().x(); ++x) {
            int max = y * grid->getDimensions().x() + x;
            for (int i = 0; i < max; ++i) {
                (*grid)[Coord<2>(x, y)].cargo.push_back(i);
            }
        }
    }
 }

int main(int argc, char **argv)
{
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    MPILayer mpiLayer;

    if (world.size() != 2) {
        std::cerr << "needs to be run with 2 processes\n";
        return 1;
    }

    int myStartY = 0;
    int myEndY = 5;
    if (world.rank() == 1) {
        myStartY += 5;
        myEndY += 5;
    }

    int other = (world.rank()? 0 : 1);
    
    Grid<DemoCell> grid(Coord<2>(20, 10));
    Grid<DemoCell> expected(Coord<2>(20, 10));
    initGrid(&grid, myStartY, myEndY);
    initGrid(&expected, 0, 10);

    Region<2> sendRegion;
    Region<2> recvRegion;
    for (int y = myStartY; y < myEndY; ++y) {
        sendRegion << Streak<2>(Coord<2>(0, y), 20);
    }
    
    if (world.rank() == 0) {
        mpiLayer.sendRegion(sendRegion,  other);
        mpiLayer.recvRegion(&recvRegion, other);
    } else {
        mpiLayer.recvRegion(&recvRegion, other);
        mpiLayer.sendRegion(sendRegion,  other);
    }
    
    // mpiLayer.sendUnregisteredRegion(&grid, sendRegion, other, 0, fixme);
    // mpiLayer.sendUnregisteredRegion(&grid, recvRegion, other, 0, fixme);
    // mpiLayer.waitAll();
    
    if (grid == expected) {
        std::cout << "rank " << world.rank() << " is good\n";
    } else {
        throw std::logic_error("uh oh!");
    }

    return 0;
}
