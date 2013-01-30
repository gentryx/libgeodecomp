#include <cxxtest/TestSuite.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CacheBlockingSimulatorTest : public CxxTest::TestSuite 
{
public:
    typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Cube<3>::Topology> MyTestCell;

    void testBasic()
    {
        
        CacheBlockingSimulator<MyTestCell> sim(
            new TestInitializer<MyTestCell>(
                Coord<3>(40, 1, 10), 
                10000,
                0),
            5, 
            Coord<2>(16, 16));
        std::string buf;

        sim.buffer.atEdge() = sim.curGrid->atEdge();
        sim.buffer.fill(sim.buffer.boundingBox(), sim.curGrid->getEdgeCell());

        printState(sim);
        sim.pipelinedUpdate(Coord<2>(0, 0), 0, 0, 0, 1);
        sim.pipelinedUpdate(Coord<2>(0, 0), 1, 1, 0, 1);

        sim.pipelinedUpdate(Coord<2>(0, 0), 2, 2, 0, 2);
        sim.pipelinedUpdate(Coord<2>(0, 0), 3, 3, 0, 2);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 4, 4, 0, 3);
        sim.pipelinedUpdate(Coord<2>(0, 0), 5, 5, 0, 3);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 6, 6, 0, 4);
        sim.pipelinedUpdate(Coord<2>(0, 0), 7, 7, 0, 4);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 8, 8, 0, 5);
        sim.pipelinedUpdate(Coord<2>(0, 0), 9, 9, 0, 5);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 10, 10, 0, 5);
        printState(sim);
        sim.pipelinedUpdate(Coord<2>(0, 0), 11, 11, 1, 5);
        printState(sim);
        sim.pipelinedUpdate(Coord<2>(0, 0), 12, 12, 1, 5);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 13, 13, 2, 5);
        printState(sim);
        sim.pipelinedUpdate(Coord<2>(0, 0), 14, 14, 2, 5);
        printState(sim);


        sim.pipelinedUpdate(Coord<2>(0, 0), 15, 15, 3, 5);
        printState(sim);
        sim.pipelinedUpdate(Coord<2>(0, 0), 16, 16, 3, 5);
        printState(sim);

        sim.pipelinedUpdate(Coord<2>(0, 0), 17, 17, 4, 5);
        printState(sim);

        // sim.pipelinedUpdate(Coord<2>(0, 0), 12, 12, 2, 5);
        // printState(sim);

        // for (int z = 0; z < 20; ++z) {
        //     int lastStage = std::min(5, z);
        //     sim.pipelinedUpdate(Coord<2>(1, 1), z, z, 0, lastStage);
        // }
        // // std::cin >> buf;
    }


    void printState(const CacheBlockingSimulator<MyTestCell>& sim)
    {
        return;

        std::cout << "curGrid:\n";
        printGrid(*sim.curGrid);
        std::cout << "buffer:\n";
        printGrid(sim.buffer);
        std::cout << "newGrid:\n";
        printGrid(*sim.newGrid);
        std::cout << "\n";
    }

    template<typename GRID>
    void printGrid(GRID grid)
    {
        Coord<3> dim = grid.getDimensions();
        for (int z = 0; z < dim.z(); ++z) {
            std::cout << std::setw(3) << z << " " 
                      << std::setw(3) << grid[Coord<3>(5, 0, z)].pos.z() << "  ";
            for (int x = 0; x < dim.x(); ++x) {        
                MyTestCell c = grid[Coord<3>(x, 0, z)];
                if (c.isEdgeCell) {
                    std::cout << "X"; 
                } else {
                    std::cout << c.cycleCounter;
                }
            }
            std::cout << "\n";
        }
    }
};

}
