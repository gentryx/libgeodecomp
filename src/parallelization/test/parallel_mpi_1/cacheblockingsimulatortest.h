#include <cxxtest/TestSuite.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CacheBlockingSimulatorTest : public CxxTest::TestSuite 
{
public:
    typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Cube<3>::Topology> MyTestCell;

    void setUp()
    {
    }

    void tearDown()
    {
        delete sim;
    }

    void testPipelinedUpdate()
    {
        init(Coord<3>(40, 1, 20));

        sim->buffer.atEdge() = sim->curGrid->atEdge();
        sim->buffer.fill(sim->buffer.boundingBox(), sim->curGrid->getEdgeCell());
        sim->fixBufferOrigin(Coord<2>(0, 0));
        int i = 0;

        for (; i < 2 * pipelineLength - 2; ++i) {
            // printState(sim);
            int lastStage = (i >> 1) + 1;
            sim->pipelinedUpdate(Coord<2>(0, 0), i, i, 0, lastStage);
        } 

        for (; i < dim.z(); ++i) {
            // printState(sim);
            sim->pipelinedUpdate(Coord<2>(0, 0), i, i, 0, pipelineLength);
        }
        for (; i < (dim.z() + 2 * pipelineLength - 2); ++i) {
            // printState(sim);
            int firstStage = (i - dim.z() + 1) >> 1 ;
            sim->pipelinedUpdate(Coord<2>(0, 0), i, i, firstStage, pipelineLength);            
        }
    }

    void testUpdateWavefront()
    {
        init(Coord<3>(40, 30, 20));

        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 0);
        sim->updateWavefront(Coord<2>(0, 0));
        sim->updateWavefront(Coord<2>(1, 0));
        sim->updateWavefront(Coord<2>(2, 0));

        sim->updateWavefront(Coord<2>(0, 1));
        sim->updateWavefront(Coord<2>(1, 1));
        sim->updateWavefront(Coord<2>(2, 1));

        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->newGrid), pipelineLength);
    }

    void testHop()
    {
        init(Coord<3>(40, 30, 20));

        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  0);
        sim->hop();
        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  5);
        sim->hop();
        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 10);
        sim->hop();
        TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 15);
    }
    
private:
    int pipelineLength;
    Coord<3> dim;
    CacheBlockingSimulator<MyTestCell> *sim;

    void init(const Coord<3> gridDim)
    {
        pipelineLength = 5;
        dim = gridDim;
        sim = new CacheBlockingSimulator<MyTestCell>(
            new TestInitializer<MyTestCell>(
                dim, 
                10000,
                0),
            pipelineLength, 
            Coord<2>(16, 16));
    }

    void printState()
    {
        std::cout << "curGrid:\n";
        printGrid(*(sim->curGrid));
        std::cout << "buffer:\n";
        printGrid(sim->buffer);
        std::cout << "newGrid:\n";
        printGrid(*(sim->newGrid));
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
