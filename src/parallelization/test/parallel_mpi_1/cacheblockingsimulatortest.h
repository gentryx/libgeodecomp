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
        // sim.reset();
    }

    void testPipelinedUpdate()
    {
//         init(Coord<3>(40, 1, 20));

//         sim->buffer.atEdge() = sim->curGrid->atEdge();
//         sim->buffer.fill(sim->buffer.boundingBox(), sim->curGrid->getEdgeCell());
//         sim->fixBufferOrigin(Coord<2>(0, 0));
//         int i = 0;

//         for (; i < 2 * pipelineLength - 2; ++i) {
//             int lastStage = (i >> 1) + 1;
//             sim->pipelinedUpdate(Coord<2>(0, 0), i, i, 0, lastStage);
//         }
//         for (; i < dim.z(); ++i) {
//             sim->pipelinedUpdate(Coord<2>(0, 0), i, i, 0, pipelineLength);
//         }
//         for (; i < (dim.z() + 2 * pipelineLength - 2); ++i) {
//             int firstStage = ((i - dim.z() + 1) >> 1);
//             sim->pipelinedUpdate(Coord<2>(0, 0), i, i, firstStage, pipelineLength);            
//         }
//     }

//     void testUpdateWavefront()
//     {
//         init(Coord<3>(40, 30, 20));

//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 0);
//         sim->updateWavefront(Coord<2>(0, 0));
//         sim->updateWavefront(Coord<2>(1, 0));
//         sim->updateWavefront(Coord<2>(2, 0));

//         sim->updateWavefront(Coord<2>(0, 1));
//         sim->updateWavefront(Coord<2>(1, 1));
//         sim->updateWavefront(Coord<2>(2, 1));

//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->newGrid), pipelineLength);
//     }

//     void testHop1()
//     {
//         init(Coord<3>(40, 30, 20));

//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  0);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  5);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 10);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 15);
//     }

//     void testHop2()
//     {
//         init(Coord<3>(40, 30, 20), 7);

//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  0);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  7);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 14);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid), 21);
//     }

//     void testHop3()
//     {
//         init(Coord<3>(40, 30, 20), 1);

//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  0);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  1);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  2);
//         sim->hop();
//         TS_ASSERT_TEST_GRID(CacheBlockingSimulator<MyTestCell>::GridType, *(sim->curGrid),  3);
//     }
    
// private:
//     int pipelineLength;
//     Coord<3> dim;
//     boost::shared_ptr<CacheBlockingSimulator<MyTestCell> > sim;

//     void init(const Coord<3>& gridDim, int newPipelineLength = 5, const Coord<2> wavefrontDim = Coord<2>(16, 16))
//     {
//         pipelineLength = newPipelineLength;
//         dim = gridDim;
//         sim.reset(new CacheBlockingSimulator<MyTestCell>(
//                       new TestInitializer<MyTestCell>(
//                           dim, 
//                           10000,
//                           0),
//                       pipelineLength, 
//                       wavefrontDim));
    }
};

}
